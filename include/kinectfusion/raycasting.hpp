#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_RAYCASTING_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_RAYCASTING_HPP

#include <Eigen/Core>
#include <cstddef>
#include <kinectfusion/cuda_compat.hpp>
#include <kinectfusion/grid.hpp>
#include <kinectfusion/rgbd.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <kinectfusion/volume_sampler.hpp>

namespace kinectfusion {

struct RaycastOptions {
  float min_depth{kDefaultMinDepthMeters};
  float max_depth{kDefaultMaxDepthMeters};
  float step_scale{1.0F};
  CornerPolicy tsdf_corner_policy{CornerPolicy::kSkipMissing};
};

struct RaycastCamera {
  CameraIntrinsics intrinsics{};
  std::size_t width{};
  std::size_t height{};
  Eigen::Matrix4f camera_to_world{Eigen::Matrix4f::Identity()};

  [[nodiscard]] RigidTransform pose() const {
    return from_eigen(camera_to_world);
  }
};

// Marches a ray per pixel and writes the surface samples it finds.
template <MemorySpace Space = MemorySpace::kHost>
class SurfaceRaycast {
 public:
  SurfaceRaycast(const VolumeSampler<Space>& sampler,
                 const RaycastOptions& options, const RigidTransform& pose,
                 const CameraIntrinsics& intrinsics)
      : sampler_(sampler),
        options_(options),
        pose_(pose),
        intrinsics_(intrinsics) {}

  // Builds a raycast of `volume` as seen from `camera`.
  [[nodiscard]] static SurfaceRaycast from_camera(
      const ConstVolumeView<Space>& volume, const RaycastOptions& options,
      const RaycastCamera& camera) {
    return {VolumeSampler<Space>{volume}, options, camera.pose(),
            camera.intrinsics};
  }

  KINECTFUSION_HOST_DEVICE void render_pixel(SurfaceMapsView<Space> maps,
                                             std::size_t col,
                                             std::size_t row) const {
    auto pixel = Pixel{.x = col, .y = row}.as_vector();
    auto translated = intrinsics_.back_project(pixel, 1.0F);

    const Vec3f direction = pose_.rotation * translated;
    const float direction_norm = norm(direction);

    if (direction_norm <= 0.0F) {
      return;
    }

    const auto surface = find_zero_crossing(direction / direction_norm);
    if (surface) {
      write_surface_sample(maps, col, row, *surface);
    }
  }

 private:
  // Returns the interpolated zero crossing where the ray first passes from
  // front to back; gives up at max_depth or when leaving a surface from
  // behind.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE compat::optional<Vec3f>
  find_zero_crossing(const Vec3f& direction) const {
    struct Sample {
      float tsdf{};
      Vec3f point{};
    };

    const float base_step = sampler_.voxel_size() * options_.step_scale;
    compat::optional<Sample> previous;
    float ray_length = options_.min_depth;
    while (ray_length <= options_.max_depth) {
      const Vec3f point = pose_.translation + (ray_length * direction);
      const auto tsdf = sampler_.tsdf(point, options_.tsdf_corner_policy);
      if (!tsdf) {
        previous.reset();
        ray_length += base_step;
        continue;
      }
      if (previous && previous->tsdf > 0.0F && *tsdf <= 0.0F) {
        const float gap = previous->tsdf - *tsdf;
        const float alpha = gap != 0.0F ? previous->tsdf / gap : 0.0F;
        return previous->point + (alpha * (point - previous->point));
      }
      if (previous && previous->tsdf < 0.0F && *tsdf > 0.0F) {
        return compat::nullopt;
      }
      previous = Sample{.tsdf = *tsdf, .point = point};
      auto step =
          compat::max(base_step, *tsdf * sampler_.truncation_distance() *
                                     options_.step_scale);
      ray_length += *tsdf > 0.0F ? step : base_step;
    }
    return compat::nullopt;
  }

  // Modifies in-place, can't be changed as a return value because the maps are
  // a view into a preallocated buffer.
  KINECTFUSION_HOST_DEVICE void write_surface_sample(
      SurfaceMapsView<Space> maps, std::size_t col, std::size_t row,
      const Vec3f& surface) const {
    const auto normal = sampler_.normal(surface, options_.tsdf_corner_policy);
    if (!normal) {
      return;
    }
    maps.points.at(col, row) = surface;
    maps.normals.at(col, row) = *normal;
    if (const auto color = sampler_.color(surface)) {
      maps.colors.at(col, row) = pixel_from_color(*color);
    }
  }

  VolumeSampler<Space> sampler_;
  RaycastOptions options_;
  RigidTransform pose_;
  CameraIntrinsics intrinsics_;
};

using HostSurfaceRaycast = SurfaceRaycast<MemorySpace::kHost>;
using DeviceSurfaceRaycast = SurfaceRaycast<MemorySpace::kDevice>;

// Host driver, renders the zero-crossing surface of a volume from a camera.
class Raycaster {
 public:
  // Throws std::invalid_argument
  explicit Raycaster(RaycastOptions options = {});

  // Non-finite points mark pixels where no surface was hit.
  [[nodiscard]] SurfaceMaps raycast(ConstHostVolumeView volume,
                                    const RaycastCamera& camera) const;

  // The validated configuration; space-specific drivers launch from these.
  [[nodiscard]] const RaycastOptions& options() const { return options_; }

  // Throws std::invalid_argument
  static void validate_camera(const RaycastCamera& camera);

 private:
  [[nodiscard]] static RaycastOptions validated(RaycastOptions options);

  RaycastOptions options_;
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_RAYCASTING_HPP */
