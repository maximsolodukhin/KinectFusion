#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_RAYCASTING_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_RAYCASTING_HPP

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <kinectfusion/cuda_compat.hpp>
#include <kinectfusion/empty_space_skip.hpp>
#include <kinectfusion/grid.hpp>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/occupancy.hpp>
#include <kinectfusion/rgbd.hpp>
#include <kinectfusion/trilinear.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <kinectfusion/volume_sampler.hpp>
#include <optional>
#include <type_traits>

namespace kinectfusion {

enum class RaycastBackend : std::uint8_t { kMarch, kBitmapMarch, kBandMarch };

// Owns the empty-space acceleration structure of one backend, or nothing for
// the plain march. `visit` hands the render callback the matching skip
// policy.
template <MemorySpace Space>
class EmptySpaceIndex {
 public:
  EmptySpaceIndex() = default;

  EmptySpaceIndex(RaycastBackend backend, const VolumeGeometry& geometry) {
    if (backend == RaycastBackend::kBitmapMarch) {
      bitmap_.emplace(geometry);
    } else if (backend == RaycastBackend::kBandMarch) {
      band_.emplace(geometry);
    }
  }

  template <DenseVoxelGridView VolumeViewT>
  void rebuild(const VolumeViewT& volume) {
    if (bitmap_) {
      bitmap_->rebuild(volume);
    }
    if (band_) {
      band_->rebuild(volume);
    }
  }

  template <typename Render>
  [[nodiscard]] decltype(auto) visit(const Render& render) const {
    if (bitmap_) {
      return render(BitmapSkip{.occupancy = bitmap_->view()});
    }
    if (band_) {
      return render(BandSkip{.band = band_->view()});
    }
    return render(NoSkip{});
  }

 private:
  std::optional<BasicOccupancyBitmap<Space>> bitmap_;
  std::optional<BasicBandBitmap<Space>> band_;
};

struct RaycastOptions {
  float min_depth{kDefaultMinDepthMeters};
  float max_depth{kDefaultMaxDepthMeters};
  float step_scale{1.0F};
  CornerPolicy tsdf_corner_policy{CornerPolicy::kSkipMissing};
  // Ablation: compute normals from the 8 corners of the final sample (one
  // gather), not from six extra trilinear samples (48 gathers).
  bool cell_gradient_normals{false};
  // Ablation: start each ray just in front of the previous surface at the
  // same pixel, not at min_depth. The host path re-allocates its maps and
  // thus falls back to min_depth.
  bool seed_from_previous{false};
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

// Marches a ray per pixel and writes the surface samples it finds. Generic
// over the sampler, so a different volume representation swaps in behind
// TsdfVolumeSampler without touching the marcher.
template <MemorySpace Space = MemorySpace::kHost,
          TsdfVolumeSampler Sampler = VolumeSampler<Space>,
          SkipPolicy Skip = NoSkip>
class SurfaceRaycast {
 public:
  SurfaceRaycast(const Sampler& sampler, const RaycastOptions& options,
                 const RigidTransform& pose, const CameraIntrinsics& intrinsics,
                 const Skip& skip = {})
      : sampler_(sampler),
        options_(options),
        pose_(pose),
        intrinsics_(intrinsics),
        skip_(skip) {}

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
    // Read the previous surface before this pixel is invalidated. Each
    // thread only touches its own pixel, so the in-place read is safe.
    float seed = options_.min_depth;
    if (options_.seed_from_previous) {
      const Vec3f previous_point = maps.points.at(col, row);
      // Zero is the freshly-allocated map fill, not a surface point.
      if (all_finite(previous_point) && squared_norm(previous_point) > 0.0F) {
        const float previous_depth = norm(previous_point - pose_.translation);
        const float truncated =
            previous_depth -
            kSeedMarginTruncations * sampler_.truncation_distance();

        seed = compat::max(options_.min_depth, truncated);
      }
    }
    maps.points.at(col, row) = invalid_vec3f();
    maps.normals.at(col, row) = invalid_vec3f();
    maps.colors.at(col, row) = 0;

    auto pixel = Pixel{.x = col, .y = row}.as_vector();
    auto translated = intrinsics_.back_project(pixel, 1.0F);

    const Vec3f direction = pose_.rotation * translated;
    const float direction_norm = norm(direction);

    if (direction_norm <= 0.0F) {
      return;
    }

    const auto surface = find_zero_crossing(direction / direction_norm, seed);
    if (surface) {
      write_surface_sample(maps, col, row, *surface);
    }
  }

 private:
  // Two truncation bands of slack keep the seed on the positive side under
  // inter-frame motion.
  static constexpr float kSeedMarginTruncations = 2.0F;

  // Returns the interpolated zero crossing where the ray first passes from
  // front to back; gives up at max_depth or when leaving a surface from
  // behind.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE compat::optional<Vec3f>
  find_zero_crossing(const Vec3f& direction, float seed) const {
    struct Sample {
      float tsdf{};
      Vec3f point{};
    };

    const float base_step = sampler_.voxel_size() * options_.step_scale;
    compat::optional<Sample> previous;
    float ray_length = seed;
    while (ray_length <= options_.max_depth) {
      const float advanced =
          skip_.advance(pose_.translation, direction, ray_length, base_step,
                        options_.max_depth);
      if (advanced != ray_length) {
        previous.reset();  // the skipped samples were all nullopt
        ray_length = advanced;
        if (ray_length > options_.max_depth) {
          break;
        }
      }

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
      const float precomp_step =
          *tsdf * sampler_.truncation_distance() * options_.step_scale;

      auto step = compat::max(base_step, precomp_step);
      ray_length += *tsdf > 0.0F ? step : base_step;
    }
    return compat::nullopt;
  }

  // Modifies in-place, can't be changed as a return value because the maps are
  // a view into a preallocated buffer.
  KINECTFUSION_HOST_DEVICE void write_surface_sample(
      SurfaceMapsView<Space> maps, std::size_t col, std::size_t row,
      const Vec3f& surface) const {
    compat::optional<Vec3f> normal;
    if constexpr (requires { sampler_.cell_normal(surface); }) {
      normal = options_.cell_gradient_normals
                   ? sampler_.cell_normal(surface)
                   : sampler_.normal(surface, options_.tsdf_corner_policy);
    } else {
      normal = sampler_.normal(surface, options_.tsdf_corner_policy);
    }
    if (!normal) {
      return;
    }
    maps.points.at(col, row) = surface;
    maps.normals.at(col, row) = *normal;
    if constexpr (Sampler::kHasColor) {
      if (const auto color = sampler_.color(surface)) {
        maps.colors.at(col, row) = pixel_from_color(*color);
      }
    } else {
      // Without color, render headlight-shaded geometry to keep the raycast
      // images inspectable.
      const Vec3f view = normalized(pose_.translation - surface);
      const float intensity =
          kMaxColorChannelValueF * compat::max(0.0F, dot(*normal, view));

      maps.colors.at(col, row) =
          pixel_from_color(make_vec3f(intensity, intensity, intensity));
    }
  }

  Sampler sampler_;
  RaycastOptions options_;
  RigidTransform pose_;
  CameraIntrinsics intrinsics_;
  Skip skip_;
};

using HostSurfaceRaycast = SurfaceRaycast<MemorySpace::kHost>;
using DeviceSurfaceRaycast = SurfaceRaycast<MemorySpace::kDevice>;

// What a raycast sweep launches per pixel. The renderer crosses into the
// kernel by value, so it must be trivially copyable.
template <typename R, MemorySpace Space>
concept PixelRenderer =
    std::is_trivially_copyable_v<R> &&
    requires(const R renderer, SurfaceMapsView<Space> maps, std::size_t index) {
      renderer.render_pixel(maps, index, index);
    };

static_assert(PixelRenderer<HostSurfaceRaycast, MemorySpace::kHost>);

// Host driver, renders the zero-crossing surface of a volume from a camera.
class Raycaster {
 public:
  // Throws std::invalid_argument
  explicit Raycaster(RaycastOptions options = {});

  // Non-finite points mark pixels where no surface was hit.
  [[nodiscard]] SurfaceMaps raycast(ConstHostVolumeView volume,
                                    const RaycastCamera& camera) const;

  // The sampler-generic host march. `skip` selects the empty-space policy.
  template <TsdfVolumeSampler Sampler, SkipPolicy Skip = NoSkip>
  [[nodiscard]] SurfaceMaps render(const Sampler& sampler,
                                   const RaycastCamera& camera,
                                   const Skip& skip = {}) const {
    validate_camera(camera);
    using Vec3fImg = image_proc::Vector3fImage;
    using ColorImg = image_proc::ColorImage;
    SurfaceMaps maps{
        .points = Vec3fImg{camera.width, camera.height, invalid_vec3f()},
        .normals = Vec3fImg{camera.width, camera.height, invalid_vec3f()},
        .colors = ColorImg{camera.width, camera.height}};

    const SurfaceRaycast<MemorySpace::kHost, Sampler, Skip> raycast{
        sampler, options_, camera.pose(), camera.intrinsics, skip};
    const HostSurfaceMapsView maps_view = view(maps);

    for (const auto [col, row] : PixelIndices{camera.width, camera.height}) {
      raycast.render_pixel(maps_view, col, row);
    }

    return maps;
  }

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
