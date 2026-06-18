#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/util.hpp>

namespace kinectfusion {

inline constexpr float kMaxWeight = 196.0F;
inline constexpr float kObservationWeight = 1.0F;
inline constexpr float kRaycastStepScale = 1.0F;

struct Voxel {
  float distance{1.0F};
  float weight{0.0F};
};

struct ColorVoxel {
  Eigen::Vector3f color{Eigen::Vector3f::Zero()};
  float weight{0.0F};
};

struct SurfaceMaps {
  image_proc::Image<Eigen::Vector3f> points;
  image_proc::Image<Eigen::Vector3f> normals;
  image_proc::ColorImage colors;
};

class Volume {
 public:
  Volume(const Eigen::Vector3i& resolution, float voxel_size,
         const Eigen::Vector3f& origin, float truncation_distance)
      : resolution_(resolution),
        voxel_size_(voxel_size),
        origin_(origin),
        truncation_distance_(truncation_distance),
        voxels_(voxel_count()),
        colors_(voxel_count()) {}

  [[nodiscard]] Eigen::Vector3i resolution() const { return resolution_; }
  [[nodiscard]] float voxel_size() const { return voxel_size_; }

  [[nodiscard]] std::size_t observed_voxel_count() const {
    std::size_t count = 0;
    for (const Voxel& voxel : voxels_) {
      if (voxel.weight > 0.0F) {
        ++count;
      }
    }
    return count;
  }

  // Fuses a depth (and optional color) frame into the volume. world_to_camera
  // maps world points into the camera frame; optional live normals enable
  // view-angle observation weighting.
  void integrate_depth_image(
      const image_proc::DepthImage& depth_image,
      const CameraIntrinsics& intrinsics,
      const Eigen::Matrix4f& world_to_camera,
      const image_proc::ColorImage* color_image = nullptr,
      const image_proc::Image<Eigen::Vector3f>* normals = nullptr);

  // Renders the zero-crossing surface seen from camera_to_world into per-pixel
  // point, normal and color maps (non-finite points where no surface is hit).
  [[nodiscard]] SurfaceMaps raycast(const CameraIntrinsics& intrinsics,
                                    unsigned int width, unsigned int height,
                                    const Eigen::Matrix4f& camera_to_world) const;

 private:
  struct GridSample {
    int base_x{};
    int base_y{};
    int base_z{};
    float tx{};
    float ty{};
    float tz{};
  };

  [[nodiscard]] std::size_t voxel_count() const {
    return static_cast<std::size_t>(resolution_.x()) *
           static_cast<std::size_t>(resolution_.y()) *
           static_cast<std::size_t>(resolution_.z());
  }

  [[nodiscard]] std::size_t index(int x, int y, int z) const {
    return (static_cast<std::size_t>(z) *
                static_cast<std::size_t>(resolution_.y()) +
            static_cast<std::size_t>(y)) *
               static_cast<std::size_t>(resolution_.x()) +
           static_cast<std::size_t>(x);
  }

  [[nodiscard]] bool in_bounds(int x, int y, int z) const {
    return x >= 0 && y >= 0 && z >= 0 && x < resolution_.x() &&
           y < resolution_.y() && z < resolution_.z();
  }

  [[nodiscard]] bool contains(const Eigen::Vector3f& point) const {
    const Eigen::Vector3f local = (point - origin_) / voxel_size_;
    return local.x() >= 0.0F && local.y() >= 0.0F && local.z() >= 0.0F &&
           local.x() < static_cast<float>(resolution_.x()) &&
           local.y() < static_cast<float>(resolution_.y()) &&
           local.z() < static_cast<float>(resolution_.z());
  }

  [[nodiscard]] Voxel& at(int x, int y, int z) {
    return voxels_[index(x, y, z)];
  }
  [[nodiscard]] const Voxel& at(int x, int y, int z) const {
    return voxels_[index(x, y, z)];
  }
  [[nodiscard]] ColorVoxel& color_at(int x, int y, int z) {
    return colors_[index(x, y, z)];
  }
  [[nodiscard]] const ColorVoxel& color_at(int x, int y, int z) const {
    return colors_[index(x, y, z)];
  }

  [[nodiscard]] GridSample grid_sample(const Eigen::Vector3f& point) const;

  [[nodiscard]] static float trilinear_weight(const GridSample& sample, int dx,
                                              int dy, int dz);

  [[nodiscard]] bool sample_tsdf(const Eigen::Vector3f& point,
                                 float& distance) const;

  [[nodiscard]] bool sample_color(const Eigen::Vector3f& point,
                                  Eigen::Vector3f& color) const;

  [[nodiscard]] bool sample_normal(const Eigen::Vector3f& point,
                                   Eigen::Vector3f& normal) const;

  [[nodiscard]] static std::uint32_t pixel_from_color(
      const Eigen::Vector3f& color);

  Eigen::Vector3i resolution_;
  float voxel_size_{};
  Eigen::Vector3f origin_;
  float truncation_distance_{};
  std::vector<Voxel> voxels_;
  std::vector<ColorVoxel> colors_;
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_HPP */
