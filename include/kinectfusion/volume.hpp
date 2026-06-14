#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/util.hpp>

namespace kinectfusion {

struct Voxel {
  float distance{1.0F};
  float weight{0.0F};
};

struct TsdfIntegrationOptions {
  float depth_scale{kTumDepthScale};
  float observation_weight{1.0F};
  float max_weight{100.0F};
  float min_depth{0.2F};
  float max_depth{5.0F};
  bool projective_distance{true};
  bool distance_scaled_truncation{true};
  float truncation_distance_scale{0.01F};
  bool view_angle_weighting{true};
};

struct ColorVoxel {
  float r{};
  float g{};
  float b{};
  float weight{};

  [[nodiscard]] ColorRgba rgba() const {
    return ColorRgba{static_cast<std::uint8_t>(std::clamp(r, 0.0F, 255.0F)),
                     static_cast<std::uint8_t>(std::clamp(g, 0.0F, 255.0F)),
                     static_cast<std::uint8_t>(std::clamp(b, 0.0F, 255.0F)),
                     255};
  }
};

struct RaycastOptions {
  CameraIntrinsics intrinsics{};
  unsigned int width{};
  unsigned int height{};
  Eigen::Matrix4f camera_to_world{Eigen::Matrix4f::Identity()};
  float min_depth{0.2F};
  float max_depth{5.0F};
  float step_scale{1.0F};
  bool missing_tsdf_as_zero{false};
};

struct SurfaceMaps {
  image_proc::Image<Eigen::Vector3f> points;
  image_proc::Image<Eigen::Vector3f> normals;
  image_proc::ColorImage colors;
};

class Volume {
 public:
  Volume(Eigen::Vector3i resolution, float voxel_size,
         Eigen::Vector3f origin = Eigen::Vector3f::Zero(), float truncation_distance = 0.04F)
      : resolution_(resolution),
        voxel_size_(voxel_size),
        origin_(origin),
        truncation_distance_(truncation_distance),
        voxels_(voxel_count(resolution)),
        colors_(voxel_count(resolution)) {
    if (resolution.x() <= 0 || resolution.y() <= 0 || resolution.z() <= 0) {
      throw std::invalid_argument("Volume resolution must be positive");
    }
    if (voxel_size <= 0.0F) {
      throw std::invalid_argument("Volume voxel size must be positive");
    }
    if (truncation_distance <= 0.0F) {
      throw std::invalid_argument("Volume truncation distance must be positive");
    }
  }

  [[nodiscard]] Eigen::Vector3i resolution() const { return resolution_; }
  [[nodiscard]] float voxel_size() const { return voxel_size_; }
  [[nodiscard]] Eigen::Vector3f origin() const { return origin_; }
  [[nodiscard]] float truncation_distance() const {
    return truncation_distance_;
  }
  [[nodiscard]] const std::vector<Voxel>& voxels() const { return voxels_; }
  [[nodiscard]] const std::vector<ColorVoxel>& colors() const {
    return colors_;
  }

  [[nodiscard]] const Voxel& at(int x, int y, int z) const {
    return voxels_[index(x, y, z)];
  }

  [[nodiscard]] Voxel& at(int x, int y, int z) {
    return voxels_[index(x, y, z)];
  }

  [[nodiscard]] const ColorVoxel& color_at(int x, int y, int z) const {
    return colors_[index(x, y, z)];
  }

  [[nodiscard]] bool contains(const Eigen::Vector3f& point) const {
    const Eigen::Vector3f local = (point - origin_) / voxel_size_;
    return local.x() >= 0.0F && local.y() >= 0.0F && local.z() >= 0.0F &&
           local.x() < static_cast<float>(resolution_.x()) &&
           local.y() < static_cast<float>(resolution_.y()) &&
           local.z() < static_cast<float>(resolution_.z());
  }

  [[nodiscard]] Eigen::Vector3f voxel_center(int x, int y, int z) const {
    return origin_ + voxel_size_ *
                         Eigen::Vector3f{static_cast<float>(x) + 0.5F,
                                  static_cast<float>(y) + 0.5F,
                                  static_cast<float>(z) + 0.5F};
  }

  void reset() {
    std::ranges::fill(voxels_, Voxel{});
    std::ranges::fill(colors_, ColorVoxel{});
  }

  void integrate_depth_image(
      const image_proc::DepthImage& depth_image,
      const CameraIntrinsics& intrinsics, const Eigen::Matrix4f& world_to_camera,
      const TsdfIntegrationOptions& options = {},
      const image_proc::ColorImage* color_image = nullptr,
      const image_proc::Image<Eigen::Vector3f>* normals = nullptr);

  [[nodiscard]] std::size_t observed_voxel_count() const {
    return static_cast<std::size_t>(
        std::ranges::count_if(voxels_, [](const Voxel& voxel) {
          return voxel.weight > 0.0F;
        }));
  }

  [[nodiscard]] SurfaceMaps raycast(const RaycastOptions& options) const;

 private:
  struct ImagePixel {
    unsigned int x{};
    unsigned int y{};
  };

  struct GridSamplePosition {
    int base_x{};
    int base_y{};
    int base_z{};
    float tx{};
    float ty{};
    float tz{};
  };

  struct RaycastContext {
    Eigen::Matrix3f rotation{};
    Eigen::Vector3f origin{};
    float base_step{};
  };

  [[nodiscard]] bool in_bounds(int x, int y, int z) const {
    return x >= 0 && y >= 0 && z >= 0 && x < resolution_.x() &&
           y < resolution_.y() && z < resolution_.z();
  }

  static std::size_t voxel_count(const Eigen::Vector3i& resolution) {
    return static_cast<std::size_t>(resolution.x()) * resolution.y() *
           resolution.z();
  }

  static void validate_options(const TsdfIntegrationOptions& options) {
    if (options.depth_scale <= 0.0F) {
      throw std::invalid_argument("Depth scale must be positive");
    }
    if (options.observation_weight <= 0.0F) {
      throw std::invalid_argument("Observation weight must be positive");
    }
    if (options.max_weight <= 0.0F) {
      throw std::invalid_argument("Max weight must be positive");
    }
    if (options.min_depth < 0.0F || options.max_depth <= options.min_depth) {
      throw std::invalid_argument("Depth range is invalid");
    }
    if (options.truncation_distance_scale < 0.0F) {
      throw std::invalid_argument("Truncation distance scale must be non-negative");
    }
  }

  [[nodiscard]] std::size_t index(int x, int y, int z) const {
    return (static_cast<std::size_t>(z) * resolution_.y() + y) *
               resolution_.x() +
           x;
  }

  void integrate_voxel(int x, int y, int z,
                       const image_proc::DepthImage& depth_image,
                       const CameraIntrinsics& intrinsics,
                       const Eigen::Matrix4f& world_to_camera,
                       const TsdfIntegrationOptions& options,
                       const image_proc::ColorImage* color_image,
                       const image_proc::Image<Eigen::Vector3f>* normals);

  void integrate_color(int x, int y, int z, std::uint32_t color,
                       float observation_weight, float max_weight) {
    auto& voxel = colors_[index(x, y, z)];
    const auto rgba = rgba_from_pixel(color);
    const float blended_weight = voxel.weight + observation_weight;
    voxel.r = ((voxel.r * voxel.weight) +
               (static_cast<float>(rgba.x()) * observation_weight)) /
              blended_weight;
    voxel.g = ((voxel.g * voxel.weight) +
               (static_cast<float>(rgba.y()) * observation_weight)) /
              blended_weight;
    voxel.b = ((voxel.b * voxel.weight) +
               (static_cast<float>(rgba.z()) * observation_weight)) /
              blended_weight;
    voxel.weight = std::min(blended_weight, max_weight);
  }

  [[nodiscard]] bool sample_tsdf(const Eigen::Vector3f& point,
                                 float& distance) const;

  [[nodiscard]] bool sample_tsdf_relaxed(const Eigen::Vector3f& point,
                                                  float& distance) const;

  [[nodiscard]] GridSamplePosition grid_sample_position(
      const Eigen::Vector3f& point) const;

  [[nodiscard]] static float trilinear_weight(
      const GridSamplePosition& position, int offset_x, int offset_y,
      int offset_z);

  [[nodiscard]] bool strict_tsdf_corner(int x, int y, int z,
                                        float& distance) const;

  [[nodiscard]] float relaxed_tsdf_corner(int x, int y, int z) const;

  [[nodiscard]] bool sample_normal(const Eigen::Vector3f& point,
                                   Eigen::Vector3f& normal,
                                   bool missing_tsdf_as_zero) const;

  [[nodiscard]] ColorRgba sample_color(const Eigen::Vector3f& point) const;

  [[nodiscard]] static Eigen::Vector3f ray_direction_for_pixel(
      const RaycastOptions& options, const RaycastContext& context,
      ImagePixel pixel);

  void raycast_pixel(const RaycastOptions& options,
                     const RaycastContext& context, ImagePixel pixel,
                     SurfaceMaps& maps) const;

  Eigen::Vector3i resolution_;
  float voxel_size_{};
  Eigen::Vector3f origin_;
  float truncation_distance_{};
  std::vector<Voxel> voxels_;
  std::vector<ColorVoxel> colors_;
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_VOLUME_HPP */
