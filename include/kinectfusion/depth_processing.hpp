#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_DEPTH_PROCESSING_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_DEPTH_PROCESSING_HPP

#include <Eigen/Core>
#include <cstdint>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/rgbd.hpp>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace kinectfusion {

// Depth scale and range defaults are the shared sensor conventions in
// rgbd.hpp (kDefaultTumDepthScale, kDefaultMin/MaxDepthMeters).
inline constexpr unsigned int kDefaultDepthPyramidLevels = 3U;
inline constexpr float kDefaultMaxNormalDepthJumpMeters = 0.1F;
inline constexpr float kDefaultMaxDownsampleDepthJumpMeters = 0.1F;
inline constexpr int kDefaultBilateralRadiusPixels = 2;
inline constexpr float kDefaultBilateralSpatialSigmaPixels = 2.0F;
inline constexpr float kDefaultBilateralDepthSigmaMeters = 0.08F;

struct DepthProcessingOptions {
  unsigned int levels{kDefaultDepthPyramidLevels};
  float depth_scale{kDefaultTumDepthScale};
  float min_depth{kDefaultMinDepthMeters};
  float max_depth{kDefaultMaxDepthMeters};
  float max_normal_depth_jump{kDefaultMaxNormalDepthJumpMeters};
  float max_downsample_depth_jump{kDefaultMaxDownsampleDepthJumpMeters};
  bool bilateral_filter{true};
  int bilateral_radius{kDefaultBilateralRadiusPixels};
  float bilateral_spatial_sigma{kDefaultBilateralSpatialSigmaPixels};
  float bilateral_depth_sigma{kDefaultBilateralDepthSigmaMeters};
};

struct VertexNormalMaps {
  image_proc::Vector3fImage vertices;
  image_proc::Vector3fImage normals;
};

// IsConst toggles the pointee types, so the mutable and read-only views share
// one definition. Views have pointer semantics: constness is shallow, like
// std::span.
template <MemorySpace Space = MemorySpace::kHost, bool IsConst = false>
struct VertexNormalMapsView {
  template <typename T>
  using Pointee = std::conditional_t<IsConst, const T, T>;

  image_proc::ImageView<Pointee<Vec3f>, Space> vertices;
  image_proc::ImageView<Pointee<Vec3f>, Space> normals;

  static constexpr MemorySpace kMemorySpace = Space;
};

using HostVertexNormalMapsView = VertexNormalMapsView<MemorySpace::kHost>;
using DeviceVertexNormalMapsView = VertexNormalMapsView<MemorySpace::kDevice>;
using ConstHostVertexNormalMapsView =
    VertexNormalMapsView<MemorySpace::kHost, true>;
using ConstDeviceVertexNormalMapsView =
    VertexNormalMapsView<MemorySpace::kDevice, true>;

[[nodiscard]] inline HostVertexNormalMapsView view(VertexNormalMaps& maps) {
  return HostVertexNormalMapsView{.vertices = maps.vertices.view(),
                                  .normals = maps.normals.view()};
}

[[nodiscard]] inline ConstHostVertexNormalMapsView view(
    const VertexNormalMaps& maps) {
  return ConstHostVertexNormalMapsView{.vertices = maps.vertices.view(),
                                       .normals = maps.normals.view()};
}

struct DepthProcessingLevel {
  DepthProcessingLevel(image_proc::DepthImage depth,
                       const CameraIntrinsics& camera_intrinsics,
                       image_proc::Vector3fImage vertices,
                       image_proc::Vector3fImage normals)
      : depth_image(std::move(depth)),
        intrinsics(camera_intrinsics),
        maps{.vertices = std::move(vertices), .normals = std::move(normals)} {}

  image_proc::DepthImage depth_image;
  CameraIntrinsics intrinsics;
  VertexNormalMaps maps;
};

class DepthProcessor {
 public:
  // Throws std::invalid_argument
  explicit DepthProcessor(DepthProcessingOptions options = {});

  // Edge-preserving bilateral filter on raw depth. Samples outside the usable
  // depth range (and zeros) are ignored; invalid output pixels stay zero.
  [[nodiscard]] image_proc::DepthImage bilateral_filter(
      const image_proc::DepthImage& depth_image) const;

  // Conservative 2x2 reduction: ignore zero depths, reject neighbourhoods
  // with a large depth jump, and average the remaining valid raw depths.
  [[nodiscard]] image_proc::DepthImage downsample(
      const image_proc::DepthImage& depth_image) const;

  // Back-project one depth image into world space (camera_pose = Identity for
  // camera-space maps). Invalid pixels are set to quiet NaN.
  [[nodiscard]] image_proc::Vector3fImage project_to_vertices(
      const image_proc::DepthImage& depth_image,
      const CameraIntrinsics& intrinsics,
      const Eigen::Matrix4f& camera_pose = Eigen::Matrix4f::Identity()) const;

  // Central-difference surface normals. Border pixels and pixels whose
  // stencil crosses missing vertices or a depth discontinuity stay NaN.
  [[nodiscard]] image_proc::Vector3fImage compute_normals(
      const image_proc::Vector3fImage& vertices) const;

  // Bilateral-filtered depth pyramid with image-aligned vertex/normal maps
  // per level for projective ICP.
  [[nodiscard]] std::vector<DepthProcessingLevel> build_surface_pyramid(
      const image_proc::DepthImage& depth_image,
      const CameraIntrinsics& intrinsics,
      const Eigen::Matrix4f& camera_pose = Eigen::Matrix4f::Identity()) const;

 private:
  // Raw depth usable for filtering/back-projection: non-zero and in range.
  [[nodiscard]] std::optional<float> usable_depth(std::uint16_t raw) const;

  // One output pixel of the bilateral filter: Gaussian-weighted average of
  // the usable raw depths around (col, row), weighted by pixel distance and
  // by metric depth difference to the center sample.
  [[nodiscard]] std::optional<std::uint16_t> bilateral_filtered_pixel(
      const image_proc::DepthImage& depth_image, int col, int row,
      float center_meters) const;

  // Average of the valid raw depths in one kDownsampleFactor^2 input block;
  // nullopt for empty blocks and across depth discontinuities.
  [[nodiscard]] std::optional<std::uint16_t> downsampled_block(
      const image_proc::DepthImage& depth_image, std::size_t col,
      std::size_t row) const;

  // Normal from the cross product of central differences over the 4-neighbour
  // stencil; nullopt at unmeasured vertices and depth discontinuities.
  [[nodiscard]] std::optional<Vec3f> stencil_normal(
      const image_proc::Vector3fImage& vertices, std::size_t col,
      std::size_t row) const;

  DepthProcessingOptions options_;
  // Gaussian exponent scales of the bilateral kernel, precomputed from the
  // sigmas in options_.
  float spatial_scale_{};
  float range_scale_{};
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_DEPTH_PROCESSING_HPP */
