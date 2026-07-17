#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_DEPTH_PROCESSING_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_DEPTH_PROCESSING_HPP

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <kinectfusion/cuda_compat.hpp>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/rgbd.hpp>
#include <kinectfusion/tsdf_integration.hpp>
#include <kinectfusion/vector.hpp>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
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

  // Throws std::invalid_argument; shared by the host and device processors.
  [[nodiscard]] static DepthProcessingOptions validated(
      DepthProcessingOptions options);
};

// Linear resolution reduction per pyramid level: each level halves width and
// height, downsampling over a kDownsampleFactor x kDownsampleFactor block.
inline constexpr std::size_t kDownsampleFactor = 2U;

// Edge-preserving bilateral filter on raw depth. Samples outside the usable
// depth range (and zeros) are ignored; invalid output pixels stay zero.
template <MemorySpace Space = MemorySpace::kHost>
class BilateralFilter {
 public:
  BilateralFilter(
      const image_proc::ImageView<const std::uint16_t, Space>& depth,
      const DepthProcessingOptions& options)
      : depth_(depth),
        radius_(options.bilateral_radius),
        depth_scale_(options.depth_scale),
        min_depth_(options.min_depth),
        max_depth_(options.max_depth),
        spatial_scale_(
            gaussian_exponent_scale(options.bilateral_spatial_sigma)),
        range_scale_(gaussian_exponent_scale(options.bilateral_depth_sigma)) {}

  // nullopt when the center sample is unusable or no neighbour contributes.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE compat::optional<std::uint16_t>
  operator()(std::size_t col, std::size_t row) const {
    const auto center_meters = usable_depth(depth_.at(col, row));
    if (!center_meters) {
      return compat::nullopt;
    }

    const int center_x = static_cast<int>(col);
    const int center_y = static_cast<int>(row);
    const int width = static_cast<int>(depth_.width);
    const int height = static_cast<int>(depth_.height);

    const int x_begin = compat::max(center_x - radius_, 0);
    const int y_begin = compat::max(center_y - radius_, 0);
    const int x_end = compat::min(center_x + radius_, width - 1);
    const int y_end = compat::min(center_y + radius_, height - 1);

    float weighted_sum = 0.0F;
    float weight_sum = 0.0F;

    for (int y = y_begin; y <= y_end; ++y) {
      for (int x = x_begin; x <= x_end; ++x) {
        const auto raw =
            depth_.at(static_cast<std::size_t>(x), static_cast<std::size_t>(y));
        const auto sample_meters = usable_depth(raw);
        if (!sample_meters) {
          continue;
        }
        const auto pixel_distance2 =
            static_cast<float>(((x - center_x) * (x - center_x)) +
                               ((y - center_y) * (y - center_y)));
        const float depth_difference = *sample_meters - *center_meters;
        const float weight =
            std::exp((pixel_distance2 * spatial_scale_) +
                     (depth_difference * depth_difference * range_scale_));
        weighted_sum += weight * static_cast<float>(raw);
        weight_sum += weight;
      }
    }
    if (weight_sum <= 0.0F) {
      return compat::nullopt;
    }
    return static_cast<std::uint16_t>(
        compat::lround(weighted_sum / weight_sum));
  }

 private:
  // Exponent scale of a Gaussian falloff: exp(-d^2 / (2 sigma^2)).
  [[nodiscard]] static constexpr float gaussian_exponent_scale(float sigma) {
    return -0.5F / (sigma * sigma);
  }

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE compat::optional<float>
  usable_depth(std::uint16_t raw) const {
    return depth_in_range(raw, depth_scale_, min_depth_, max_depth_);
  }

  image_proc::ImageView<const std::uint16_t, Space> depth_;
  int radius_;
  float depth_scale_;
  float min_depth_;
  float max_depth_;
  float spatial_scale_;
  float range_scale_;
};

using HostBilateralFilter = BilateralFilter<MemorySpace::kHost>;
using DeviceBilateralFilter = BilateralFilter<MemorySpace::kDevice>;

// Conservative 2x2 reduction: ignore zero depths, reject neighbourhoods with
// a large depth jump, and average the remaining valid raw depths.
template <MemorySpace Space = MemorySpace::kHost>
class BlockDownsample {
 public:
  BlockDownsample(
      const image_proc::ImageView<const std::uint16_t, Space>& depth,
      const DepthProcessingOptions& options)
      : depth_(depth),
        depth_scale_(options.depth_scale),
        max_depth_jump_(options.max_downsample_depth_jump) {}

  // nullopt for empty blocks and across depth discontinuities.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE compat::optional<std::uint16_t>
  operator()(std::size_t col, std::size_t row) const {
    std::uint32_t sum = 0;
    std::uint32_t count = 0;
    std::uint16_t minimum_observed_depth =
        std::numeric_limits<std::uint16_t>::max();
    std::uint16_t maximum_observed_depth = 0;

    for (std::size_t dy = 0; dy < kDownsampleFactor; ++dy) {
      for (std::size_t dx = 0; dx < kDownsampleFactor; ++dx) {
        const std::uint16_t sample = depth_.at((col * kDownsampleFactor) + dx,
                                               (row * kDownsampleFactor) + dy);
        if (sample == 0) {
          continue;
        }
        sum += sample;
        ++count;
        minimum_observed_depth = compat::min(minimum_observed_depth, sample);
        maximum_observed_depth = compat::max(maximum_observed_depth, sample);
      }
    }

    if (count == 0) {
      return compat::nullopt;
    }

    const float max_depth =
        depth_to_meters(maximum_observed_depth, depth_scale_);
    const float min_depth =
        depth_to_meters(minimum_observed_depth, depth_scale_);
    if (max_depth - min_depth > max_depth_jump_) {
      return compat::nullopt;
    }

    return static_cast<std::uint16_t>((sum + (count / 2U)) / count);
  }

 private:
  image_proc::ImageView<const std::uint16_t, Space> depth_;
  float depth_scale_;
  float max_depth_jump_;
};

using HostBlockDownsample = BlockDownsample<MemorySpace::kHost>;
using DeviceBlockDownsample = BlockDownsample<MemorySpace::kDevice>;

// Back-projects one depth pixel into world space; invalid samples yield an
// invalid vertex.
template <MemorySpace Space = MemorySpace::kHost>
class VertexProjection {
 public:
  VertexProjection(
      const image_proc::ImageView<const std::uint16_t, Space>& depth,
      const CameraIntrinsics& intrinsics, const RigidTransform& camera_pose,
      const DepthProcessingOptions& options)
      : depth_(depth),
        intrinsics_(intrinsics),
        camera_pose_(camera_pose),
        depth_scale_(options.depth_scale),
        min_depth_(options.min_depth),
        max_depth_(options.max_depth) {}

  // nullopt at unmeasured or out-of-range samples.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE compat::optional<Vec3f> operator()(
      std::size_t col, std::size_t row) const {
    const auto depth = depth_in_range(depth_.at(col, row), depth_scale_,
                                      min_depth_, max_depth_);
    if (!depth) {
      return compat::nullopt;
    }
    const Vec3f camera_point = intrinsics_.back_project(
        Vec2f{.x = static_cast<float>(col), .y = static_cast<float>(row)},
        *depth);
    return camera_pose_ * camera_point;
  }

 private:
  image_proc::ImageView<const std::uint16_t, Space> depth_;
  CameraIntrinsics intrinsics_;
  RigidTransform camera_pose_;
  float depth_scale_;
  float min_depth_;
  float max_depth_;
};

using HostVertexProjection = VertexProjection<MemorySpace::kHost>;
using DeviceVertexProjection = VertexProjection<MemorySpace::kDevice>;

// Central-difference surface normal over the 4-neighbour stencil; invalid on
// the border, at unmeasured vertices, and across depth discontinuities.
template <MemorySpace Space = MemorySpace::kHost>
class NormalEstimation {
 public:
  NormalEstimation(const image_proc::ImageView<const Vec3f, Space>& vertices,
                   const DepthProcessingOptions& options)
      : vertices_(vertices), max_depth_jump_(options.max_normal_depth_jump) {}

  // nullopt on the border (the stencil needs all four neighbours), at
  // unmeasured vertices, and across depth discontinuities.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE compat::optional<Vec3f> operator()(
      std::size_t col, std::size_t row) const {
    if (col == 0 || row == 0 || col + 1 >= vertices_.width ||
        row + 1 >= vertices_.height) {
      return compat::nullopt;
    }

    const Vec3f& left = vertices_.at(col - 1, row);
    const Vec3f& right = vertices_.at(col + 1, row);

    const Vec3f& top = vertices_.at(col, row - 1);
    const Vec3f& bottom = vertices_.at(col, row + 1);

    // NaN propagates through the sum
    if (!all_finite(vertices_.at(col, row) + left + right + top + bottom)) {
      return compat::nullopt;
    }
    if (std::abs(right.z - left.z) > max_depth_jump_ ||
        std::abs(bottom.z - top.z) > max_depth_jump_) {
      return compat::nullopt;
    }
    const Vec3f normal = cross(bottom - top, right - left);
    const float length = norm(normal);

    if (length <= 0.0F) {
      return compat::nullopt;
    }
    return normal / length;
  }

 private:
  image_proc::ImageView<const Vec3f, Space> vertices_;
  float max_depth_jump_;
};

using HostNormalEstimation = NormalEstimation<MemorySpace::kHost>;
using DeviceNormalEstimation = NormalEstimation<MemorySpace::kDevice>;

template <MemorySpace Space>
struct SurfaceFor {
  image_proc::Vector3fImageFor<Space> vertices;
  image_proc::Vector3fImageFor<Space> normals;
};

using Surface = SurfaceFor<MemorySpace::kHost>;
using DeviceSurface = SurfaceFor<MemorySpace::kDevice>;

// IsConst toggles the pointee types, so the mutable and read-only views share
// one definition. Views have pointer semantics: constness is shallow, like
// std::span.
template <MemorySpace Space = MemorySpace::kHost, bool IsConst = false>
struct SurfaceView {
  template <typename T>
  using Pointee = std::conditional_t<IsConst, const T, T>;

  image_proc::ImageView<Pointee<Vec3f>, Space> vertices;
  image_proc::ImageView<Pointee<Vec3f>, Space> normals;

  static constexpr MemorySpace kMemorySpace = Space;

  // Mutable views convert to read-only views implicitly, like std::span.
  template <bool TargetConst = true>
    requires(TargetConst && !IsConst)
  [[nodiscard]] KINECTFUSION_HOST_DEVICE
  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  operator SurfaceView<Space, TargetConst>() const {
    return {.vertices = vertices, .normals = normals};
  }
};

template <MemorySpace Space>
using ConstSurfaceView = SurfaceView<Space, true>;

using HostSurfaceView = SurfaceView<MemorySpace::kHost>;
using DeviceSurfaceView = SurfaceView<MemorySpace::kDevice>;

using ConstHostSurfaceView = SurfaceView<MemorySpace::kHost, true>;
using ConstDeviceSurfaceView = SurfaceView<MemorySpace::kDevice, true>;

template <MemorySpace Space>
[[nodiscard]] SurfaceView<Space> view(SurfaceFor<Space>& surface) {
  return {.vertices = surface.vertices.view(),
          .normals = surface.normals.view()};
}

template <MemorySpace Space>
[[nodiscard]] ConstSurfaceView<Space> view(const SurfaceFor<Space>& surface) {
  return {.vertices = surface.vertices.view(),
          .normals = surface.normals.view()};
}

template <MemorySpace Space = MemorySpace::kHost>
struct DepthProcessingLevel {
  using depth_image_type = image_proc::DepthImageFor<Space>;

  DepthProcessingLevel(image_proc::DepthImageFor<Space> depth,
                       const CameraIntrinsics& camera_intrinsics,
                       image_proc::Vector3fImageFor<Space> vertices,
                       image_proc::Vector3fImageFor<Space> normals)
      : depth_image(std::move(depth)),
        intrinsics(camera_intrinsics),
        surface{.vertices = std::move(vertices),
                .normals = std::move(normals)} {}

  image_proc::DepthImageFor<Space> depth_image;
  CameraIntrinsics intrinsics;
  SurfaceFor<Space> surface;
};

template <MemorySpace Space>
using SurfacePyramidFor = std::vector<DepthProcessingLevel<Space>>;

using SurfacePyramid = SurfacePyramidFor<MemorySpace::kHost>;

template <MemorySpace Space = MemorySpace::kHost>
class DepthProcessor;

template <>
class DepthProcessor<MemorySpace::kHost> {
  using DepthImg = image_proc::DepthImage;
  using Vec3fImg = image_proc::Vector3fImage;

 public:
  // Throws std::invalid_argument
  explicit DepthProcessor(DepthProcessingOptions options = {});

  // Edge-preserving bilateral filter on raw depth. Samples outside the usable
  // depth range (and zeros) are ignored; invalid output pixels stay zero.
  [[nodiscard]] image_proc::DepthImage bilateral_filter(
      const DepthImg& depth_image) const;

  // Conservative 2x2 reduction: ignore zero depths, reject neighbourhoods
  // with a large depth jump, and average the remaining valid raw depths.
  [[nodiscard]] DepthImg downsample(const DepthImg& depth_image) const;

  // Back-project one depth image into world space (camera_pose = Identity for
  // camera-space maps). Invalid pixels are set to quiet NaN.
  [[nodiscard]] image_proc::Vector3fImage project_to_vertices(
      const DepthImg& depth_image, const CameraIntrinsics& intrinsics,
      const Eigen::Matrix4f& camera_pose = Eigen::Matrix4f::Identity()) const;

  // Central-difference surface normals. Border pixels and pixels whose
  // stencil crosses missing vertices or a depth discontinuity stay NaN.
  [[nodiscard]] Vec3fImg compute_normals(const Vec3fImg& vertices) const;

  // Bilateral-filtered depth pyramid with image-aligned vertex/normal maps
  // per level for projective ICP.
  [[nodiscard]] SurfacePyramid build_surface_pyramid(
      const DepthImg& depth_image, const CameraIntrinsics& intrinsics,
      const Eigen::Matrix4f& camera_pose = Eigen::Matrix4f::Identity()) const;

 private:
  DepthProcessingOptions options_;
};

DepthProcessor(DepthProcessingOptions = {})
    -> DepthProcessor<MemorySpace::kHost>;

using HostDepthProcessor = DepthProcessor<MemorySpace::kHost>;
using DeviceDepthProcessor = DepthProcessor<MemorySpace::kDevice>;

// The live maps of one pyramid level, in the memory space they were built in.
using LiveViewsVariant =
    std::variant<ConstHostSurfaceView, ConstDeviceSurfaceView>;

// One pyramid level as ICP consumes it: the live maps and the camera they
// were projected with. Kept together so the two cannot name different levels.
struct PyramidLevel {
  LiveViewsVariant surface;
  CameraIntrinsics intrinsics;
};

// One pyramid builder per reconstruction, type-erased over the memory space
// the pyramid lives in
class PyramidSource {
 public:
  PyramidSource(const PyramidSource&) = delete;
  PyramidSource& operator=(const PyramidSource&) = delete;
  PyramidSource(PyramidSource&&) = delete;
  PyramidSource& operator=(PyramidSource&&) = delete;
  virtual ~PyramidSource() = default;

  // Rebuilds the pyramid for one raw frame and returns the level count;
  // previously returned views invalidate
  virtual std::size_t build(const image_proc::DepthImage& raw_depth,
                            const CameraIntrinsics& intrinsics) = 0;

  [[nodiscard]] virtual PyramidLevel level(std::size_t index) const = 0;

  [[nodiscard]] virtual const image_proc::Vector3fImage* host_normals()
      const = 0;

  // Integration upload reusing device raw depth and pyramid normals.
  // nullptr on host sources. Valid until the next build call.
  [[nodiscard]] virtual const DeviceDepthFrame* device_frame(
      const DepthFrame& frame) = 0;

  struct Creation {
    std::unique_ptr<PyramidSource> source;
    // Non-empty when the requested space was unavailable and the pyramid
    // fell back to host processing.
    std::string fallback_reason;
  };

  // Warn-and-fallback factory; misconfiguration throws std::invalid_argument.
  [[nodiscard]] static Creation create(MemorySpace space,
                                       const DepthProcessingOptions& options);

 protected:
  PyramidSource() = default;

 private:
  // Defined in the CUDA backend, should have a concept?
  [[nodiscard]] static std::unique_ptr<PyramidSource> create_device(
      const DepthProcessingOptions& options);
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_DEPTH_PROCESSING_HPP */
