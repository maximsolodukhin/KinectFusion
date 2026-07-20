#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_TSDF_INTEGRATION_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_TSDF_INTEGRATION_HPP

#include <Eigen/Core>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <kinectfusion/cuda_compat.hpp>
#include <kinectfusion/grid.hpp>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/occupancy.hpp>
#include <kinectfusion/rgbd.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <variant>
#include <vector>

namespace kinectfusion {

inline constexpr float kDefaultTsdfMaxWeight = 196.0F;
inline constexpr float kDefaultTruncationDistanceScale = 0.01F;

// kFull sweeps the whole grid and carves free space everywhere. kBand
// integrates only blocks in the truncation band of the depth image
// (Niessner semantics): voxels outside the band keep their value.
// The mode is an ablation axis, not a transparent optimization.
// Is lossy.
enum class IntegrationMode : std::uint8_t { kFull, kBand };

struct TsdfIntegrationOptions {
  float depth_scale{kDefaultTumDepthScale};
  float observation_weight{1.0F};
  float max_weight{kDefaultTsdfMaxWeight};
  float min_depth{kDefaultMinDepthMeters};
  float max_depth{kDefaultMaxDepthMeters};
  bool projective_distance{true};
  bool distance_scaled_truncation{false};
  float truncation_distance_scale{kDefaultTruncationDistanceScale};
  IntegrationMode mode{IntegrationMode::kFull};

  // The truncation band around a surface at `surface_depth`.
  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE float truncation_for(
      float base, float surface_depth) const {
    if (distance_scaled_truncation) {
      return base + (truncation_distance_scale * surface_depth);
    }
    return base;
  }
};

// One depth-camera observation as non-owning views; an empty view means the
// input is absent.
template <MemorySpace Space = MemorySpace::kHost>
struct DepthFrameView {
  image_proc::ImageView<const std::uint16_t, Space> depth{};
  image_proc::ImageView<const std::uint32_t, Space> color{};
  image_proc::ImageView<const Vec3f, Space> normals{};
  CameraIntrinsics intrinsics{};
  RigidTransform world_to_camera{};
};

using HostDepthFrameView = DepthFrameView<MemorySpace::kHost>;
using DeviceDepthFrameView = DepthFrameView<MemorySpace::kDevice>;

// Host-side API: owning host images and the Eigen pose. view() is the
// per-frame Eigen to Device boundary.
struct DepthFrame {
  const image_proc::DepthImage* depth{};
  const image_proc::ColorImage* color{};
  const image_proc::Vector3fImage* normals{};
  CameraIntrinsics intrinsics{};
  Eigen::Matrix4f world_to_camera{Eigen::Matrix4f::Identity()};

  [[nodiscard]] HostDepthFrameView view() const {
    HostDepthFrameView result{.intrinsics = intrinsics,
                              .world_to_camera = from_eigen(world_to_camera)};
    if (depth != nullptr) {
      result.depth = depth->view();
    }
    if (color != nullptr) {
      result.color = color->view();
    }
    if (normals != nullptr) {
      result.normals = normals->view();
    }
    return result;
  }
};

// One frame's device upload, shared across the device pipelines of a set as
// a borrowed pointer. Check CUDA backend.
class DeviceDepthFrame;

struct VoxelObservation {
  Pixel pixel{};
  float surface_depth{};
  // Unit-depth pixel ray in camera space (unnormalized).
  Vec3f ray{};
  // Signed distance clamped to [-1, 1] in truncation units.
  float tsdf{};
  bool integrate_color{};
};

// Per-frame plumbing shared by all TSDF update rules; rules add only their
// weighting policy.
template <MemorySpace Space = MemorySpace::kHost>
class IntegrationContext {
 public:
  IntegrationContext(const DepthFrameView<Space>& frame,
                     const TsdfIntegrationOptions& options)
      : frame_(frame), options_(options) {}

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE const DepthFrameView<Space>&
  frame() const {
    return frame_;
  }
  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE const TsdfIntegrationOptions&
  options() const {
    return options_;
  }

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE Vec3f
  to_camera(const Vec3f& world_point) const {
    return frame_.world_to_camera * world_point;
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE compat::optional<Pixel>
  project_to_pixel(const Vec3f& camera_point) const {
    if (camera_point.z <= 0.0F) {
      return compat::nullopt;
    }

    const Vec2f pixel = frame_.intrinsics.project(camera_point);

    const auto rounded_x = compat::lround(pixel.x);
    const auto rounded_y = compat::lround(pixel.y);

    if (rounded_x < 0 || rounded_y < 0) {
      return compat::nullopt;
    }

    const auto col = static_cast<std::size_t>(rounded_x);
    const auto row = static_cast<std::size_t>(rounded_y);

    if (col >= frame_.depth.width || row >= frame_.depth.height) {
      return compat::nullopt;
    }

    return Pixel{.x = col, .y = row};
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE compat::optional<float> measured_depth(
      const Pixel& pixel) const {
    std::uint16_t depth = frame_.depth.at(pixel.x, pixel.y);
    return depth_in_range(depth, options_.depth_scale, options_.min_depth,
                          options_.max_depth);
  }

  // The truncated-signed-distance observation of voxel (x, y, z); nullopt
  // when the voxel is invisible, unmeasured, or behind the truncation band.
  // Generic over the volume view so every storage representation reuses it.
  template <VoxelGridView VolumeViewT>
  [[nodiscard]] KINECTFUSION_HOST_DEVICE compat::optional<VoxelObservation>
  observe(const VolumeViewT& volume, std::size_t x, std::size_t y,
          std::size_t z) const {
    const Vec3f camera_point = to_camera(volume.cell_center(x, y, z));

    const auto pixel = project_to_pixel(camera_point);
    if (!pixel) {
      return compat::nullopt;
    }

    const auto measured = measured_depth(*pixel);
    if (!measured) {
      return compat::nullopt;
    }
    const float surface_depth = *measured;

    const Vec3f ray = frame_.intrinsics.back_project(pixel->as_vector(), 1.0F);
    const float lambda = norm(ray);
    if (!all_finite(ray) || lambda == 0.0F) {
      return compat::nullopt;
    }

    const float truncation =
        truncation_for(volume.truncation_distance(), surface_depth);
    const float signed_distance =
        projective_sdf(camera_point, lambda, surface_depth);
    if (signed_distance < -truncation) {
      return compat::nullopt;
    }

    const float truncated_sd =
        compat::clamp(signed_distance / truncation, -1.0F, 1.0F);
    const bool integrate_color =
        signed_distance <= truncation * kColorIntegrationTruncationFraction;

    return VoxelObservation{.pixel = *pixel,
                            .surface_depth = surface_depth,
                            .ray = ray,
                            .tsdf = truncated_sd,
                            .integrate_color = integrate_color};
  }

  // Fuses one weighted observation into the voxel (and its colour when the
  // representation stores colour and the observation lies within the colour
  // band). Non-positive weights are dropped.
  template <VoxelGridView VolumeViewT>
  KINECTFUSION_HOST_DEVICE void fuse(const VolumeViewT& volume, std::size_t x,
                                     std::size_t y, std::size_t z,
                                     const VoxelObservation& observation,
                                     float weight) const {
    if (weight <= 0.0F) {
      return;
    }

    auto& voxel = volume.voxel_at(x, y, z);
    voxel = voxel.fused(observation.tsdf, weight, options_.max_weight);

    if constexpr (VolumeViewT::ColorFacet::kEnabled) {
      if (frame_.color.data == nullptr || !observation.integrate_color) {
        return;
      }

      const Vec3f color = color_from_pixel(
          frame_.color.at(observation.pixel.x, observation.pixel.y));

      auto& color_voxel = volume.color_at(x, y, z);
      color_voxel = color_voxel.fused(color, weight, options_.max_weight);
    }
  }

 private:
  static constexpr float kColorIntegrationTruncationFraction = 0.5F;

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE float projective_sdf(
      const Vec3f& camera_point, float lambda, float surface_depth) const {
    if (options_.projective_distance) {
      return surface_depth - (norm(camera_point) / lambda);
    }
    return surface_depth - camera_point.z;
  }

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE float truncation_for(
      float base, float surface_depth) const {
    return options_.truncation_for(base, surface_depth);
  }

  DepthFrameView<Space> frame_;
  TsdfIntegrationOptions options_;
};

using HostIntegrationContext = IntegrationContext<MemorySpace::kHost>;
using DeviceIntegrationContext = IntegrationContext<MemorySpace::kDevice>;

// The contract a TSDF update rule satisfies in every memory space. Each rule
// is one ablation variant and, in the CUDA port, one kernel instantiation.
template <typename R, MemorySpace Space>
concept TsdfUpdateRule =
    requires(const R rule, const VolumeView<Space>& volume,
             const IntegrationContext<Space>& context, std::size_t index) {
      rule.update(volume, context, index, index, index);
    };

// Constant observation weight (3.3: w = 1 suffices)
struct ClassicTsdf {
  template <VoxelGridView VolumeViewT, MemorySpace Space>
  KINECTFUSION_HOST_DEVICE static void update(
      const VolumeViewT& volume, const IntegrationContext<Space>& context,
      std::size_t x, std::size_t y, std::size_t z) {
    const auto observation = context.observe(volume, x, y, z);

    if (!observation) {
      return;
    }

    context.fuse(volume, x, y, z, *observation,
                 context.options().observation_weight);
  }
};

// Weights observations by cos(theta)/depth when live normals are
// available; falls back to the constant weight otherwise.
struct AngleWeightedTsdf {
  template <VoxelGridView VolumeViewT, MemorySpace Space>
  KINECTFUSION_HOST_DEVICE static void update(
      const VolumeViewT& volume, const IntegrationContext<Space>& context,
      std::size_t x, std::size_t y, std::size_t z) {
    const auto observation = context.observe(volume, x, y, z);

    if (!observation) {
      return;
    }

    context.fuse(volume, x, y, z, *observation,
                 observation_weight(context, *observation));
  }

 private:
  template <MemorySpace Space>
  [[nodiscard]] KINECTFUSION_HOST_DEVICE static float observation_weight(
      const IntegrationContext<Space>& context,
      const VoxelObservation& observation) {
    float weight = context.options().observation_weight;
    const auto& normals = context.frame().normals;

    if (normals.data != nullptr) {
      const Vec3f& normal_sample =
          normals.at(observation.pixel.x, observation.pixel.y);
      if (all_finite(normal_sample)) {
        // View direction is the (quantised) pixel ray, matching the SDF.
        const Vec3f view = -normalized(observation.ray);
        const float cos_theta =
            compat::max(0.0F, dot(normalized(normal_sample), view));

        weight *= cos_theta / observation.surface_depth;
      }
    }
    return weight;
  }
};

static_assert(TsdfUpdateRule<ClassicTsdf, MemorySpace::kHost>);
static_assert(TsdfUpdateRule<ClassicTsdf, MemorySpace::kDevice>);
static_assert(TsdfUpdateRule<AngleWeightedTsdf, MemorySpace::kHost>);
static_assert(TsdfUpdateRule<AngleWeightedTsdf, MemorySpace::kDevice>);

using TsdfRuleVariant = std::variant<ClassicTsdf, AngleWeightedTsdf>;

// Runs the rule sweep of one frame over a volume view in its memory space.
// The CUDA backend defines the device specialization.
template <MemorySpace Space>
struct IntegrationSweep;

template <>
struct IntegrationSweep<MemorySpace::kHost> {
  template <VoxelGridView VolumeViewT>
  static void run(const VolumeViewT& volume,
                  const IntegrationContext<MemorySpace::kHost>& context,
                  const TsdfRuleVariant& rule) {
    std::visit(
        [&](const auto& chosen) {
          for (const auto [x, y, z] : GridIndices{volume.resolution()}) {
            chosen.update(volume, context, x, y, z);
          }
        },
        rule);
  }
};

using HostIntegrationSweep = IntegrationSweep<MemorySpace::kHost>;

// Walks the truncation band of each depth pixel in `step` increments. It
// reports the flat x-major index of each block a sample lands in, and the
// visitor decides what a hit means. The host loop and the CUDA mark kernels
// share `visit_pixel`.
struct TruncationBandWalk {
  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE static float half_block_step(
      const VolumeGeometry& geometry) {
    return 0.5F * static_cast<float>(kVoxelBlockEdge) * geometry.voxel_size;
  }

  template <MemorySpace Space, typename VisitBlock>
  KINECTFUSION_HOST_DEVICE static void visit_pixel(
      const DepthFrameView<Space>& frame, const TsdfIntegrationOptions& options,
      const VolumeGeometry& geometry, const BlockGrid& blocks, float step,
      const RigidTransform& camera_to_world, std::size_t col, std::size_t row,
      const VisitBlock& visit_block) {
    const auto measured =
        depth_in_range(frame.depth.at(col, row), options.depth_scale,
                       options.min_depth, options.max_depth);
    if (!measured) {
      return;
    }
    const float band =
        options.truncation_for(geometry.truncation_distance, *measured);

    const Vec2f pixel{.x = static_cast<float>(col),
                      .y = static_cast<float>(row)};

    const float inv_voxel = 1.0F / geometry.voxel_size;
    const int samples = static_cast<int>((2.0F * band) / step) + 1;
    std::size_t previous_index = ~std::size_t{0};

    for (int sample = 0; sample <= samples; ++sample) {
      const float depth =
          (*measured - band) + (static_cast<float>(sample) * step);
      const Vec3f world =
          camera_to_world * frame.intrinsics.back_project(pixel, depth);
      const Vec3f grid = (world - geometry.origin) * inv_voxel;

      if (!geometry.resolution.contains(grid)) {
        continue;
      }

      const std::size_t index =
          blocks.block_of_voxel(static_cast<std::size_t>(grid.x),   //
                                static_cast<std::size_t>(grid.y),   //
                                static_cast<std::size_t>(grid.z));  //

      if (index == previous_index) {
        continue;
      }
      previous_index = index;
      visit_block(index);
    }
  }

  template <std::invocable<std::size_t> VisitBlock>
  static void visit(const DepthFrameView<MemorySpace::kHost>& frame,
                    const TsdfIntegrationOptions& options,
                    const VolumeGeometry& geometry, const BlockGrid& blocks,
                    float step, const VisitBlock& visit_block) {
    const RigidTransform camera_to_world = inverse(frame.world_to_camera);
    for (const auto [col, row] :
         PixelIndices{frame.depth.width, frame.depth.height}) {
      visit_pixel(frame, options, geometry, blocks, step, camera_to_world, col,
                  row, visit_block);
    }
  }
};

// Band integration: mark the blocks along each band, compact the set bits
// into a work list, then integrate only those blocks. One instance per
// pipeline owns the scratch.
template <MemorySpace Space>
class BandIntegrationSweep;

template <>
class BandIntegrationSweep<MemorySpace::kHost> {
 public:
  template <VoxelGridView VolumeViewT>
  void run(const VolumeViewT& volume,
           const IntegrationContext<MemorySpace::kHost>& context,
           const TsdfRuleVariant& rule) {
    const BlockGrid blocks = BlockGrid::for_resolution(volume.resolution());
    const std::size_t word_count = BlockBitmapOps::word_count(blocks.count());
    bitmap_.assign(word_count, 0U);
    list_.clear();

    TruncationBandWalk::visit(context.frame(), context.options(),
                              volume.geometry, blocks, volume.voxel_size(),
                              [this](std::size_t block) {
                                BlockBitmapOps::set(bitmap_.data(), block);
                              });
    for (std::size_t word = 0; word < word_count; ++word) {
      BlockBitmapOps::for_each_set_bit(
          bitmap_[word], word * kBitmapWordBits, [this](std::size_t block) {
            list_.push_back(static_cast<std::uint32_t>(block));
          });
    }

    std::visit(
        [&](const auto& chosen) {
          for (const std::uint32_t block : list_) {
            integrate_block(volume, context, chosen, blocks, block);
          }
        },
        rule);
  }

 private:
  template <VoxelGridView VolumeViewT, TsdfUpdateRule<MemorySpace::kHost> Rule>
  static void integrate_block(
      const VolumeViewT& volume,
      const IntegrationContext<MemorySpace::kHost>& context, const Rule& rule,
      const BlockGrid& blocks, std::uint32_t block) {
    for (const auto [x, y, z] :
         BlockVoxels{block, blocks, volume.resolution()}) {
      rule.update(volume, context, x, y, z);
    }
  }

  std::vector<std::uint32_t> bitmap_;
  std::vector<std::uint32_t> list_;
};

using HostBandIntegrationSweep = BandIntegrationSweep<MemorySpace::kHost>;

class TsdfIntegrator {
 public:
  // Throws std::invalid_argument
  explicit TsdfIntegrator(TsdfRuleVariant rule = AngleWeightedTsdf{},
                          TsdfIntegrationOptions options = {});

  void integrate(const HostVolumeView& volume, const DepthFrame& frame) const;

  // The validated configuration; space-specific drivers launch from these.
  [[nodiscard]] const TsdfIntegrationOptions& options() const {
    return options_;
  }
  [[nodiscard]] const TsdfRuleVariant& rule() const { return rule_; }

  // Throws std::invalid_argument
  static void validate_frame(const DepthFrame& frame);

 private:
  [[nodiscard]] static TsdfIntegrationOptions validated(
      TsdfIntegrationOptions options);

  TsdfIntegrationOptions options_;
  TsdfRuleVariant rule_;
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_TSDF_INTEGRATION_HPP */
