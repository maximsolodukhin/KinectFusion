#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_ICP_CORRESPONDENCE_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_ICP_CORRESPONDENCE_HPP

#include <Eigen/Core>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <kinectfusion/cuda_compat.hpp>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/rgbd.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <memory>
#include <type_traits>

namespace kinectfusion {

// Degrees of freedom of the linearised point-plane ICP system (three rotation
// + three translation parameters).
inline constexpr std::size_t kIcpDof = 6;
inline constexpr std::size_t kIcpUpperTriangleSize =
    kIcpDof * (kIcpDof + 1) / 2;

inline constexpr float kDefaultMaxIcpPointDistanceMeters = 0.05F;
// cos(15 degrees); reject correspondences whose normals disagree by more.
inline constexpr float kDefaultMinIcpNormalDot = 0.9659258F;

inline constexpr float kDefaultIcpDampingLambda = 1.0e-3F;
inline constexpr float kDefaultIcpLambdaIncrease = 10.0F;
inline constexpr float kDefaultIcpLambdaDecrease = 0.1F;
inline constexpr float kMinIcpLambda = 1.0e-9F;
inline constexpr float kMaxIcpLambda = 1.0e6F;
inline constexpr unsigned int kMaxIcpDampingEscalations = 8;

// One accepted live-to-model correspondence, linearised for the point-plane
// system: `jacobian` is its 6-DOF row, `residual` the point-plane error.
struct IcpCorrespondence {
  std::array<float, kIcpDof> jacobian{};
  float residual{};
  float distance{};

  // Rotational gradient rows + model normal as the translational rows.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE static IcpCorrespondence from_match(
      const Vec3f& source, const Vec3f& model_vertex, const Vec3f& model_normal,
      float distance) {
    const Vec3f rotational = cross(source, model_normal);
    return {.jacobian = {rotational.x, rotational.y, rotational.z,
                         model_normal.x, model_normal.y, model_normal.z},
            .residual = dot(model_normal, model_vertex - source),
            .distance = distance};
  }
};

static_assert(std::is_trivially_copyable_v<IcpCorrespondence>);

// kIdentity adds lambda * I. kDiagonal adds lambda * diag(J^T J).
enum class IcpDampingMode : std::uint8_t { kNone, kIdentity, kDiagonal };

struct IcpDamping {
  IcpDampingMode mode{IcpDampingMode::kNone};
  float lambda{kDefaultIcpDampingLambda};

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE bool active() const {
    return mode != IcpDampingMode::kNone;
  }

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE float diagonal_offset(
      float jtj_diagonal, float scale) const {
    switch (mode) {
      case IcpDampingMode::kIdentity:
        return scale;
      case IcpDampingMode::kDiagonal:
        return scale * jtj_diagonal;
      case IcpDampingMode::kNone:
        break;
    }
    return 0.0F;
  }
};

static_assert(std::is_trivially_copyable_v<IcpDamping>);

struct IcpDampingSchedule {
  float increase{kDefaultIcpLambdaIncrease};
  float decrease{kDefaultIcpLambdaDecrease};
  float min_lambda{kMinIcpLambda};
  float max_lambda{kMaxIcpLambda};

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE float grow(float lambda) const {
    return compat::min(lambda * increase, max_lambda);
  }

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE float shrink(
      float lambda) const {
    return compat::max(lambda * decrease, min_lambda);
  }
};

static_assert(std::is_trivially_copyable_v<IcpDampingSchedule>);

// Accumulated point-plane normal equations (row-major upper triangle of
// J^T J plus J^T r) and the correspondence statistics the diagnostics report.
struct IcpNormalEquations {
  std::size_t count{};
  float distance_sum{};
  float residual_sum_squares{};
  std::array<float, kIcpUpperTriangleSize> jtj{};
  std::array<float, kIcpDof> jtr{};

  KINECTFUSION_HOST_DEVICE void accumulate(
      const IcpCorrespondence& correspondence) {
    std::size_t entry = 0;
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
    for (std::size_t row = 0; row < kIcpDof; ++row) {
      for (std::size_t col = row; col < kIcpDof; ++col) {
        jtj[entry] +=
            correspondence.jacobian[row] * correspondence.jacobian[col];
        ++entry;
      }
    }
    for (std::size_t row = 0; row < kIcpDof; ++row) {
      jtr[row] += correspondence.jacobian[row] * correspondence.residual;
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
    distance_sum += correspondence.distance;
    residual_sum_squares += correspondence.residual * correspondence.residual;
    ++count;
  }

  // Projective association re-gates the correspondence set each trial, so the
  // cost must be per-correspondence to stay comparable across trials.
  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE float mean_squared_residual()
      const {
    if (count == 0) {
      return 0.0F;
    }
    return residual_sum_squares / static_cast<float>(count);
  }

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE static std::size_t
  triangle_entry(std::size_t row, std::size_t col) {
    return (row * kIcpDof) - (row * (row + 1) / 2) + col;
  }

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE float diagonal(
      std::size_t row) const {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    return jtj[triangle_entry(row, row)];
  }

  [[nodiscard]] Eigen::Matrix<float, kIcpDof, kIcpDof> matrix() const {
    Eigen::Matrix<float, kIcpDof, kIcpDof> dense;
    std::size_t entry = 0;
    for (std::size_t row = 0; row < kIcpDof; ++row) {
      for (std::size_t col = row; col < kIcpDof; ++col) {
        const auto first = static_cast<Eigen::Index>(row);
        const auto second = static_cast<Eigen::Index>(col);

        dense(first, second) = jtj.at(entry);
        dense(second, first) = jtj.at(entry);

        ++entry;
      }
    }
    return dense;
  }

  [[nodiscard]] Eigen::Matrix<float, kIcpDof, 1> rhs() const {
    Eigen::Matrix<float, kIcpDof, 1> dense;
    for (std::size_t row = 0; row < kIcpDof; ++row) {
      dense(static_cast<Eigen::Index>(row)) = jtr.at(row);
    }
    return dense;
  }
};

static_assert(std::is_trivially_copyable_v<IcpNormalEquations>);

struct IcpIterationTransforms {
  using TranslateMat = Eigen::Matrix4f;
  RigidTransform camera{};  // live camera_to_world
  RigidTransform model{};   // model world_to_camera

  [[nodiscard]] static IcpIterationTransforms from_poses(
      const TranslateMat& camera_to_world,
      const TranslateMat& model_world_to_camera) {
    return {.camera = from_eigen(camera_to_world),
            .model = from_eigen(model_world_to_camera)};
  }
};

static_assert(std::is_trivially_copyable_v<IcpIterationTransforms>);

struct CorrespondenceGates {
  float max_point_distance{kDefaultMaxIcpPointDistanceMeters};
  float min_normal_dot{kDefaultMinIcpNormalDot};
};

template <MemorySpace Space>
struct TrackingSurfaces {
  ConstSurfaceView<Space> live{};
  ConstSurfaceView<Space> model{};

  // Pairs the live views with the vertex/normal planes of a rendered model.
  [[nodiscard]] static TrackingSurfaces from_render(
      const ConstSurfaceView<Space>& live_views,
      const ConstSurfaceMapsView<Space>& model_render) {
    return {.live = live_views,
            .model = {.vertices = model_render.points,
                      .normals = model_render.normals}};
  }
};

using HostTrackingSurfaces = TrackingSurfaces<MemorySpace::kHost>;
using DeviceTrackingSurfaces = TrackingSurfaces<MemorySpace::kDevice>;

// Projects one live vertex into the model frame and gates the surface pair it
// lands on; nullopt when the pixel is unmeasured, the projection misses the
// model image, or the distance/normal checks reject the pair.
template <MemorySpace Space = MemorySpace::kHost>
class CorrespondenceSearch {
 public:
  CorrespondenceSearch(const TrackingSurfaces<Space>& surfaces,
                       const CameraIntrinsics& model_intrinsics,
                       const IcpIterationTransforms& transforms,
                       const CorrespondenceGates& gates)
      : surfaces_(surfaces),
        intrinsics_(model_intrinsics),
        transforms_(transforms),
        gates_(gates) {}

  [[nodiscard]] KINECTFUSION_FORCEINLINE_DEVICE const ConstSurfaceView<Space>&
  live() const {
    return surfaces_.live;
  }

  // The device GN loop composes each increment onto the pose in place.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE IcpIterationTransforms&
  transforms_ref() {
    return transforms_;
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE compat::optional<IcpCorrespondence>
  match(std::size_t x, std::size_t y) const {
    const auto& [live, model] = surfaces_;
    const Vec3f& live_vertex = live.vertices.at(x, y);
    const Vec3f& live_normal = live.normals.at(x, y);
    if (!all_finite(live_vertex) || !all_finite(live_normal)) {
      return compat::nullopt;
    }

    const Vec3f source = transforms_.camera * live_vertex;
    const Vec3f source_in_model = transforms_.model * source;
    if (!all_finite(source_in_model) || source_in_model.z <= 0.0F) {
      return compat::nullopt;
    }

    const Vec2f pixel = intrinsics_.project(source_in_model);

    const auto model_x = compat::lround(pixel.x);
    const auto model_y = compat::lround(pixel.y);

    if (model_x < 0 || model_y < 0) {
      return compat::nullopt;
    }

    const auto col = static_cast<std::size_t>(model_x);
    const auto row = static_cast<std::size_t>(model_y);
    if (col >= model.vertices.width || row >= model.vertices.height) {
      return compat::nullopt;
    }

    const Vec3f& model_vertex = model.vertices.at(col, row);
    const Vec3f& model_normal = model.normals.at(col, row);
    if (!all_finite(model_vertex) || !all_finite(model_normal)) {
      return compat::nullopt;
    }

    const Vec3f source_normal =
        normalized(transforms_.camera.rotation * live_normal);
    const float distance = norm(source - model_vertex);

    if (distance > gates_.max_point_distance) {
      return compat::nullopt;
    }

    if (dot(source_normal, model_normal) < gates_.min_normal_dot) {
      return compat::nullopt;
    }

    return IcpCorrespondence::from_match(source, model_vertex, model_normal,
                                         distance);
  }

 private:
  TrackingSurfaces<Space> surfaces_;
  CameraIntrinsics intrinsics_;
  IcpIterationTransforms transforms_;
  CorrespondenceGates gates_;
};

using HostCorrespondenceSearch = CorrespondenceSearch<MemorySpace::kHost>;
using DeviceCorrespondenceSearch = CorrespondenceSearch<MemorySpace::kDevice>;

// Selects how one iteration's reduction graph is constructed. Both tags record
// the same nodes and produce the same result; they exist to ablate the
// graph-construction cost against each other.
struct ExplicitGraphBuild {};  // assembled node by node through the graph API
struct CapturedGraphBuild {};  // recorded from stream capture

// The closed strategy set: the sweep is instantiated for each tag.
template <typename B>
concept GraphBuildStrategy =
    std::same_as<B, ExplicitGraphBuild> || std::same_as<B, CapturedGraphBuild>;

// Parameters and result of the whole-level device GN loop. Status values
// mirror IcpFailure.
struct DeviceIcpLoopParams {
  std::size_t min_correspondences{};
  float min_update_translation{};
  float min_update_rotation{};
  float max_update_translation{};
  float max_update_rotation{};
  IcpDamping damping{};
  IcpDampingSchedule schedule{};
  bool adaptive_damping{};
};

struct DeviceIcpLoopResult {
  IcpIterationTransforms transforms{};
  IcpNormalEquations equations{};
  // A rejected trial rolls the pose back to here.
  RigidTransform best_camera{};
  float best_cost{};
  float lambda{};
  std::int32_t trials{};
  std::int32_t rejected{};
  float update_translation{};
  float update_rotation{};
  std::int32_t status{};  // 0 max-iters, 1 converged, 2 too few, 3 solve, 4 big
};

static_assert(std::is_trivially_copyable_v<DeviceIcpLoopResult>);

// Owns the per-iteration reduction: the device buffers, the private stream, and
// the grid-indexed executable-graph cache. `Build` picks the graph-construction
// strategy without touching the shared reduction kernel.
template <GraphBuildStrategy Build>
class BasicDeviceCorrespondenceSweep {
 public:
  BasicDeviceCorrespondenceSweep();
  ~BasicDeviceCorrespondenceSweep();

  BasicDeviceCorrespondenceSweep(const BasicDeviceCorrespondenceSweep&) =
      delete;
  BasicDeviceCorrespondenceSweep& operator=(
      const BasicDeviceCorrespondenceSweep&) = delete;
  BasicDeviceCorrespondenceSweep(BasicDeviceCorrespondenceSweep&&) noexcept;
  BasicDeviceCorrespondenceSweep& operator=(
      BasicDeviceCorrespondenceSweep&&) noexcept;

  [[nodiscard]] IcpNormalEquations run(
      const DeviceCorrespondenceSearch& search);

  // Runs `iterations` GN steps on the device and syncs once at the end.
  [[nodiscard]] DeviceIcpLoopResult run_loop(
      const DeviceCorrespondenceSearch& search, unsigned int iterations,
      const DeviceIcpLoopParams& params);

 private:
  struct Scratch;

  std::unique_ptr<Scratch> scratch_;
};

using DeviceCorrespondenceSweep =
    BasicDeviceCorrespondenceSweep<ExplicitGraphBuild>;
using CapturedCorrespondenceSweep =
    BasicDeviceCorrespondenceSweep<CapturedGraphBuild>;

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_ICP_CORRESPONDENCE_HPP */
