#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_ICP_CORRESPONDENCE_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_ICP_CORRESPONDENCE_HPP

#include <Eigen/Core>
#include <array>
#include <cstddef>
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

// Accumulated point-plane normal equations (row-major upper triangle of
// J^T J plus J^T r) and the correspondence statistics the diagnostics report.
struct IcpNormalEquations {
  std::size_t count{};
  float distance_sum{};
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
    ++count;
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

// Owns the per-iteration reduction: the device buffers, the private stream, and
// the grid-indexed executable-graph cache. `Build` picks the graph-construction
// strategy without touching the shared reduction kernel.
template <typename Build>
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
