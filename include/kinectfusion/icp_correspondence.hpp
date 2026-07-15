#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_ICP_CORRESPONDENCE_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_ICP_CORRESPONDENCE_HPP

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <kinectfusion/cuda_compat.hpp>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/rgbd.hpp>
#include <kinectfusion/vector.hpp>
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
  Mat3f rotation{};
  Vec3f translation{};
  Mat3f model_rotation{};
  Vec3f model_translation{};

  [[nodiscard]] static IcpIterationTransforms from_poses(
      const TranslateMat& camera_to_world,
      const TranslateMat& model_world_to_camera) {
    auto rot = Eigen::Matrix3f{camera_to_world.block<3, 3>(0, 0)};
    auto model_rot = Eigen::Matrix3f{model_world_to_camera.block<3, 3>(0, 0)};
    auto trans = Eigen::Vector3f{camera_to_world.block<3, 1>(0, 3)};
    auto model_trans = Eigen::Vector3f{model_world_to_camera.block<3, 1>(0, 3)};

    return {.rotation = from_eigen(rot),
            .translation = from_eigen(trans),
            .model_rotation = from_eigen(model_rot),
            .model_translation = from_eigen(model_trans)};
  }
};

static_assert(std::is_trivially_copyable_v<IcpIterationTransforms>);

struct CorrespondenceGates {
  float max_point_distance{kDefaultMaxIcpPointDistanceMeters};
  float min_normal_dot{kDefaultMinIcpNormalDot};
};

// Projects one live vertex into the model frame and gates the surface pair it
// lands on; nullopt when the pixel is unmeasured, the projection misses the
// model image, or the distance/normal checks reject the pair.
template <MemorySpace Space = MemorySpace::kHost>
class CorrespondenceSearch {
 public:
  CorrespondenceSearch(const VertexNormalMapsView<Space, true>& live,
                       const VertexNormalMapsView<Space, true>& model,
                       const CameraIntrinsics& model_intrinsics,
                       const IcpIterationTransforms& transforms,
                       const CorrespondenceGates& gates)
      : live_(live),
        model_(model),
        intrinsics_(model_intrinsics),
        transforms_(transforms),
        gates_(gates) {}

  [[nodiscard]] KINECTFUSION_HOST_DEVICE compat::optional<IcpCorrespondence>
  match(std::size_t x, std::size_t y) const {
    const Vec3f& live_vertex = live_.vertices.at(x, y);
    const Vec3f& live_normal = live_.normals.at(x, y);
    if (!all_finite(live_vertex) || !all_finite(live_normal)) {
      return compat::nullopt;
    }

    const Vec3f source =
        (transforms_.rotation * live_vertex) + transforms_.translation;
    const Vec3f source_in_model =
        (transforms_.model_rotation * source) + transforms_.model_translation;
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
    if (col >= model_.vertices.width || row >= model_.vertices.height) {
      return compat::nullopt;
    }

    const Vec3f& model_vertex = model_.vertices.at(col, row);
    const Vec3f& model_normal = model_.normals.at(col, row);
    if (!all_finite(model_vertex) || !all_finite(model_normal)) {
      return compat::nullopt;
    }

    const Vec3f source_normal = normalized(transforms_.rotation * live_normal);
    const float distance = norm(source - model_vertex);
    if (distance > gates_.max_point_distance) {
      return compat::nullopt;
    }
    if (dot(source_normal, model_normal) < gates_.min_normal_dot) {
      return compat::nullopt;
    }

    const Vec3f rotational = cross(source, model_normal);
    return IcpCorrespondence{
        .jacobian = {rotational.x, rotational.y, rotational.z, model_normal.x,
                     model_normal.y, model_normal.z},
        .residual = dot(model_normal, model_vertex - source),
        .distance = distance};
  }

 private:
  VertexNormalMapsView<Space, true> live_;
  VertexNormalMapsView<Space, true> model_;
  CameraIntrinsics intrinsics_;
  IcpIterationTransforms transforms_;
  CorrespondenceGates gates_;
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_ICP_CORRESPONDENCE_HPP */
