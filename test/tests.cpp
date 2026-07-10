#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/icp_optimizer.hpp>
#include <kinectfusion/volume.hpp>
#include <utility>
#include <variant>

namespace {

// Fixture depth values in raw TUM units (5000 raw units == 1 meter at the
// default depth scale).
constexpr std::uint16_t kFixtureDepth1m = 5000;
constexpr std::uint16_t kFixtureDepth1p4m = 7000;

// Synthetic TSDF volume geometry used by the raycast test.
constexpr std::size_t kSyntheticVolumeResolution = 32;
constexpr float kSyntheticVoxelSize = 0.05F;
constexpr float kSyntheticOriginX = -0.8F;
constexpr float kSyntheticOriginY = -0.8F;
constexpr float kSyntheticOriginZ = 0.2F;
constexpr float kSyntheticTruncationDistance = 0.05F;

// Tolerances for the ICP convergence tests.
constexpr float kIcpIdentityPoseTolerance = 1.0e-5F;
constexpr float kIcpPerturbationPoseTolerance = 1.0e-3F;
constexpr float kIcpPerturbationDistanceTolerance = 1.0e-4F;

// Perturbation applied to the initial pose in the recovery test.
constexpr float kIcpPerturbationAngle = 0.01F;
constexpr float kIcpPerturbationTx = 0.002F;
constexpr float kIcpPerturbationTy = -0.001F;
constexpr float kIcpPerturbationTz = 0.0015F;

[[nodiscard]] std::uint32_t valid_raycast_pixels(
    const kinectfusion::SurfaceMaps& maps) {
  std::uint32_t count = 0;
  const auto& points = maps.points.data();
  const auto& normals = maps.normals.data();
  for (std::size_t i = 0; i < points.size(); ++i) {
    if (kinectfusion::all_finite(points.at(i)) &&
        kinectfusion::all_finite(normals.at(i))) {
      ++count;
    }
  }
  return count;
}

struct SyntheticSurface {
  kinectfusion::VertexNormalMaps live;
  kinectfusion::SurfaceMaps model;
};

// Build a 3x3 patch where the live frame and the model surface are identical,
// so the true relative pose is the identity. Depths vary across the patch and
// the per-pixel normals are spread over many directions, which keeps the 6-DOF
// point-to-plane system full rank.
[[nodiscard]] SyntheticSurface make_identical_surface(
    const kinectfusion::CameraIntrinsics& intrinsics) {
  constexpr unsigned int kWidth = 3;
  constexpr unsigned int kHeight = 3;
  SyntheticSurface surface{
      .live = {.vertices =
                   kinectfusion::image_proc::Vector3fImage{kWidth, kHeight},
               .normals =
                   kinectfusion::image_proc::Vector3fImage{kWidth, kHeight}},
      .model = {
          .points = kinectfusion::image_proc::Vector3fImage{kWidth, kHeight},
          .normals = kinectfusion::image_proc::Vector3fImage{kWidth, kHeight},
          .colors = kinectfusion::image_proc::ColorImage{kWidth, kHeight}}};

  const std::array normal_seeds{
      kinectfusion::Vec3f{.x = 1.0F, .y = 0.0F, .z = 0.0F},
      kinectfusion::Vec3f{.x = 0.0F, .y = 1.0F, .z = 0.0F},
      kinectfusion::Vec3f{.x = 0.0F, .y = 0.0F, .z = 1.0F},
      kinectfusion::Vec3f{.x = 1.0F, .y = 1.0F, .z = 0.0F},
      kinectfusion::Vec3f{.x = 1.0F, .y = 0.0F, .z = 1.0F},
      kinectfusion::Vec3f{.x = 0.0F, .y = 1.0F, .z = 1.0F},
      kinectfusion::Vec3f{.x = 1.0F, .y = -1.0F, .z = 0.5F},
      kinectfusion::Vec3f{.x = -1.0F, .y = 0.5F, .z = 1.0F},
      kinectfusion::Vec3f{.x = 0.5F, .y = 1.0F, .z = -1.0F}};
  for (unsigned int y = 0; y < kHeight; ++y) {
    for (unsigned int x = 0; x < kWidth; ++x) {
      const float z = 1.0F + (0.05F * static_cast<float>(x + y));
      const auto point = kinectfusion::from_eigen(intrinsics.back_project(
          {static_cast<float>(x), static_cast<float>(y)}, z));
      const auto seed_index =
          (static_cast<std::size_t>(y) * static_cast<std::size_t>(kWidth)) +
          static_cast<std::size_t>(x);
      const auto normal = kinectfusion::normalized(normal_seeds.at(seed_index));

      surface.live.vertices.at(x, y) = point;
      surface.live.normals.at(x, y) = normal;
      surface.model.points.at(x, y) = point;
      surface.model.normals.at(x, y) = normal;
    }
  }

  return surface;
}

}  // namespace

TEST_CASE("Depth processing back-projects a flat depth image",
          "[depth_processing]") {
  kinectfusion::image_proc::DepthImage depth{3, 3};
  std::ranges::fill(depth.data(), kFixtureDepth1m);

  const kinectfusion::CameraIntrinsics intrinsics{
      .fx = 1.0F, .fy = 1.0F, .cx = 1.0F, .cy = 1.0F};
  const kinectfusion::DepthProcessor processor{};
  const auto vertices = processor.project_to_vertices(depth, intrinsics);
  const auto center = vertices.at(1, 1);
  const auto right = vertices.at(2, 1);

  REQUIRE(center.x == Catch::Approx(0.0F));
  REQUIRE(center.y == Catch::Approx(0.0F));
  REQUIRE(center.z == Catch::Approx(1.0F));
  REQUIRE(right.x == Catch::Approx(1.0F));
  REQUIRE(right.y == Catch::Approx(0.0F));
  REQUIRE(right.z == Catch::Approx(1.0F));

  const auto normals = processor.compute_normals(vertices);
  REQUIRE(normals.at(1, 1).x == Catch::Approx(0.0F));
  REQUIRE(normals.at(1, 1).y == Catch::Approx(0.0F));
  REQUIRE(normals.at(1, 1).z == Catch::Approx(-1.0F));
}

TEST_CASE("Depth pyramid rejects mixed-depth neighborhoods",
          "[depth_processing]") {
  kinectfusion::image_proc::DepthImage depth{2, 2};
  depth.at(0, 0) = kFixtureDepth1m;
  depth.at(1, 0) = kFixtureDepth1p4m;
  depth.at(0, 1) = kFixtureDepth1m;
  depth.at(1, 1) = kFixtureDepth1m;

  const auto level = kinectfusion::DepthProcessor{}.downsample(depth);

  REQUIRE(level.width() == 1);
  REQUIRE(level.height() == 1);
  REQUIRE(level.at(0, 0) == 0);
}

TEST_CASE("Volume integrates and raycasts a synthetic depth plane",
          "[volume]") {
  constexpr unsigned int kWidth = 16;
  constexpr unsigned int kHeight = 16;
  kinectfusion::image_proc::DepthImage depth{kWidth, kHeight};
  std::ranges::fill(depth.data(), kFixtureDepth1m);

  kinectfusion::Volume volume{
      kinectfusion::Vector3s{kSyntheticVolumeResolution,
                             kSyntheticVolumeResolution,
                             kSyntheticVolumeResolution},
      kSyntheticVoxelSize,
      Eigen::Vector3f{kSyntheticOriginX, kSyntheticOriginY, kSyntheticOriginZ},
      kSyntheticTruncationDistance};
  const kinectfusion::CameraIntrinsics intrinsics{
      .fx = 20.0F, .fy = 20.0F, .cx = 7.5F, .cy = 7.5F};
  volume.integrate_depth_image({.depth = &depth, .intrinsics = intrinsics});

  REQUIRE(volume.observed_voxel_count() > 0);

  const auto maps =
      volume.raycast(intrinsics, kWidth, kHeight, Eigen::Matrix4f::Identity());

  REQUIRE(valid_raycast_pixels(maps) > 0);
}

TEST_CASE("Projective ICP converges on identical synthetic maps",
          "[projective_icp]") {
  const kinectfusion::CameraIntrinsics intrinsics{
      .fx = 100.0F, .fy = 100.0F, .cx = 1.0F, .cy = 1.0F};
  const auto surface = make_identical_surface(intrinsics);

  const kinectfusion::ProjectiveIcpTracker tracker{
      kinectfusion::ProjectiveIcpOptions{.min_correspondences = 6,
                                         .max_point_distance = 0.1F,
                                         .min_normal_dot = 0.99F,
                                         .min_system_eigenvalue = 1.0e-12F,
                                         .max_condition_number = 1.0e12F}};
  const auto result =
      tracker.estimate_pose({.live = view(surface.live),
                             .model = {.vertices = surface.model.points.view(),
                                       .normals = surface.model.normals.view()},
                             .model_intrinsics = intrinsics,
                             .iterations = 3});

  REQUIRE(result.result.has_value());
  REQUIRE(std::holds_alternative<kinectfusion::Converged>(*result.result));
  REQUIRE(result.diagnostics.correspondences == 9);
  REQUIRE(result.diagnostics.update_translation ==
          Catch::Approx(0.0F).margin(kIcpIdentityPoseTolerance));
  REQUIRE(result.diagnostics.update_rotation ==
          Catch::Approx(0.0F).margin(kIcpIdentityPoseTolerance));
}

TEST_CASE("Projective ICP recovers a small pose perturbation",
          "[projective_icp]") {
  const kinectfusion::CameraIntrinsics intrinsics{
      .fx = 100.0F, .fy = 100.0F, .cx = 1.0F, .cy = 1.0F};
  const auto surface = make_identical_surface(intrinsics);

  // The true relative pose is the identity; start the tracker from a small
  // rotation + translation offset and require it to drive the pose back.
  Eigen::Matrix4f initial = Eigen::Matrix4f::Identity();
  initial.block<3, 3>(0, 0) =
      Eigen::AngleAxisf(kIcpPerturbationAngle, Eigen::Vector3f::UnitZ())
          .toRotationMatrix();
  initial.block<3, 1>(0, 3) = Eigen::Vector3f{
      kIcpPerturbationTx, kIcpPerturbationTy, kIcpPerturbationTz};

  const kinectfusion::ProjectiveIcpTracker tracker{
      kinectfusion::ProjectiveIcpOptions{.min_correspondences = 6,
                                         .max_point_distance = 0.1F,
                                         .min_normal_dot = 0.9F,
                                         .min_system_eigenvalue = 1.0e-12F,
                                         .max_condition_number = 1.0e12F}};
  const auto result =
      tracker.estimate_pose({.live = view(surface.live),
                             .model = {.vertices = surface.model.points.view(),
                                       .normals = surface.model.normals.view()},
                             .model_intrinsics = intrinsics,
                             .initial_camera_to_world = initial,
                             .iterations = 15});

  REQUIRE(result.result.has_value());
  REQUIRE(std::holds_alternative<kinectfusion::Converged>(*result.result));
  REQUIRE(result.diagnostics.correspondences == 9);
  // The recovered pose should return to (near) the identity.
  REQUIRE((result.pose - Eigen::Matrix4f::Identity()).norm() ==
          Catch::Approx(0.0F).margin(kIcpPerturbationPoseTolerance));
  REQUIRE(result.diagnostics.mean_point_distance ==
          Catch::Approx(0.0F).margin(kIcpPerturbationDistanceTolerance));
}

TEST_CASE("Projective ICP rejects empty correspondence sets",
          "[projective_icp]") {
  constexpr unsigned int kWidth = 2;
  constexpr unsigned int kHeight = 2;
  kinectfusion::VertexNormalMaps live_maps{
      .vertices = kinectfusion::image_proc::Vector3fImage{kWidth, kHeight},
      .normals = kinectfusion::image_proc::Vector3fImage{kWidth, kHeight}};
  kinectfusion::SurfaceMaps model_maps{
      .points = kinectfusion::image_proc::Vector3fImage{kWidth, kHeight},
      .normals = kinectfusion::image_proc::Vector3fImage{kWidth, kHeight},
      .colors = kinectfusion::image_proc::ColorImage{kWidth, kHeight}};
  std::ranges::fill(live_maps.vertices.data(), kinectfusion::invalid_vec3f());
  std::ranges::fill(live_maps.normals.data(), kinectfusion::invalid_vec3f());
  std::ranges::fill(model_maps.points.data(), kinectfusion::invalid_vec3f());
  std::ranges::fill(model_maps.normals.data(), kinectfusion::invalid_vec3f());

  const kinectfusion::ProjectiveIcpTracker tracker{
      kinectfusion::ProjectiveIcpOptions{.min_correspondences = 1}};
  const auto result = tracker.estimate_pose(
      {.live = view(std::as_const(live_maps)),
       .model = {.vertices = std::as_const(model_maps).points.view(),
                 .normals = std::as_const(model_maps).normals.view()},
       .iterations = 10});

  REQUIRE_FALSE(result.result.has_value());
  REQUIRE(result.result.error() ==
          kinectfusion::IcpFailure::kTooFewCorrespondences);
  REQUIRE(result.diagnostics.correspondences == 0);
}
