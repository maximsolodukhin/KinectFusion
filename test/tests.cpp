#include <algorithm>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/icp_optimizer.hpp>
#include <kinectfusion/sample_library.hpp>
#include <kinectfusion/volume.hpp>
#include <variant>

namespace {

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
  constexpr unsigned int width = 3;
  constexpr unsigned int height = 3;
  SyntheticSurface surface{
      .live = {.vertices =
                   kinectfusion::image_proc::Vector3fImage{width, height},
               .normals =
                   kinectfusion::image_proc::Vector3fImage{width, height}},
      .model = {
          .points = kinectfusion::image_proc::Vector3fImage{width, height},
          .normals = kinectfusion::image_proc::Vector3fImage{width, height},
          .colors = kinectfusion::image_proc::ColorImage{width, height}}};

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
  for (unsigned int y = 0; y < height; ++y) {
    for (unsigned int x = 0; x < width; ++x) {
      const float z = 1.0F + (0.05F * static_cast<float>(x + y));
      const auto point = kinectfusion::from_eigen(intrinsics.back_project(
          {static_cast<float>(x), static_cast<float>(y)}, z));
      const auto seed_index =
          (static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) +
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

TEST_CASE("Factorials are computed", "[factorial]") {
  REQUIRE(factorial(0) == 1);
  REQUIRE(factorial(1) == 1);
  REQUIRE(factorial(2) == 2);
  REQUIRE(factorial(3) == 6);
  REQUIRE(factorial(10) == 3628800);
}

TEST_CASE("Depth processing back-projects a flat depth image",
          "[depth_processing]") {
  kinectfusion::image_proc::DepthImage depth{3, 3};
  std::ranges::fill(depth.data(), std::uint16_t{5000});

  const kinectfusion::CameraIntrinsics intrinsics{
      .fx = 1.0F, .fy = 1.0F, .cx = 1.0F, .cy = 1.0F};
  const auto vertices = kinectfusion::project_depth_to_vertices(
      depth, intrinsics, Eigen::Matrix4f::Identity());
  const auto center = vertices.at(1, 1);
  const auto right = vertices.at(2, 1);

  REQUIRE(center.x == Catch::Approx(0.0F));
  REQUIRE(center.y == Catch::Approx(0.0F));
  REQUIRE(center.z == Catch::Approx(1.0F));
  REQUIRE(right.x == Catch::Approx(1.0F));
  REQUIRE(right.y == Catch::Approx(0.0F));
  REQUIRE(right.z == Catch::Approx(1.0F));

  const auto normals =
      kinectfusion::compute_normals_central_differences(vertices);
  REQUIRE(normals.at(1, 1).x == Catch::Approx(0.0F));
  REQUIRE(normals.at(1, 1).y == Catch::Approx(0.0F));
  REQUIRE(normals.at(1, 1).z == Catch::Approx(-1.0F));
}

TEST_CASE("Depth pyramid rejects mixed-depth neighborhoods",
          "[depth_processing]") {
  kinectfusion::image_proc::DepthImage depth{2, 2};
  depth.at(0, 0) = 5000;
  depth.at(1, 0) = 7000;
  depth.at(0, 1) = 5000;
  depth.at(1, 1) = 5000;

  const auto level = kinectfusion::build_depth_pyramid_level(depth);

  REQUIRE(level.width() == 1);
  REQUIRE(level.height() == 1);
  REQUIRE(level.at(0, 0) == 0);
}

TEST_CASE("Volume integrates and raycasts a synthetic depth plane",
          "[volume]") {
  constexpr unsigned int width = 16;
  constexpr unsigned int height = 16;
  kinectfusion::image_proc::DepthImage depth{width, height};
  std::ranges::fill(depth.data(), std::uint16_t{5000});

  kinectfusion::Volume volume{kinectfusion::Vector3s{32, 32, 32}, 0.05F,
                              Eigen::Vector3f{-0.8F, -0.8F, 0.2F}, 0.05F};
  const kinectfusion::CameraIntrinsics intrinsics{
      .fx = 20.0F, .fy = 20.0F, .cx = 7.5F, .cy = 7.5F};
  volume.integrate_depth_image(depth, intrinsics, Eigen::Matrix4f::Identity());

  REQUIRE(volume.observed_voxel_count() > 0);

  const auto maps =
      volume.raycast(intrinsics, width, height, Eigen::Matrix4f::Identity());

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
  const auto result = tracker.estimate_pose(
      3, surface.live, surface.model, intrinsics, Eigen::Matrix4f::Identity(),
      Eigen::Matrix4f::Identity());

  REQUIRE(result.result.has_value());
  REQUIRE(std::holds_alternative<kinectfusion::Converged>(*result.result));
  REQUIRE(result.diagnostics.correspondences == 9);
  REQUIRE(result.diagnostics.update_translation ==
          Catch::Approx(0.0F).margin(1.0e-5F));
  REQUIRE(result.diagnostics.update_rotation ==
          Catch::Approx(0.0F).margin(1.0e-5F));
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
      Eigen::AngleAxisf(0.01F, Eigen::Vector3f::UnitZ()).toRotationMatrix();
  initial.block<3, 1>(0, 3) = Eigen::Vector3f{0.002F, -0.001F, 0.0015F};

  const kinectfusion::ProjectiveIcpTracker tracker{
      kinectfusion::ProjectiveIcpOptions{.min_correspondences = 6,
                                         .max_point_distance = 0.1F,
                                         .min_normal_dot = 0.9F,
                                         .min_system_eigenvalue = 1.0e-12F,
                                         .max_condition_number = 1.0e12F}};
  const auto result =
      tracker.estimate_pose(15, surface.live, surface.model, intrinsics,
                            Eigen::Matrix4f::Identity(), initial);

  REQUIRE(result.result.has_value());
  REQUIRE(std::holds_alternative<kinectfusion::Converged>(*result.result));
  REQUIRE(result.diagnostics.correspondences == 9);
  // The recovered pose should return to (near) the identity.
  REQUIRE((result.pose - Eigen::Matrix4f::Identity()).norm() ==
          Catch::Approx(0.0F).margin(1.0e-3F));
  REQUIRE(result.diagnostics.mean_point_distance ==
          Catch::Approx(0.0F).margin(1.0e-4F));
}

TEST_CASE("Projective ICP rejects empty correspondence sets",
          "[projective_icp]") {
  constexpr unsigned int width = 2;
  constexpr unsigned int height = 2;
  kinectfusion::VertexNormalMaps live_maps{
      .vertices = kinectfusion::image_proc::Vector3fImage{width, height},
      .normals = kinectfusion::image_proc::Vector3fImage{width, height}};
  kinectfusion::SurfaceMaps model_maps{
      .points = kinectfusion::image_proc::Vector3fImage{width, height},
      .normals = kinectfusion::image_proc::Vector3fImage{width, height},
      .colors = kinectfusion::image_proc::ColorImage{width, height}};
  std::ranges::fill(live_maps.vertices.data(), kinectfusion::invalid_vector());
  std::ranges::fill(live_maps.normals.data(), kinectfusion::invalid_vector());
  std::ranges::fill(model_maps.points.data(), kinectfusion::invalid_vector());
  std::ranges::fill(model_maps.normals.data(), kinectfusion::invalid_vector());

  const kinectfusion::ProjectiveIcpTracker tracker{
      kinectfusion::ProjectiveIcpOptions{.min_correspondences = 1}};
  const auto result = tracker.estimate_pose(
      10, live_maps, model_maps, kinectfusion::CameraIntrinsics{},
      Eigen::Matrix4f::Identity(), Eigen::Matrix4f::Identity());

  REQUIRE_FALSE(result.result.has_value());
  REQUIRE(result.result.error() ==
          kinectfusion::IcpFailure::too_few_correspondences);
  REQUIRE(result.diagnostics.correspondences == 0);
}
