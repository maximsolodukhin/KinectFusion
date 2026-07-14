#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <kinectfusion/comparison.hpp>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/grid.hpp>
#include <kinectfusion/icp_optimizer.hpp>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/pipeline.hpp>
#include <kinectfusion/pipeline_set.hpp>
#include <kinectfusion/raycasting.hpp>
#include <kinectfusion/rgbd.hpp>
#include <kinectfusion/tsdf_integration.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <numbers>
#include <stdexcept>
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

// Image size and intrinsics of the synthetic depth camera used by the
// volume/comparison tests; centred on a 16x16 image observing the volume.
constexpr unsigned int kSyntheticImageSize = 16;

[[nodiscard]] kinectfusion::VolumeGeometry synthetic_volume_geometry() {
  return kinectfusion::VolumeGeometry{
      .resolution = {.x = kSyntheticVolumeResolution,
                     .y = kSyntheticVolumeResolution,
                     .z = kSyntheticVolumeResolution},
      .voxel_size = kSyntheticVoxelSize,
      .origin = {.x = kSyntheticOriginX,
                 .y = kSyntheticOriginY,
                 .z = kSyntheticOriginZ},
      .truncation_distance = kSyntheticTruncationDistance};
}

[[nodiscard]] kinectfusion::CameraIntrinsics synthetic_intrinsics() {
  return kinectfusion::CameraIntrinsics{
      .fx = 20.0F, .fy = 20.0F, .cx = 7.5F, .cy = 7.5F};
}

[[nodiscard]] kinectfusion::image_proc::DepthImage flat_depth_image(
    std::uint16_t raw_depth) {
  kinectfusion::image_proc::DepthImage depth{kSyntheticImageSize,
                                             kSyntheticImageSize};
  std::ranges::fill(depth.data(), raw_depth);
  return depth;
}

// Valid raycast pixels compare cleanly against themselves; the library's
// validity rule (finite point and normal) is the single definition.
[[nodiscard]] std::size_t valid_raycast_pixels(
    const kinectfusion::SurfaceMaps& maps) {
  return kinectfusion::Comparator::compare(maps, maps).compared_pixels;
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
          Eigen::Vector2f{static_cast<float>(x), static_cast<float>(y)}, z));
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

TEST_CASE("Grid index ranges walk the index space in storage order", "[grid]") {
  const kinectfusion::GridIndices indices{{.x = 2, .y = 2, .z = 2}};
  std::vector<kinectfusion::GridIndex> visited;
  for (const kinectfusion::GridIndex index : indices) {
    visited.push_back(index);
  }
  REQUIRE(visited.size() == 8);
  REQUIRE(visited.front() == kinectfusion::GridIndex{});
  REQUIRE(visited.at(1) == kinectfusion::GridIndex{.x = 1});
  REQUIRE(visited.at(2) == kinectfusion::GridIndex{.y = 1});
  REQUIRE(visited.at(4) == kinectfusion::GridIndex{.z = 1});
  REQUIRE(visited.back() == kinectfusion::GridIndex{.x = 1, .y = 1, .z = 1});

  // An empty dimension empties the whole range.
  REQUIRE(std::ranges::distance(
              kinectfusion::GridIndices{{.x = 0, .y = 4, .z = 4}}) == 0);

  const kinectfusion::PixelIndices pixels{3, 2};
  REQUIRE(std::ranges::distance(pixels) == 6);
  REQUIRE(*pixels.begin() == kinectfusion::Pixel{});
  REQUIRE(*std::next(pixels.begin()) == kinectfusion::Pixel{.x = 1});
  REQUIRE(*std::next(pixels.begin(), 3) == kinectfusion::Pixel{.y = 1});
}

TEST_CASE("Volume integrates and raycasts a synthetic depth plane",
          "[volume]") {
  const auto depth = flat_depth_image(kFixtureDepth1m);

  kinectfusion::HostVolume volume{synthetic_volume_geometry()};
  const auto intrinsics = synthetic_intrinsics();
  const kinectfusion::TsdfIntegrator integrator{};
  integrator.integrate(volume.view(),
                       {.depth = &depth, .intrinsics = intrinsics});

  REQUIRE(volume.observed_voxel_count() > 0);

  const kinectfusion::Raycaster raycaster{};
  const auto maps =
      raycaster.raycast(volume.view(), {.intrinsics = intrinsics,
                                        .width = kSyntheticImageSize,
                                        .height = kSyntheticImageSize});

  REQUIRE(valid_raycast_pixels(maps) > 0);
}

TEST_CASE("Volume data copies between matching host volumes", "[volume]") {
  const auto depth = flat_depth_image(kFixtureDepth1m);

  kinectfusion::HostVolume source{synthetic_volume_geometry()};
  const kinectfusion::TsdfIntegrator integrator{};
  integrator.integrate(source.view(),
                       {.depth = &depth, .intrinsics = synthetic_intrinsics()});

  kinectfusion::HostVolume destination{synthetic_volume_geometry()};
  destination.copy_from(source);

  REQUIRE(destination.observed_voxel_count() == source.observed_voxel_count());
  const auto deviation =
      kinectfusion::Comparator::compare(destination.view(), source.view());
  REQUIRE(deviation.compared_voxels > 0);
  REQUIRE(deviation.only_primary == 0);
  REQUIRE(deviation.only_reference == 0);
  REQUIRE(deviation.max_distance_delta == 0.0F);
  REQUIRE(deviation.max_weight_delta == 0.0F);

  auto mismatched_geometry = synthetic_volume_geometry();
  mismatched_geometry.resolution.x /= 2;
  kinectfusion::HostVolume mismatched{mismatched_geometry};
  REQUIRE_THROWS_AS(mismatched.copy_from(source), std::invalid_argument);
}

TEST_CASE("TSDF update rules produce measurably different volumes",
          "[comparison]") {
  const auto intrinsics = synthetic_intrinsics();
  const auto near_depth = flat_depth_image(kFixtureDepth1m);
  const auto far_depth = flat_depth_image(kFixtureDepth1p4m);
  // Camera-facing live normals switch the view-angle weighting term on.
  const kinectfusion::image_proc::Vector3fImage normals{
      kSyntheticImageSize, kSyntheticImageSize,
      kinectfusion::make_vec3f(0.0F, 0.0F, -1.0F)};

  // Two planes at different depths: the rules weight the conflicting
  // observations differently, so the fused distances must diverge.
  const auto integrate_two_planes =
      [&](const kinectfusion::TsdfIntegrator& integrator,
          kinectfusion::HostVolume& volume) {
        integrator.integrate(volume.view(), {.depth = &near_depth,
                                             .normals = &normals,
                                             .intrinsics = intrinsics});
        integrator.integrate(volume.view(), {.depth = &far_depth,
                                             .normals = &normals,
                                             .intrinsics = intrinsics});
      };

  kinectfusion::HostVolume classic_volume{synthetic_volume_geometry()};
  integrate_two_planes(
      kinectfusion::TsdfIntegrator{kinectfusion::ClassicTsdf{}},
      classic_volume);

  kinectfusion::HostVolume angle_volume{synthetic_volume_geometry()};
  integrate_two_planes(
      kinectfusion::TsdfIntegrator{kinectfusion::AngleWeightedTsdf{}},
      angle_volume);

  const auto deviation = kinectfusion::Comparator::compare(
      classic_volume.view(), angle_volume.view());
  REQUIRE(deviation.compared_voxels > 0);
  REQUIRE(deviation.only_primary == 0);
  REQUIRE(deviation.only_reference == 0);
  REQUIRE(deviation.max_weight_delta > 0.0F);
  REQUIRE(deviation.max_distance_delta > 0.0F);

  // The same rule and options reconstruct identically.
  kinectfusion::HostVolume repeat_volume{synthetic_volume_geometry()};
  integrate_two_planes(
      kinectfusion::TsdfIntegrator{kinectfusion::AngleWeightedTsdf{}},
      repeat_volume);
  const auto identical = kinectfusion::Comparator::compare(
      angle_volume.view(), repeat_volume.view());
  REQUIRE(identical.compared_voxels > 0);
  REQUIRE(identical.only_primary == 0);
  REQUIRE(identical.only_reference == 0);
  REQUIRE(identical.max_distance_delta == 0.0F);
  REQUIRE(identical.max_weight_delta == 0.0F);
}

TEST_CASE("Pipeline factory falls back to host execution", "[pipeline]") {
  const auto depth = flat_depth_image(kFixtureDepth1m);

  auto creation = kinectfusion::Pipeline::create(
      {.name = "gpu",
       .space = kinectfusion::MemorySpace::kDevice,
       .volume = synthetic_volume_geometry()});
  REQUIRE(creation.pipeline != nullptr);
  // No device support on this branch: the request degrades with a reason.
  REQUIRE_FALSE(creation.fallback_reason.empty());

  creation.pipeline->integrate(
      {.depth = &depth, .intrinsics = synthetic_intrinsics()});
  REQUIRE(creation.pipeline->observed_voxel_count() > 0);

  REQUIRE_THROWS_AS(
      kinectfusion::Pipeline::create({.volume = synthetic_volume_geometry()}),
      std::invalid_argument);
}

TEST_CASE("Factory pipeline matches a directly composed pipeline",
          "[pipeline]") {
  const auto depth = flat_depth_image(kFixtureDepth1m);
  const auto intrinsics = synthetic_intrinsics();

  auto creation = kinectfusion::Pipeline::create(
      {.name = "factory", .volume = synthetic_volume_geometry()});
  creation.pipeline->integrate({.depth = &depth, .intrinsics = intrinsics});

  kinectfusion::HostVolume direct_volume{synthetic_volume_geometry()};
  const kinectfusion::TsdfIntegrator integrator{};
  integrator.integrate(direct_volume.view(),
                       {.depth = &depth, .intrinsics = intrinsics});

  std::optional<kinectfusion::HostVolume> staging;
  const auto deviation = kinectfusion::Comparator::compare(
      creation.pipeline->host_view(staging), direct_volume.view());
  // Host pipelines expose their own storage; staging stays untouched.
  REQUIRE_FALSE(staging.has_value());
  REQUIRE(deviation.compared_voxels > 0);
  REQUIRE(deviation.only_primary == 0);
  REQUIRE(deviation.only_reference == 0);
  REQUIRE(deviation.max_distance_delta == 0.0F);
  REQUIRE(deviation.max_weight_delta == 0.0F);
  REQUIRE(creation.pipeline->observed_voxel_count() ==
          direct_volume.observed_voxel_count());
}

TEST_CASE("Pipeline set compares variants against the reference",
          "[pipeline]") {
  const auto intrinsics = synthetic_intrinsics();
  const auto near_depth = flat_depth_image(kFixtureDepth1m);
  const auto far_depth = flat_depth_image(kFixtureDepth1p4m);
  const kinectfusion::image_proc::Vector3fImage normals{
      kSyntheticImageSize, kSyntheticImageSize,
      kinectfusion::make_vec3f(0.0F, 0.0F, -1.0F)};

  auto set = kinectfusion::PipelineSet::create(
      {.pipelines = {{.name = "baseline",
                      .tsdf_rule = kinectfusion::AngleWeightedTsdf{},
                      .volume = synthetic_volume_geometry()},
                     {.name = "classic",
                      .tsdf_rule = kinectfusion::ClassicTsdf{},
                      .volume = synthetic_volume_geometry()}},
       .reference = "baseline"});

  REQUIRE(set.size() == 2);
  REQUIRE(set.should_compare(0));
  REQUIRE(&set.reference() == set.members().front().pipeline.get());

  set.integrate(
      {.depth = &near_depth, .normals = &normals, .intrinsics = intrinsics});
  set.integrate(
      {.depth = &far_depth, .normals = &normals, .intrinsics = intrinsics});

  const auto outputs = set.raycast_all({.intrinsics = intrinsics,
                                        .width = kSyntheticImageSize,
                                        .height = kSyntheticImageSize});
  REQUIRE(outputs.size() == 2);
  const auto comparisons = set.compare(outputs);
  REQUIRE(comparisons.size() == 1);
  REQUIRE(comparisons.front().name == "classic");
  REQUIRE(comparisons.front().volume.compared_voxels > 0);
  REQUIRE(comparisons.front().volume.max_distance_delta > 0.0F);
  REQUIRE(comparisons.front().surface.has_value());
  REQUIRE(comparisons.front().surface->compared_pixels > 0);

  // Misconfigurations are rejected up front.
  REQUIRE_THROWS_AS(kinectfusion::PipelineSet::create({}),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(
      kinectfusion::PipelineSet::create(
          {.pipelines = {{.name = "a", .volume = synthetic_volume_geometry()},
                         {.name = "a",
                          .volume = synthetic_volume_geometry()}}}),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      kinectfusion::PipelineSet::create(
          {.pipelines = {{.name = "a", .volume = synthetic_volume_geometry()}},
           .reference = "missing"}),
      std::invalid_argument);
}

TEST_CASE("Comparator measures surface map deviations per pixel",
          "[comparison]") {
  constexpr std::size_t kMapSize = 2;
  constexpr float kPointOffset = 0.1F;
  constexpr float kQuarterTurn = std::numbers::pi_v<float> / 2.0F;

  const auto make_maps = [] {
    return kinectfusion::SurfaceMaps{
        .points =
            kinectfusion::image_proc::Vector3fImage{
                kMapSize, kMapSize, kinectfusion::invalid_vec3f()},
        .normals =
            kinectfusion::image_proc::Vector3fImage{
                kMapSize, kMapSize, kinectfusion::invalid_vec3f()},
        .colors = kinectfusion::image_proc::ColorImage{kMapSize, kMapSize}};
  };
  auto primary = make_maps();
  auto reference = make_maps();

  // (0, 0): compared on both sides — offset point, quarter-turn normal.
  primary.points.at(0, 0) = kinectfusion::make_vec3f(0.0F, 0.0F, 1.0F);
  primary.normals.at(0, 0) = kinectfusion::make_vec3f(1.0F, 0.0F, 0.0F);
  reference.points.at(0, 0) =
      kinectfusion::make_vec3f(kPointOffset, 0.0F, 1.0F);
  reference.normals.at(0, 0) = kinectfusion::make_vec3f(0.0F, 1.0F, 0.0F);
  // (1, 0): valid in the primary only.
  primary.points.at(1, 0) = kinectfusion::make_vec3f(0.0F, 0.0F, 1.0F);
  primary.normals.at(1, 0) = kinectfusion::make_vec3f(0.0F, 0.0F, -1.0F);
  // (0, 1): valid in the reference only.
  reference.points.at(0, 1) = kinectfusion::make_vec3f(0.0F, 0.0F, 1.0F);
  reference.normals.at(0, 1) = kinectfusion::make_vec3f(0.0F, 0.0F, -1.0F);
  // (1, 1) stays invalid on both sides.

  const auto comparison = kinectfusion::Comparator::compare(primary, reference);
  REQUIRE(comparison.compared_pixels == 1);
  REQUIRE(comparison.only_primary == 1);
  REQUIRE(comparison.only_reference == 1);
  REQUIRE(comparison.max_point_distance == Catch::Approx(kPointOffset));
  REQUIRE(comparison.mean_point_distance == Catch::Approx(kPointOffset));
  REQUIRE(comparison.max_normal_angle == Catch::Approx(kQuarterTurn));
  REQUIRE(comparison.mean_normal_angle == Catch::Approx(kQuarterTurn));
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
