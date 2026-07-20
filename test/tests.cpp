#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <kinectfusion/block_rep.hpp>
#include <kinectfusion/comparison.hpp>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/grid.hpp>
#include <kinectfusion/icp_correspondence.hpp>
#include <kinectfusion/icp_optimizer.hpp>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/marching_cubes.hpp>
#include <kinectfusion/occupancy.hpp>
#include <kinectfusion/pipeline.hpp>
#include <kinectfusion/pipeline_set.hpp>
#include <kinectfusion/raycasting.hpp>
#include <kinectfusion/rgbd.hpp>
#include <kinectfusion/tsdf_integration.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <kinectfusion/volume_representation.hpp>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

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
  kinectfusion::Surface live;
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

TEST_CASE("Block voxel range visits one block clamped to the resolution",
          "[grid]") {
  const kinectfusion::Size3 resolution{.x = 12, .y = 16, .z = 9};
  const auto blocks = kinectfusion::BlockGrid::for_resolution(resolution);
  constexpr std::size_t kEdge = kinectfusion::kVoxelBlockEdge;
  REQUIRE(blocks.extent() == kinectfusion::Size3{.x = 2, .y = 2, .z = 2});

  // Interior block 0 covers the full 8^3.
  const kinectfusion::BlockVoxels interior{0, blocks, resolution};
  std::vector<kinectfusion::GridIndex> visited;
  for (const kinectfusion::GridIndex index : interior) {
    visited.push_back(index);
  }
  REQUIRE(visited.size() == kinectfusion::kVoxelBlockVolume);
  REQUIRE(visited.front() == kinectfusion::GridIndex{});
  REQUIRE(visited.at(1) == kinectfusion::GridIndex{.x = 1});  // x fastest
  REQUIRE(visited.back() == kinectfusion::GridIndex{.x = 7, .y = 7, .z = 7});

  // Flat block 7 = (1, 1, 1): clamped to 4 x 8 x 1 in-bounds voxels.
  const kinectfusion::BlockVoxels corner{7, blocks, resolution};
  std::size_t count = 0;
  for (const auto [x, y, z] : corner) {
    REQUIRE(x >= kEdge);
    REQUIRE(x < resolution.x);
    REQUIRE(y >= kEdge);
    REQUIRE(z == kEdge);
    ++count;
  }
  REQUIRE(count == (resolution.x - kEdge) * kEdge * (resolution.z - kEdge));
}

TEST_CASE("Size3 bounds checks and unflatten round-trip", "[grid]") {
  const kinectfusion::Size3 extent{.x = 4, .y = 3, .z = 2};
  REQUIRE(extent.contains(kinectfusion::make_vec3f(0.0F, 0.0F, 0.0F)));
  REQUIRE(extent.contains(kinectfusion::make_vec3f(3.5F, 2.5F, 1.5F)));
  REQUIRE_FALSE(extent.contains(kinectfusion::make_vec3f(4.0F, 0.0F, 0.0F)));
  REQUIRE_FALSE(extent.contains(kinectfusion::make_vec3f(-0.5F, 0.0F, 0.0F)));
  REQUIRE(extent.contains(3, 2, 1));
  REQUIRE_FALSE(extent.contains(-1, 0, 0));
  REQUIRE_FALSE(extent.contains(0, 3, 0));
  for (std::size_t flat = 0; flat < extent.x * extent.y * extent.z; ++flat) {
    const auto coords = extent.unflatten(flat);
    REQUIRE((((coords.z * extent.y) + coords.y) * extent.x) + coords.x == flat);
  }
}

TEST_CASE("BlockGrid maps voxels to flat blocks and back", "[grid]") {
  const kinectfusion::Size3 resolution{.x = 17, .y = 8, .z = 9};
  const auto blocks = kinectfusion::BlockGrid::for_resolution(resolution);
  REQUIRE(blocks.extent() == kinectfusion::Size3{.x = 3, .y = 1, .z = 2});
  REQUIRE(blocks.count() == 6);
  for (std::size_t flat = 0; flat < blocks.count(); ++flat) {
    REQUIRE(blocks.flat_of(blocks.coords_of(flat)) == flat);
    const auto base = blocks.voxel_base(flat);
    REQUIRE(blocks.block_of_voxel(base.x, base.y, base.z) == flat);
  }
  REQUIRE(blocks.block_of_voxel(16, 7, 8) ==
          blocks.flat_of({.x = 2, .y = 0, .z = 1}));
  REQUIRE(kinectfusion::BlockGrid::intra_of_voxel(9, 2, 8) ==
          kinectfusion::BlockGrid::intra_of_voxel(1, 2, 0));
}

TEST_CASE("Set-bit enumeration visits exactly the set bits", "[grid]") {
  std::vector<std::size_t> visited;
  kinectfusion::BlockBitmapOps::for_each_set_bit(
      0b1010'0001U, 64, [&](std::size_t index) { visited.push_back(index); });
  REQUIRE(visited == std::vector<std::size_t>{64, 69, 71});
}

TEST_CASE("Marching cubes meshes a synthetic sphere", "[mesh]") {
  const auto geometry = synthetic_volume_geometry();
  kinectfusion::HostVolume volume{geometry};
  const float extent =
      static_cast<float>(geometry.resolution.x) * geometry.voxel_size;
  const kinectfusion::Vec3f center =
      geometry.origin + kinectfusion::make_vec3f(0.5F, 0.5F, 0.5F) * extent;
  const float radius = 0.3F * extent;

  const auto view = volume.view();
  for (const auto [x, y, z] : kinectfusion::GridIndices{geometry.resolution}) {
    const float distance =
        kinectfusion::norm(geometry.cell_center(x, y, z) - center) - radius;
    view.voxel_at(x, y, z) = kinectfusion::Voxel{
        .distance =
            std::clamp(distance / geometry.truncation_distance, -1.0F, 1.0F),
        .weight = 1.0F};
  }

  const auto mesh = kinectfusion::MarchingCubes::extract(volume.view());
  REQUIRE_FALSE(mesh.positions.empty());
  REQUIRE_FALSE(mesh.triangles.empty());
  REQUIRE(mesh.normals.size() == mesh.positions.size());
  REQUIRE(mesh.colors.size() == mesh.positions.size());

  for (std::size_t i = 0; i < mesh.positions.size(); ++i) {
    const kinectfusion::Vec3f radial = mesh.positions[i] - center;
    // Vertices lie on the sphere, and gradient normals point outward.
    REQUIRE(std::abs(kinectfusion::norm(radial) - radius) <
            geometry.voxel_size);
    REQUIRE(kinectfusion::dot(mesh.normals[i], radial) > 0.0F);
  }

  // Consistent outward winding.
  for (const auto& triangle : mesh.triangles) {
    for (const std::uint32_t index : triangle) {
      REQUIRE(index < mesh.positions.size());
    }
    const kinectfusion::Vec3f face = kinectfusion::cross(
        mesh.positions[triangle[1]] - mesh.positions[triangle[0]],
        mesh.positions[triangle[2]] - mesh.positions[triangle[0]]);
    const kinectfusion::Vec3f centroid =
        (mesh.positions[triangle[0]] + mesh.positions[triangle[1]] +
         mesh.positions[triangle[2]]) /
        3.0F;
    REQUIRE(kinectfusion::dot(face, centroid - center) > 0.0F);
  }

  // Welded closed surface: every edge is shared by exactly two triangles,
  // and the Euler characteristic of a sphere is 2.
  std::unordered_map<std::uint64_t, int> edge_uses;
  for (const auto& triangle : mesh.triangles) {
    for (std::size_t corner = 0; corner < 3; ++corner) {
      const std::uint64_t head = triangle.at(corner);
      const std::uint64_t tail = triangle.at((corner + 1) % 3);
      edge_uses[(std::min(head, tail) << 32U) | std::max(head, tail)] += 1;
    }
  }
  for (const auto& [edge, uses] : edge_uses) {
    REQUIRE(uses == 2);
  }
  const auto vertex_count = static_cast<std::ptrdiff_t>(mesh.positions.size());
  const auto edge_count = static_cast<std::ptrdiff_t>(edge_uses.size());
  const auto face_count = static_cast<std::ptrdiff_t>(mesh.triangles.size());
  REQUIRE(vertex_count - edge_count + face_count == 2);
}

TEST_CASE("Marching cubes meshes an integrated depth plane", "[mesh]") {
  const auto depth = flat_depth_image(kFixtureDepth1m);
  kinectfusion::HostVolume volume{synthetic_volume_geometry()};
  const kinectfusion::TsdfIntegrator integrator{};
  integrator.integrate(volume.view(),
                       {.depth = &depth, .intrinsics = synthetic_intrinsics()});

  const auto mesh = kinectfusion::MarchingCubes::extract(volume.view());
  REQUIRE_FALSE(mesh.triangles.empty());
  for (const auto& position : mesh.positions) {
    // The plane lies at 1 m depth.
    REQUIRE(std::abs(position.z - 1.0F) < kSyntheticVoxelSize);
  }
}

TEST_CASE("Marching cubes of an unobserved volume is empty", "[mesh]") {
  const kinectfusion::HostVolume volume{synthetic_volume_geometry()};
  const auto mesh = kinectfusion::MarchingCubes::extract(volume.view());
  REQUIRE(mesh.positions.empty());
  REQUIRE(mesh.triangles.empty());
}

TEST_CASE("Sparse marching cubes matches the dense band mesh", "[mesh]") {
  const auto depth = flat_depth_image(kFixtureDepth1m);
  const auto geometry = synthetic_volume_geometry();
  const kinectfusion::TsdfIntegrationOptions options{
      .mode = kinectfusion::IntegrationMode::kBand};
  const kinectfusion::TsdfRuleVariant rule = kinectfusion::AngleWeightedTsdf{};
  const kinectfusion::DepthFrame frame{.depth = &depth,
                                       .intrinsics = synthetic_intrinsics()};

  kinectfusion::HostBlockRep sparse{geometry};
  sparse.integrate(frame.view(), options, rule);
  kinectfusion::DenseRep<kinectfusion::MemorySpace::kHost> dense{geometry};
  dense.integrate(frame.view(), options, rule);

  auto sparse_mesh = kinectfusion::MarchingCubes::extract(sparse.view());
  auto dense_mesh = kinectfusion::MarchingCubes::extract(dense.view());

  REQUIRE_FALSE(sparse_mesh.triangles.empty());
  REQUIRE(sparse_mesh.triangles.size() == dense_mesh.triangles.size());
  REQUIRE(sparse_mesh.positions.size() == dense_mesh.positions.size());

  const auto sorted = [](std::vector<kinectfusion::Vec3f> positions) {
    std::ranges::sort(positions, [](const auto& lhs, const auto& rhs) {
      return std::tie(lhs.x, lhs.y, lhs.z) < std::tie(rhs.x, rhs.y, rhs.z);
    });
    return positions;
  };
  REQUIRE(sorted(sparse_mesh.positions) == sorted(dense_mesh.positions));
}

TEST_CASE("Marching cubes drops cells below the weight threshold", "[mesh]") {
  const auto depth = flat_depth_image(kFixtureDepth1m);
  kinectfusion::HostVolume volume{synthetic_volume_geometry()};
  const kinectfusion::TsdfIntegrator integrator{};
  integrator.integrate(volume.view(),
                       {.depth = &depth, .intrinsics = synthetic_intrinsics()});

  // One integrated frame leaves every weight at most 1: a threshold of 2
  // must reject everything, a threshold of 1 must keep the plane.
  REQUIRE(kinectfusion::MarchingCubes::extract(volume.view(), 2.0F)
              .triangles.empty());
  REQUIRE_FALSE(kinectfusion::MarchingCubes::extract(volume.view(), 1.0F)
                    .triangles.empty());
}

TEST_CASE("Volume integrates and raycasts a synthetic depth plane",
          "[volume]") {
  const auto depth = flat_depth_image(kFixtureDepth1m);

  kinectfusion::HostVolume volume{synthetic_volume_geometry()};
  const auto intrinsics = synthetic_intrinsics();
  const kinectfusion::TsdfIntegrator integrator{};
  integrator.integrate(volume.view(),
                       {.depth = &depth, .intrinsics = intrinsics});

  REQUIRE(kinectfusion::HostVolumeReduction::observed_voxel_count(
              volume.view()) > 0);

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

  REQUIRE(
      kinectfusion::HostVolumeReduction::observed_voxel_count(
          destination.view()) ==
      kinectfusion::HostVolumeReduction::observed_voxel_count(source.view()));
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

TEST_CASE("Pipeline factory always serves device requests", "[pipeline]") {
  const auto depth = flat_depth_image(kFixtureDepth1m);

  auto creation = kinectfusion::Pipeline::create(
      {.name = "gpu",
       .space = kinectfusion::MemorySpace::kDevice,
       .volume = synthetic_volume_geometry()});
  REQUIRE(creation.pipeline != nullptr);
  // Without CUDA the request degrades to host execution with a reason; on a
  // CUDA build with a live device it is served natively with none. Either
  // way the pipeline below must work.
  REQUIRE((creation.space == kinectfusion::MemorySpace::kDevice) ==
          creation.fallback_reason.empty());

  creation.pipeline->integrate(
      {.depth = &depth, .intrinsics = synthetic_intrinsics()});
  REQUIRE(creation.pipeline->observed_voxel_count() > 0);

  REQUIRE_THROWS_AS(kinectfusion::Pipeline::create(
                        {.name = {}, .volume = synthetic_volume_geometry()}),
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
          kinectfusion::HostVolumeReduction::observed_voxel_count(
              direct_volume.view()));
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
                         {.name = "a", .volume = synthetic_volume_geometry()}},
           .reference = {}}),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      kinectfusion::PipelineSet::create(
          {.pipelines = {{.name = "a", .volume = synthetic_volume_geometry()}},
           .reference = "missing"}),
      std::invalid_argument);
}

TEST_CASE("Pipeline set compares voxel storages in one run", "[pipeline]") {
  const auto intrinsics = synthetic_intrinsics();
  const auto depth = flat_depth_image(kFixtureDepth1m);

  auto set = kinectfusion::PipelineSet::create(
      {.pipelines = {{.name = "float", .volume = synthetic_volume_geometry()},
                     {.name = "quantized",
                      .volume = synthetic_volume_geometry(),
                      .voxel = kinectfusion::VoxelStore::kQuantized,
                      .color = kinectfusion::ColorStore::kNone},
                     {.name = "bf16",
                      .volume = synthetic_volume_geometry(),
                      .voxel = kinectfusion::VoxelStore::kBf16,
                      .color = kinectfusion::ColorStore::kNone}},
       .reference = "float"});

  set.integrate({.depth = &depth, .intrinsics = intrinsics});
  const auto comparisons = set.compare();
  REQUIRE(comparisons.size() == 2);
  for (const auto& comparison : comparisons) {
    // Same observations: the storages agree on which voxels exist.
    REQUIRE(comparison.volume.compared_voxels > 0);
    REQUIRE(comparison.volume.only_primary == 0);
    REQUIRE(comparison.volume.only_reference == 0);
    REQUIRE(comparison.volume.max_distance_delta > 0.0F);
  }
  // int16 differs by its absolute quantum, bf16 by its relative one.
  REQUIRE(comparisons[0].volume.max_distance_delta < 1.0e-4F);
  REQUIRE(comparisons[1].volume.max_distance_delta < 5.0e-3F);
}

TEST_CASE("Band integration matches full integration near the surface",
          "[pipeline]") {
  const auto intrinsics = synthetic_intrinsics();
  const auto depth = flat_depth_image(kFixtureDepth1m);

  auto set = kinectfusion::PipelineSet::create(
      {.pipelines = {{.name = "full", .volume = synthetic_volume_geometry()},
                     {.name = "band",
                      .integration = {.mode =
                                          kinectfusion::IntegrationMode::kBand},
                      .volume = synthetic_volume_geometry()}},
       .reference = "full"});
  set.integrate({.depth = &depth, .intrinsics = intrinsics});

  const auto comparisons = set.compare();
  REQUIRE(comparisons.size() == 1);
  // Shared voxels agree exactly (same per-voxel rule).
  REQUIRE(comparisons.front().volume.compared_voxels > 0);
  REQUIRE(comparisons.front().volume.max_distance_delta == 0.0F);
  // Full mode also carves free space that the band never visits.
  REQUIRE(comparisons.front().volume.only_reference > 0);
  REQUIRE(comparisons.front().volume.only_primary == 0);
}

TEST_CASE("Sparse blocks match dense band integration exactly", "[pipeline]") {
  const auto intrinsics = synthetic_intrinsics();
  const auto depth = flat_depth_image(kFixtureDepth1m);

  auto set = kinectfusion::PipelineSet::create(
      {.pipelines =
           {{.name = "dense-band",
             .integration = {.mode = kinectfusion::IntegrationMode::kBand},
             .volume = synthetic_volume_geometry()},
            {.name = "sparse",
             .integration = {.mode = kinectfusion::IntegrationMode::kBand},
             .volume = synthetic_volume_geometry(),
             .storage = kinectfusion::StorageLayout::kSparse}},
       .reference = "dense-band"});
  set.integrate({.depth = &depth, .intrinsics = intrinsics});

  const auto comparisons = set.compare();
  REQUIRE(comparisons.size() == 1);
  // Same allocation pass, same per-voxel rule: shared voxels are exact.
  REQUIRE(comparisons.front().volume.compared_voxels > 0);
  REQUIRE(comparisons.front().volume.max_distance_delta == 0.0F);
  REQUIRE(comparisons.front().volume.only_primary == 0);
  REQUIRE(comparisons.front().volume.only_reference == 0);
}

TEST_CASE("Sparse pool overflow degrades without corruption", "[pipeline]") {
  const auto intrinsics = synthetic_intrinsics();
  const auto depth = flat_depth_image(kFixtureDepth1m);

  auto set = kinectfusion::PipelineSet::create(
      {.pipelines =
           {{.name = "dense-band",
             .integration = {.mode = kinectfusion::IntegrationMode::kBand},
             .volume = synthetic_volume_geometry()},
            {.name = "tiny",
             .integration = {.mode = kinectfusion::IntegrationMode::kBand},
             .volume = synthetic_volume_geometry(),
             .storage = kinectfusion::StorageLayout::kSparse,
             .sparse_block_capacity = 1}},
       .reference = "dense-band"});
  set.integrate({.depth = &depth, .intrinsics = intrinsics});

  const auto comparisons = set.compare();
  // The one allocated block still matches exactly. All other voxels are
  // missing relative to the reference, never wrong.
  REQUIRE(comparisons.front().volume.max_distance_delta == 0.0F);
  REQUIRE(comparisons.front().volume.only_primary == 0);
  REQUIRE(comparisons.front().volume.only_reference > 0);
}

TEST_CASE("Bitmap-march raycast is bit-identical to the plain march",
          "[pipeline]") {
  const auto intrinsics = synthetic_intrinsics();
  const auto depth = flat_depth_image(kFixtureDepth1m);
  const kinectfusion::RaycastCamera camera{.intrinsics = intrinsics,
                                           .width = kSyntheticImageSize,
                                           .height = kSyntheticImageSize};

  auto march = kinectfusion::Pipeline::create(
      {.name = "march", .volume = synthetic_volume_geometry()});
  auto bitmap = kinectfusion::Pipeline::create(
      {.name = "bitmap",
       .volume = synthetic_volume_geometry(),
       .raycast_backend = kinectfusion::RaycastBackend::kBitmapMarch});
  march.pipeline->integrate({.depth = &depth, .intrinsics = intrinsics});
  bitmap.pipeline->integrate({.depth = &depth, .intrinsics = intrinsics});

  const auto plain = march.pipeline->raycast(camera);
  const auto skipped = bitmap.pipeline->raycast(camera);
  // The dilated skip only removes nullopt samples, so each output byte
  // (including NaN invalid pixels) must match the plain march exactly.
  const auto bytes_equal = [](const auto& lhs, const auto& rhs) {
    return std::memcmp(lhs.data().data(), rhs.data().data(),
                       lhs.data().size() * sizeof(lhs.at(0, 0))) == 0;
  };
  REQUIRE(valid_raycast_pixels(plain) > 0);
  REQUIRE(bytes_equal(plain.points, skipped.points));
  REQUIRE(bytes_equal(plain.normals, skipped.normals));
  REQUIRE(bytes_equal(plain.colors, skipped.colors));
}

TEST_CASE("Pipeline set refuses to substitute a member's memory space",
          "[pipeline]") {
  const auto probe = kinectfusion::Pipeline::create(
      {.name = "probe",
       .space = kinectfusion::MemorySpace::kDevice,
       .volume = synthetic_volume_geometry()});
  const bool device_served = probe.space == kinectfusion::MemorySpace::kDevice;

  const kinectfusion::PipelineSetConfig mixed{
      .pipelines = {{.name = "cpu",
                     .space = kinectfusion::MemorySpace::kHost,
                     .volume = synthetic_volume_geometry()},
                    {.name = "gpu",
                     .space = kinectfusion::MemorySpace::kDevice,
                     .volume = synthetic_volume_geometry()}},
      .reference = "cpu"};

  // Running "gpu" on the host instead would compare the host against itself
  // and report a flawless match for a port that never ran.
  if (device_served) {
    REQUIRE_NOTHROW(kinectfusion::PipelineSet::create(mixed));
  } else {
    REQUIRE_THROWS_AS(kinectfusion::PipelineSet::create(mixed),
                      std::invalid_argument);
  }

  // A lone pipeline is a run, not a comparison, so it may still fall back.
  REQUIRE_NOTHROW(kinectfusion::PipelineSet::create(
      {.pipelines = {{.name = "gpu",
                      .space = kinectfusion::MemorySpace::kDevice,
                      .volume = synthetic_volume_geometry()}},
       .reference = {}}));
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
  const auto surfaces = kinectfusion::HostTrackingSurfaces::from_render(
      view(surface.live), view(surface.model));
  const auto result = tracker.estimate_pose(surfaces, intrinsics,
                                            Eigen::Matrix4f::Identity(), 3);

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
  const auto surfaces = kinectfusion::HostTrackingSurfaces::from_render(
      view(surface.live), view(surface.model));
  const auto result = tracker.estimate_pose({.surfaces = surfaces,
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
  kinectfusion::Surface live_maps{
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
  const auto surfaces = kinectfusion::HostTrackingSurfaces::from_render(
      view(live_maps), view(model_maps));
  const auto result =
      tracker.estimate_pose({.surfaces = surfaces, .iterations = 10});

  REQUIRE_FALSE(result.result.has_value());
  REQUIRE(result.result.error() ==
          kinectfusion::IcpFailure::kTooFewCorrespondences);
  REQUIRE(result.diagnostics.correspondences == 0);
}
