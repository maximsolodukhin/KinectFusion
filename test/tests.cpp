#include <algorithm>
#include <array>
#include <variant>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/icp_optimizer.hpp>
#include <kinectfusion/sample_library.hpp>
#include <kinectfusion/volume.hpp>

namespace {

[[nodiscard]] std::uint32_t valid_raycast_pixels(
    const kinectfusion::SurfaceMaps& maps) {
  std::uint32_t count = 0;
  const auto& points = maps.points.data();
  const auto& normals = maps.normals.data();
  for (std::size_t i = 0; i < points.size(); ++i) {
    if (points.at(i).allFinite() && normals.at(i).allFinite()) {
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
      .model = {.points =
                    kinectfusion::image_proc::Vector3fImage{width, height},
                .normals =
                    kinectfusion::image_proc::Vector3fImage{width, height},
                .colors = kinectfusion::image_proc::ColorImage{width, height}}};

  const std::array normal_seeds{
      Eigen::Vector3f{1.0F, 0.0F, 0.0F},  Eigen::Vector3f{0.0F, 1.0F, 0.0F},
      Eigen::Vector3f{0.0F, 0.0F, 1.0F},  Eigen::Vector3f{1.0F, 1.0F, 0.0F},
      Eigen::Vector3f{1.0F, 0.0F, 1.0F},  Eigen::Vector3f{0.0F, 1.0F, 1.0F},
      Eigen::Vector3f{1.0F, -1.0F, 0.5F}, Eigen::Vector3f{-1.0F, 0.5F, 1.0F},
      Eigen::Vector3f{0.5F, 1.0F, -1.0F}};
  for (unsigned int y = 0; y < height; ++y) {
    for (unsigned int x = 0; x < width; ++x) {
      const float z = 1.0F + (0.05F * static_cast<float>(x + y));
      const auto point = intrinsics.back_project({static_cast<float>(x),
                                                 static_cast<float>(y)}, z);
      const auto seed_index =
          (static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) +
          static_cast<std::size_t>(x);
      auto normal = normal_seeds.at(seed_index);
      normal.normalize();

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

  REQUIRE(center.x() == Catch::Approx(0.0F));
  REQUIRE(center.y() == Catch::Approx(0.0F));
  REQUIRE(center.z() == Catch::Approx(1.0F));
  REQUIRE(right.x() == Catch::Approx(1.0F));
  REQUIRE(right.y() == Catch::Approx(0.0F));
  REQUIRE(right.z() == Catch::Approx(1.0F));

  const auto normals =
      kinectfusion::compute_normals_central_differences(vertices);
  REQUIRE(normals.at(1, 1).x() == Catch::Approx(0.0F));
  REQUIRE(normals.at(1, 1).y() == Catch::Approx(0.0F));
  REQUIRE(normals.at(1, 1).z() == Catch::Approx(-1.0F));
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

TEST_CASE("Volume integrates and raycasts a synthetic depth plane", "[volume]") {
  constexpr unsigned int width = 16;
  constexpr unsigned int height = 16;
  kinectfusion::image_proc::DepthImage depth{width, height};
  std::ranges::fill(depth.data(), std::uint16_t{5000});

  kinectfusion::Volume volume{
      kinectfusion::Vector3s{32, 32, 32},
      0.05F,
      Eigen::Vector3f{-0.8F, -0.8F, 0.2F},
      0.05F};
  const kinectfusion::CameraIntrinsics intrinsics{
      .fx = 20.0F, .fy = 20.0F, .cx = 7.5F, .cy = 7.5F};
  volume.integrate_depth_image(depth, intrinsics,
                               Eigen::Matrix4f::Identity());

  REQUIRE(volume.observed_voxel_count() > 0);

  const auto maps = volume.raycast(intrinsics, width, height,
                                   Eigen::Matrix4f::Identity());

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

#ifdef KINECTFUSION_CUDA_ENABLED

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>

#include <cuda_runtime_api.h>
#include <Eigen/Geometry>
#include <kinectfusion/depth_processing.cuh>

namespace {

[[nodiscard]] bool cuda_device_available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// Mix of holes, below-range and in-range samples so every validity branch of
// the depth pipeline is exercised.
[[nodiscard]] kinectfusion::image_proc::DepthImage make_random_depth(
    std::size_t width, std::size_t height, unsigned int seed) {
  kinectfusion::image_proc::DepthImage depth{width, height};
  std::mt19937 rng{seed};
  std::uniform_int_distribution<int> selector{0, 9};
  std::uniform_int_distribution<int> in_range{2000, 40000};
  std::uniform_int_distribution<int> below_range{1, 1999};
  for (auto& sample : depth.data()) {
    const int kind = selector(rng);
    if (kind == 0) {
      sample = 0;
    } else if (kind == 1) {
      sample = static_cast<std::uint16_t>(below_range(rng));
    } else {
      sample = static_cast<std::uint16_t>(in_range(rng));
    }
  }
  return depth;
}

[[nodiscard]] Eigen::Matrix4f make_test_pose() {
  const Eigen::AngleAxisf rotation{
      0.3F, Eigen::Vector3f{1.0F, 2.0F, 3.0F}.normalized()};
  return kinectfusion::make_transform_matrix(
      rotation.toRotationMatrix(), Eigen::Vector3f{0.1F, -0.2F, 0.3F});
}

void require_depth_images_close(const kinectfusion::image_proc::DepthImage& a,
                                const kinectfusion::image_proc::DepthImage& b,
                                int max_difference) {
  REQUIRE(a.width() == b.width());
  REQUIRE(a.height() == b.height());
  for (std::size_t i = 0; i < a.data().size(); ++i) {
    const int difference = std::abs(static_cast<int>(a.data()[i]) -
                                    static_cast<int>(b.data()[i]));
    if (difference > max_difference) {
      CAPTURE(i, a.data()[i], b.data()[i]);
      REQUIRE(difference <= max_difference);
    }
  }
}

// Counts pixels whose finite-pattern differs or whose components deviate by
// more than `margin`. Exposed as a count because pose arithmetic may use FMA
// on one side only, which can flip validity thresholds for a stray pixel.
[[nodiscard]] std::size_t count_mismatched_vectors(
    const kinectfusion::image_proc::Vector3fImage& a,
    const kinectfusion::image_proc::Vector3fImage& b, float margin) {
  REQUIRE(a.width() == b.width());
  REQUIRE(a.height() == b.height());
  std::size_t mismatched = 0;
  for (std::size_t i = 0; i < a.data().size(); ++i) {
    const auto& lhs = a.data()[i];
    const auto& rhs = b.data()[i];
    if (lhs.allFinite() != rhs.allFinite()) {
      ++mismatched;
      continue;
    }
    if (lhs.allFinite() && (lhs - rhs).cwiseAbs().maxCoeff() > margin) {
      ++mismatched;
    }
  }
  return mismatched;
}

void require_vector_images_close(
    const kinectfusion::image_proc::Vector3fImage& a,
    const kinectfusion::image_proc::Vector3fImage& b, float margin) {
  REQUIRE(count_mismatched_vectors(a, b, margin) == 0);
}

}  // namespace

TEST_CASE("CUDA bilateral filter matches CPU", "[depth_processing_cuda]") {
  if (!cuda_device_available()) {
    SKIP("No CUDA device available");
  }
  const auto depth = make_random_depth(64, 48, 1234U);
  const auto cpu = kinectfusion::bilateral_filter_depth(depth);
  const auto gpu = kinectfusion::cuda::bilateral_filter_depth(depth);
  // expf vs std::exp can move the rounded result by one raw unit.
  require_depth_images_close(cpu, gpu, 1);
}

TEST_CASE("CUDA pyramid downsample matches CPU exactly",
          "[depth_processing_cuda]") {
  if (!cuda_device_available()) {
    SKIP("No CUDA device available");
  }
  const auto depth = make_random_depth(64, 48, 5678U);
  const auto cpu = kinectfusion::build_depth_pyramid_level(depth);
  const auto gpu = kinectfusion::cuda::build_depth_pyramid_level(depth);
  require_depth_images_close(cpu, gpu, 0);
}

TEST_CASE("CUDA back-projection matches CPU", "[depth_processing_cuda]") {
  if (!cuda_device_available()) {
    SKIP("No CUDA device available");
  }
  const auto depth = make_random_depth(64, 48, 91011U);
  const kinectfusion::CameraIntrinsics intrinsics{
      .fx = 525.0F, .fy = 525.0F, .cx = 31.5F, .cy = 23.5F};
  const auto pose = make_test_pose();
  const auto cpu =
      kinectfusion::project_depth_to_vertices(depth, intrinsics, pose);
  const auto gpu =
      kinectfusion::cuda::project_depth_to_vertices(depth, intrinsics, pose);
  require_vector_images_close(cpu, gpu, 1.0e-4F);
}

TEST_CASE("CUDA normals match CPU", "[depth_processing_cuda]") {
  if (!cuda_device_available()) {
    SKIP("No CUDA device available");
  }
  const auto depth = make_random_depth(64, 48, 1213U);
  const kinectfusion::CameraIntrinsics intrinsics{
      .fx = 525.0F, .fy = 525.0F, .cx = 31.5F, .cy = 23.5F};
  const auto vertices = kinectfusion::project_depth_to_vertices(
      depth, intrinsics, make_test_pose());
  const auto cpu = kinectfusion::compute_normals_central_differences(vertices);
  const auto gpu =
      kinectfusion::cuda::compute_normals_central_differences(vertices);
  require_vector_images_close(cpu, gpu, 1.0e-3F);
}

TEST_CASE("CUDA surface pyramid matches CPU with identity pose",
          "[depth_processing_cuda]") {
  if (!cuda_device_available()) {
    SKIP("No CUDA device available");
  }
  const auto depth = make_random_depth(64, 48, 1415U);
  const kinectfusion::CameraIntrinsics intrinsics{
      .fx = 525.0F, .fy = 525.0F, .cx = 31.5F, .cy = 23.5F};
  // Disable the bilateral filter so the pipeline is fully deterministic: it is
  // the only stage that uses __expf (vs std::exp on the CPU)
  kinectfusion::DepthProcessingOptions options{};
  options.bilateral_filter = false;
  const auto cpu = kinectfusion::build_surface_pyramid(
      depth, intrinsics, Eigen::Matrix4f::Identity(), options);
  const auto gpu = kinectfusion::cuda::build_surface_pyramid(
      depth, intrinsics, Eigen::Matrix4f::Identity(), options);
  REQUIRE(cpu.size() == gpu.size());
  for (std::size_t level = 0; level < cpu.size(); ++level) {
    CAPTURE(level);
    require_depth_images_close(cpu[level].depth_image,
                               gpu[level].depth_image, 0);
    require_vector_images_close(cpu[level].maps.vertices,
                                gpu[level].maps.vertices, 5.0e-4F);
    require_vector_images_close(cpu[level].maps.normals,
                                gpu[level].maps.normals, 2.0e-2F);
  }
}

TEST_CASE("CUDA surface pyramid matches CPU with rotated pose",
          "[depth_processing_cuda]") {
  if (!cuda_device_available()) {
    SKIP("No CUDA device available");
  }
  const auto depth = make_random_depth(64, 48, 1617U);
  const kinectfusion::CameraIntrinsics intrinsics{
      .fx = 525.0F, .fy = 525.0F, .cx = 31.5F, .cy = 23.5F};
  const auto pose = make_test_pose();
  // See the identity-pose case: the bilateral filter is disabled so the CPU/GPU
  // pyramids match exactly, rather than tolerating __expf/std::exp jitter.
  kinectfusion::DepthProcessingOptions options{};
  options.bilateral_filter = false;
  const auto cpu =
      kinectfusion::build_surface_pyramid(depth, intrinsics, pose, options);
  const auto gpu =
      kinectfusion::cuda::build_surface_pyramid(depth, intrinsics, pose, options);
  REQUIRE(cpu.size() == gpu.size());
  for (std::size_t level = 0; level < cpu.size(); ++level) {
    CAPTURE(level);
    require_depth_images_close(cpu[level].depth_image,
                               gpu[level].depth_image, 0);
    require_vector_images_close(cpu[level].maps.vertices,
                                gpu[level].maps.vertices, 5.0e-4F);
    require_vector_images_close(cpu[level].maps.normals,
                                gpu[level].maps.normals, 2.0e-2F);
  }
}

TEST_CASE("CUDA surface pyramid matches CPU on edge-case shapes",
          "[depth_processing_cuda]") {
  if (!cuda_device_available()) {
    SKIP("No CUDA device available");
  }
  const kinectfusion::CameraIntrinsics intrinsics{
      .fx = 525.0F, .fy = 525.0F, .cx = 31.5F, .cy = 23.5F};
  // Bilateral off keeps the comparison exact (see the full-size cases).
  kinectfusion::DepthProcessingOptions options{};
  options.bilateral_filter = false;

  struct Shape {
    std::size_t width;
    std::size_t height;
  };
  const std::array shapes{Shape{1, 1},   Shape{1, 48},  Shape{64, 1},
                          Shape{2, 2},   Shape{3, 3},    Shape{63, 47},
                          Shape{65, 49}};

  for (const auto shape : shapes) {
    CAPTURE(shape.width, shape.height);
    const auto depth = make_random_depth(shape.width, shape.height, 2024U);
    const auto cpu = kinectfusion::build_surface_pyramid(
        depth, intrinsics, Eigen::Matrix4f::Identity(), options);
    const auto gpu = kinectfusion::cuda::build_surface_pyramid(
        depth, intrinsics, Eigen::Matrix4f::Identity(), options);
    REQUIRE(cpu.size() == gpu.size());
    for (std::size_t level = 0; level < cpu.size(); ++level) {
      CAPTURE(level);
      require_depth_images_close(cpu[level].depth_image,
                                 gpu[level].depth_image, 0);
      require_vector_images_close(cpu[level].maps.vertices,
                                  gpu[level].maps.vertices, 5.0e-4F);
      require_vector_images_close(cpu[level].maps.normals,
                                  gpu[level].maps.normals, 2.0e-2F);
    }
  }
}

TEST_CASE("CUDA depth pipeline matches CPU on an all-holes image",
          "[depth_processing_cuda]") {
  if (!cuda_device_available()) {
    SKIP("No CUDA device available");
  }
  kinectfusion::image_proc::DepthImage depth{64, 48};
  std::ranges::fill(depth.data(), std::uint16_t{0});
  const kinectfusion::CameraIntrinsics intrinsics{
      .fx = 525.0F, .fy = 525.0F, .cx = 31.5F, .cy = 23.5F};
  const auto pose = make_test_pose();

  require_depth_images_close(kinectfusion::bilateral_filter_depth(depth),
                             kinectfusion::cuda::bilateral_filter_depth(depth),
                             0);
  require_depth_images_close(
      kinectfusion::build_depth_pyramid_level(depth),
      kinectfusion::cuda::build_depth_pyramid_level(depth), 0);
  require_vector_images_close(
      kinectfusion::project_depth_to_vertices(depth, intrinsics, pose),
      kinectfusion::cuda::project_depth_to_vertices(depth, intrinsics, pose),
      0.0F);

  const auto cpu = kinectfusion::build_surface_pyramid(depth, intrinsics, pose);
  const auto gpu =
      kinectfusion::cuda::build_surface_pyramid(depth, intrinsics, pose);
  REQUIRE(cpu.size() == gpu.size());
  for (std::size_t level = 0; level < cpu.size(); ++level) {
    CAPTURE(level);
    require_vector_images_close(cpu[level].maps.vertices,
                                gpu[level].maps.vertices, 0.0F);
    require_vector_images_close(cpu[level].maps.normals,
                                gpu[level].maps.normals, 0.0F);
  }
}

TEST_CASE("CUDA vs CPU build_surface_pyramid benchmark", "[.][benchmark]") {
  if (!cuda_device_available()) {
    SKIP("No CUDA device available");
  }
  constexpr std::size_t width = 640;
  constexpr std::size_t height = 480;
  kinectfusion::image_proc::DepthImage depth{width, height};
  for (std::size_t y = 0; y < height; ++y) {
    for (std::size_t x = 0; x < width; ++x) {
      const auto fx = static_cast<float>(x);
      const auto fy = static_cast<float>(y);
      const float value =
          5000.0F + (2000.0F * std::sin(fx * 0.01F) * std::cos(fy * 0.01F));
      const bool hole = (x * 7919U + y * 104729U) % 20U == 0U;
      depth.at(x, y) =
          hole ? std::uint16_t{0}
               : static_cast<std::uint16_t>(std::lround(value));
    }
  }
  const kinectfusion::CameraIntrinsics intrinsics{
      .fx = 525.0F, .fy = 525.0F, .cx = 319.5F, .cy = 239.5F};
  const auto pose = make_test_pose();

  const auto time_ms = [&](auto&& build, int iterations) {
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
      const auto pyramid = build();
      REQUIRE(pyramid.size() == 3);
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<double, std::milli>(elapsed).count() /
           iterations;
  };

  // Warm up the GPU (context init, workspace allocations, JIT if any).
  for (int i = 0; i < 5; ++i) {
    (void)kinectfusion::cuda::build_surface_pyramid(depth, intrinsics, pose);
  }

  const double cpu_ms = time_ms(
      [&] {
        return kinectfusion::build_surface_pyramid(depth, intrinsics, pose);
      },
      10);
  const double gpu_ms = time_ms(
      [&] {
        return kinectfusion::cuda::build_surface_pyramid(depth, intrinsics,
                                                         pose);
      },
      100);

  std::cout << "build_surface_pyramid 640x480, 3 levels\n"
            << "  CPU:  " << cpu_ms << " ms/frame\n"
            << "  CUDA: " << gpu_ms << " ms/frame\n"
            << "  speedup: " << cpu_ms / gpu_ms << "x\n";
}

#endif  // KINECTFUSION_CUDA_ENABLED
