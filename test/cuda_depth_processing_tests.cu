#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <kinectfusion/depth_processing.cuh>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/image_proc/device_image.cuh>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/rgbd.hpp>
#include <kinectfusion/vector.hpp>
#include <utility>
#include <vector>

namespace {

[[nodiscard]] bool has_cuda_device() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

[[nodiscard]] bool same_float(float expected, float actual) {
  return (std::isnan(expected) && std::isnan(actual)) ||
         std::abs(expected - actual) <= 1.0e-6F;
}

[[nodiscard]] bool same_vector(const kinectfusion::Vec3f& expected,
                               const kinectfusion::Vec3f& actual) {
  return same_float(expected.x, actual.x) && same_float(expected.y, actual.y) &&
         same_float(expected.z, actual.z);
}

[[nodiscard]] bool same_vectors(
    const kinectfusion::image_proc::Vector3fImage& expected,
    const kinectfusion::image_proc::DeviceVector3fImage& actual) {
  if (expected.width() != actual.width() ||
      expected.height() != actual.height()) {
    return false;
  }
  kinectfusion::image_proc::Vector3fImage downloaded{actual.width(),
                                                     actual.height()};
  actual.copy_to(downloaded.view());
  for (std::size_t index = 0U; index < expected.data().size(); ++index) {
    if (!same_vector(expected.data().at(index), downloaded.data().at(index))) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool same_pyramid(
    const std::vector<kinectfusion::DepthProcessingLevel<>>& expected,
    const kinectfusion::DepthProcessor<
        kinectfusion::MemorySpace::kDevice>::SurfacePyramid& actual) {
  if (expected.size() != actual.size()) {
    return false;
  }
  for (std::size_t level = 0U; level < expected.size(); ++level) {
    const auto& expected_depth = expected.at(level).depth_image;
    const auto& actual_depth = actual.at(level).depth_image;
    kinectfusion::image_proc::DepthImage downloaded{actual_depth.width(),
                                                    actual_depth.height()};
    actual_depth.copy_to(downloaded.view());
    if (expected_depth.width() != downloaded.width() ||
        expected_depth.height() != downloaded.height() ||
        expected_depth.data() != downloaded.data()) {
      return false;
    }
    const auto& expected_level = expected.at(level);
    const auto& actual_level = actual.at(level);
    if (expected_level.intrinsics.fx != actual_level.intrinsics.fx ||
        expected_level.intrinsics.fy != actual_level.intrinsics.fy ||
        expected_level.intrinsics.cx != actual_level.intrinsics.cx ||
        expected_level.intrinsics.cy != actual_level.intrinsics.cy ||
        !same_vectors(expected_level.maps.vertices,
                      actual_level.maps.vertices) ||
        !same_vectors(expected_level.maps.normals, actual_level.maps.normals)) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  if (!has_cuda_device()) {
    std::cout << "SKIP: no CUDA device is available\n";
    return EXIT_SUCCESS;
  }

  kinectfusion::DepthProcessingOptions options;
  options.levels = 3U;
  options.bilateral_filter = false;

  kinectfusion::image_proc::DepthImage depth{8U, 8U};
  for (std::size_t row = 0U; row < depth.height(); ++row) {
    for (std::size_t col = 0U; col < depth.width(); ++col) {
      depth.at(col, row) =
          static_cast<std::uint16_t>(1000U + (row * depth.width()) + col);
    }
  }
  depth.at(0U, 0U) = 0U;
  depth.at(2U, 2U) = 2000U;
  depth.at(7U, 7U) = 0U;
  const kinectfusion::CameraIntrinsics intrinsics{
      .fx = 525.0F, .fy = 525.0F, .cx = 1.5F, .cy = 1.5F};
  const auto expected =
      kinectfusion::DepthProcessor<kinectfusion::MemorySpace::kHost>{options}
          .build_surface_pyramid(depth, intrinsics);

  kinectfusion::DepthProcessor<kinectfusion::MemorySpace::kDevice> processor{
      options};
  const auto actual = processor.build_surface_pyramid(depth, intrinsics);
  if (!same_pyramid(expected, actual)) {
    std::cerr << "CUDA depth pyramid differs from the CPU implementation\n";
    return EXIT_FAILURE;
  }
  std::cout << "CUDA and CPU depth pyramids match across all levels\n";
  return EXIT_SUCCESS;
}
