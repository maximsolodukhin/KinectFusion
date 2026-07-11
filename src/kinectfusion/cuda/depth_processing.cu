#include <cuda_runtime.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <kinectfusion/cuda/check.cuh>
#include <kinectfusion/depth_processing.cuh>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/image_proc/device_image.cuh>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/vector.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kinectfusion_cuda {
namespace {

constexpr std::size_t kDownsampleFactor = 2U;
constexpr std::uint16_t kMaximumDepthValue = 0xFFFFU;
constexpr std::uint16_t kMinimumDepthValue = 0U;
constexpr unsigned int kBlockWidth = 32U;
constexpr unsigned int kBlockHeight = 8U;

[[nodiscard]] __host__ __device__ __forceinline__ float depth_to_meters(
    std::uint16_t depth, float depth_scale) {
  return static_cast<float>(depth) / depth_scale;
}

void validate_options(const DepthProcessingOptions& options) {
  if (options.levels == 0U) {
    throw std::invalid_argument("Depth pyramid must have at least one level");
  }
  if (options.depth_scale <= 0.0F) {
    throw std::invalid_argument("Depth scale must be positive");
  }
  if (options.min_depth < 0.0F || options.max_depth <= options.min_depth) {
    throw std::invalid_argument("Depth range is invalid");
  }
  if (options.max_normal_depth_jump < 0.0F) {
    throw std::invalid_argument(
        "Normal depth jump threshold must be non-negative");
  }
  if (options.max_downsample_depth_jump < 0.0F) {
    throw std::invalid_argument(
        "Downsample depth jump threshold must be non-negative");
  }
  if (options.bilateral_radius < 0) {
    throw std::invalid_argument("Bilateral filter radius must be non-negative");
  }
  if (options.bilateral_spatial_sigma <= 0.0F) {
    throw std::invalid_argument("Bilateral spatial sigma must be positive");
  }
  if (options.bilateral_depth_sigma <= 0.0F) {
    throw std::invalid_argument("Bilateral depth sigma must be positive");
  }
}

[[nodiscard]] bool usable_depth(std::uint16_t raw,
                                const DepthProcessingOptions& options,
                                float& depth_meters) {
  if (raw == 0U) {
    return false;
  }
  depth_meters = depth_to_meters(raw, options.depth_scale);
  return depth_meters >= options.min_depth && depth_meters <= options.max_depth;
}

[[nodiscard]] __device__ bool downsample_depth_block(
    image_proc::DeviceImageView<const std::uint16_t> depth_image,
    std::size_t col, std::size_t row, const DepthProcessingOptions& options,
    std::uint16_t& output) {
  std::uint32_t sum = 0U;
  unsigned int count = 0U;
  std::uint16_t minimum_observed_depth = kMaximumDepthValue;
  std::uint16_t maximum_observed_depth = kMinimumDepthValue;

  for (std::size_t dy = 0U; dy < kDownsampleFactor; ++dy) {
    for (std::size_t dx = 0U; dx < kDownsampleFactor; ++dx) {
      const std::uint16_t sample = depth_image.at(
          (col * kDownsampleFactor) + dx, (row * kDownsampleFactor) + dy);
      if (sample == 0U) {
        continue;
      }
      sum += sample;
      ++count;
      minimum_observed_depth =
          sample < minimum_observed_depth ? sample : minimum_observed_depth;
      maximum_observed_depth =
          sample > maximum_observed_depth ? sample : maximum_observed_depth;
    }
  }

  if (count == 0U) {
    return false;
  }
  if (depth_to_meters(maximum_observed_depth, options.depth_scale) -
          depth_to_meters(minimum_observed_depth, options.depth_scale) >
      options.max_downsample_depth_jump) {
    return false;
  }

  output = static_cast<std::uint16_t>((sum + (count / 2U)) / count);
  return true;
}

__global__ void downsample_depth_kernel(
    image_proc::DeviceImageView<const std::uint16_t> depth_image,
    DepthProcessingOptions options,
    image_proc::DeviceImageView<std::uint16_t> output) {
  const unsigned int col = (blockIdx.x * blockDim.x) + threadIdx.x;
  const unsigned int row = (blockIdx.y * blockDim.y) + threadIdx.y;
  if (col >= output.width || row >= output.height) {
    return;
  }

  std::uint16_t value = 0U;
  if (downsample_depth_block(depth_image, col, row, options, value)) {
    output.at(col, row) = value;
  }
}

template <typename PixelT>
[[nodiscard]] image_proc::Image<PixelT, MemorySpace::kDevice> upload(
    const image_proc::Image<PixelT, MemorySpace::kHost>& source) {
  image_proc::Image<PixelT, MemorySpace::kDevice> destination{source.width(),
                                                              source.height()};
  destination.copy_from(source.view());
  return destination;
}

[[nodiscard]] DepthProcessor<MemorySpace::kDevice>::SurfacePyramid
upload_pyramid(std::vector<DepthProcessingLevel<>> host_pyramid) {
  DepthProcessor<MemorySpace::kDevice>::SurfacePyramid device_pyramid;
  device_pyramid.reserve(host_pyramid.size());
  for (auto& level : host_pyramid) {
    device_pyramid.emplace_back(upload(level.depth_image), level.intrinsics,
                                upload(level.maps.vertices),
                                upload(level.maps.normals));
  }
  return device_pyramid;
}

}  // namespace

image_proc::Vector3fImage compute_normals_central_differences(
    const image_proc::Vector3fImage& vertices,
    const DepthProcessingOptions& options) {
  validate_options(options);
  image_proc::Vector3fImage normals{vertices.width(), vertices.height(),
                                    invalid_vec3f()};
  if (vertices.width() < 3U || vertices.height() < 3U) {
    return normals;
  }
  for (unsigned int row = 1U; row + 1U < vertices.height(); ++row) {
    for (unsigned int col = 1U; col + 1U < vertices.width(); ++col) {
      const Vec3f& center = vertices.at(col, row);
      const Vec3f& left = vertices.at(col - 1U, row);
      const Vec3f& right = vertices.at(col + 1U, row);
      const Vec3f& top = vertices.at(col, row - 1U);
      const Vec3f& bottom = vertices.at(col, row + 1U);
      if (!all_finite(center) || !all_finite(left) || !all_finite(right) ||
          !all_finite(top) || !all_finite(bottom)) {
        continue;
      }
      if (std::abs(right.z - left.z) > options.max_normal_depth_jump ||
          std::abs(bottom.z - top.z) > options.max_normal_depth_jump) {
        continue;
      }
      const Vec3f normal = cross(bottom - top, right - left);
      const float normal_length = norm(normal);
      if (normal_length > 0.0F) {
        normals.at(col, row) = normal / normal_length;
      }
    }
  }
  return normals;
}

image_proc::DepthImage bilateral_filter_depth(
    const image_proc::DepthImage& depth_image,
    const DepthProcessingOptions& options) {
  validate_options(options);
  if (!options.bilateral_filter || options.bilateral_radius == 0) {
    return depth_image;
  }

  image_proc::DepthImage filtered{depth_image.width(), depth_image.height()};
  const float spatial_sigma2 =
      options.bilateral_spatial_sigma * options.bilateral_spatial_sigma;
  const float depth_sigma2 =
      options.bilateral_depth_sigma * options.bilateral_depth_sigma;
  const float spatial_scale = -0.5F / spatial_sigma2;
  const float depth_weight_scale = -0.5F / depth_sigma2;
  const int width = static_cast<int>(depth_image.width());
  const int height = static_cast<int>(depth_image.height());
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      const auto center = depth_image.at(static_cast<unsigned int>(col),
                                         static_cast<unsigned int>(row));
      float center_meters = 0.0F;
      if (!usable_depth(center, options, center_meters)) {
        continue;
      }
      float weighted_sum = 0.0F;
      float weight_sum = 0.0F;
      for (int dy = -options.bilateral_radius; dy <= options.bilateral_radius;
           ++dy) {
        const int y = row + dy;
        if (y < 0 || y >= height) {
          continue;
        }
        for (int dx = -options.bilateral_radius; dx <= options.bilateral_radius;
             ++dx) {
          const int x = col + dx;
          if (x < 0 || x >= width) {
            continue;
          }
          const auto sample = depth_image.at(static_cast<unsigned int>(x),
                                             static_cast<unsigned int>(y));
          float sample_meters = 0.0F;
          if (!usable_depth(sample, options, sample_meters)) {
            continue;
          }
          const float depth_difference = sample_meters - center_meters;
          const auto spatial = static_cast<float>((dx * dx) + (dy * dy));
          const float weight = std::exp(
              (spatial * spatial_scale) +
              (depth_difference * depth_difference * depth_weight_scale));
          weighted_sum += weight * static_cast<float>(sample);
          weight_sum += weight;
        }
      }
      if (weight_sum > 0.0F) {
        filtered.at(static_cast<unsigned int>(col),
                    static_cast<unsigned int>(row)) =
            static_cast<std::uint16_t>(std::lround(weighted_sum / weight_sum));
      }
    }
  }
  return filtered;
}

image_proc::DepthImage build_depth_pyramid_level(
    const image_proc::DepthImage& depth_image,
    const DepthProcessingOptions& options) {
  validate_options(options);
  image_proc::DepthImage output{depth_image.width() / kDownsampleFactor,
                                depth_image.height() / kDownsampleFactor};
  if (output.width() == 0U || output.height() == 0U) {
    return output;
  }

  auto device_input = upload(depth_image);
  image_proc::DeviceDepthImage device_output{output.width(), output.height()};
  device_output.fill_zero();

  const dim3 block{kBlockWidth, kBlockHeight};
  const dim3 grid{
      static_cast<unsigned int>((output.width() + block.x - 1U) / block.x),
      static_cast<unsigned int>((output.height() + block.y - 1U) / block.y)};
  downsample_depth_kernel<<<grid, block>>>(std::as_const(device_input).view(),
                                           options, device_output.view());
  cuda::check(cudaGetLastError(), "downsample_depth_kernel launch");
  cuda::check(cudaDeviceSynchronize(), "downsample_depth_kernel synchronize");
  device_output.copy_to(output.view());
  return output;
}

image_proc::Vector3fImage project_depth_to_vertices(
    const image_proc::DepthImage& depth_image,
    const CameraIntrinsics& intrinsics, const Eigen::Matrix4f& camera_pose,
    const DepthProcessingOptions& options) {
  validate_options(options);
  image_proc::Vector3fImage vertices{depth_image.width(), depth_image.height(),
                                     invalid_vec3f()};
  for (std::size_t row = 0; row < depth_image.height(); ++row) {
    for (std::size_t col = 0; col < depth_image.width(); ++col) {
      const auto raw = depth_image.at(col, row);
      const float depth = depth_to_meters(raw, options.depth_scale);
      if (raw == 0U || depth < options.min_depth || depth > options.max_depth) {
        continue;
      }
      const Eigen::Vector3f camera_point = intrinsics.back_project(
          {static_cast<float>(col), static_cast<float>(row)}, depth);
      const Eigen::Vector3f world_point =
          (camera_pose * camera_point.homogeneous()).head<3>();
      vertices.at(col, row) = from_eigen(world_point);
    }
  }
  return vertices;
}

std::vector<DepthProcessingLevel<>> build_surface_pyramid(
    const image_proc::DepthImage& depth_image,
    const CameraIntrinsics& intrinsics, const Eigen::Matrix4f& camera_pose,
    const DepthProcessingOptions& options) {
  std::vector<DepthProcessingLevel<>> pyramid;
  image_proc::DepthImage current = bilateral_filter_depth(depth_image, options);
  for (unsigned int level = 0U; level < options.levels; ++level) {
    if (current.width() == 0U || current.height() == 0U) {
      break;
    }
    const CameraIntrinsics level_intrinsics = intrinsics.scaled(level);
    image_proc::Vector3fImage vertices = project_depth_to_vertices(
        current, level_intrinsics, camera_pose, options);
    image_proc::Vector3fImage normals =
        compute_normals_central_differences(vertices, options);
    if (level + 1U < options.levels) {
      pyramid.emplace_back(current, level_intrinsics, std::move(vertices),
                           std::move(normals));
      current = build_depth_pyramid_level(current, options);
    } else {
      pyramid.emplace_back(std::move(current), level_intrinsics,
                           std::move(vertices), std::move(normals));
    }
  }
  return pyramid;
}

}  // namespace kinectfusion_cuda

namespace kinectfusion {

DepthProcessor<MemorySpace::kDevice>::DepthProcessor(
    DepthProcessingOptions options)
    : options_(options) {
  kinectfusion_cuda::validate_options(options_);
}

auto DepthProcessor<MemorySpace::kDevice>::build_surface_pyramid(
    const image_proc::DepthImage& depth, const CameraIntrinsics& intrinsics)
    -> SurfacePyramid {
  if (!input_ || input_->width() != depth.width() ||
      input_->height() != depth.height()) {
    input_.emplace(depth.width(), depth.height());
  }
  input_->copy_from(depth.view());
  return std::as_const(*this).build_surface_pyramid(
      std::as_const(*input_).view(), intrinsics);
}

auto DepthProcessor<MemorySpace::kDevice>::build_surface_pyramid(
    image_proc::DeviceImageView<const std::uint16_t> depth,
    const CameraIntrinsics& intrinsics) const -> SurfacePyramid {
  image_proc::DepthImage host_depth{depth.width, depth.height};
  if (depth.width != 0U && depth.height != 0U) {
    cuda::check(cudaMemcpy(host_depth.data().data(), depth.data,
                           depth.width * depth.height * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy(DepthProcessor device to host)");
  }
  return kinectfusion_cuda::upload_pyramid(
      kinectfusion_cuda::build_surface_pyramid(
          host_depth, intrinsics, Eigen::Matrix4f::Identity(), options_));
}

}  // namespace kinectfusion
