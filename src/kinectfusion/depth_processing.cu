// CUDA backend for depth_processing. Each kernel is a direct port of the
// matching loop in depth_processing.cpp so the two backends can be checked
// against each other: the pyramid downsample matches bit-for-bit, the
// bilateral filter matches within one raw unit (it uses the __expf intrinsic
// where the CPU uses std::exp), and vertices and normals match within a few
// ULP.

// Device code never touches Eigen; keep nvcc from instantiating Eigen's
// device-annotated paths.
#define EIGEN_NO_CUDA 1

#include <kinectfusion/depth_processing.cuh>

#include "cuda_util.cuh"
#include "depth_processing_common.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace kinectfusion::cuda {
namespace {

using depth_processing_detail::downsample_factor;
using depth_processing_detail::usable_depth_meters;
using depth_processing_detail::validate_options;

// Eigen::Vector3f is tightly packed, so device float3 buffers can be copied
// straight into Vector3fImage storage.
static_assert(sizeof(Eigen::Vector3f) == sizeof(float3));
static_assert(std::is_trivially_copyable_v<CameraIntrinsics>);


constexpr unsigned int block_dim_x = 32U;
constexpr unsigned int block_dim_y = 8U;

// Trivially copyable mirror of DepthProcessingOptions, passed to kernels by
// value with the bilateral weight scales precomputed.
struct KernelOptions {
  float depth_scale;
  float min_depth;
  float max_depth;
  float max_normal_depth_jump;
  float max_downsample_depth_jump;
  float spatial_weight_scale;
  float depth_weight_scale;
  int bilateral_radius;
};

[[nodiscard]] KernelOptions make_kernel_options(
    const DepthProcessingOptions& options) {
  return KernelOptions{
      .depth_scale = options.depth_scale,
      .min_depth = options.min_depth,
      .max_depth = options.max_depth,
      .max_normal_depth_jump = options.max_normal_depth_jump,
      .max_downsample_depth_jump = options.max_downsample_depth_jump,
      .spatial_weight_scale = depth_processing_detail::bilateral_weight_scale(
          options.bilateral_spatial_sigma),
      .depth_weight_scale = depth_processing_detail::bilateral_weight_scale(
          options.bilateral_depth_sigma),
      .bilateral_radius = options.bilateral_radius};
}

__global__ void bilateral_filter_kernel(const std::uint16_t* __restrict__ input,
                                        std::uint16_t* __restrict__ output,
                                        int width, int height,
                                        KernelOptions options) {
  const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (col >= width || row >= height) {
    return;
  }

  const float center =
      usable_depth_meters(input[row * width + col], options);
  if (center <= 0.0F) {
    output[row * width + col] = 0;
    return;
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
      const std::uint16_t raw = input[y * width + x];
      const float sample = usable_depth_meters(raw, options);
      if (sample <= 0.0F) {
        continue;
      }
      const float depth_difference = sample - center;
      const float spatial = static_cast<float>(dx * dx + dy * dy);
      const float weight =
          __expf(spatial * options.spatial_weight_scale +
                 depth_difference * depth_difference *
                     options.depth_weight_scale);
      weighted_sum += weight * static_cast<float>(raw);
      weight_sum += weight;
    }
  }
  output[row * width + col] =
      weight_sum > 0.0F
          ? static_cast<std::uint16_t>(lroundf(weighted_sum / weight_sum))
          : std::uint16_t{0};
}

__device__ std::uint16_t downsample_depth_block(
    const std::uint16_t* __restrict__ input, int input_width, int col, int row,
    const KernelOptions& options) {
  constexpr int factor = static_cast<int>(downsample_factor);
  std::uint32_t sum = 0;
  unsigned int count = 0;
  std::uint16_t minimum = 0xFFFFU;
  std::uint16_t maximum = 0;

  for (int dy = 0; dy < factor; ++dy) {
    for (int dx = 0; dx < factor; ++dx) {
      const std::uint16_t sample =
          input[(row * factor + dy) * input_width + col * factor + dx];
      if (sample == 0) {
        continue;
      }
      sum += sample;
      ++count;
      minimum = min(minimum, sample);
      maximum = max(maximum, sample);
    }
  }

  if (count == 0) {
    return 0;
  }
  if (depth_to_meters(maximum, options.depth_scale) -
          depth_to_meters(minimum, options.depth_scale) >
      options.max_downsample_depth_jump) {
    return 0;
  }
  return static_cast<std::uint16_t>((sum + count / 2U) / count);
}

__global__ void downsample_kernel(const std::uint16_t* __restrict__ input,
                                  std::uint16_t* __restrict__ output,
                                  int input_width, int output_width,
                                  int output_height, KernelOptions options) {
  const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (col >= output_width || row >= output_height) {
    return;
  }
  output[row * output_width + col] =
      downsample_depth_block(input, input_width, col, row, options);
}

__global__ void project_vertices_kernel(const std::uint16_t* __restrict__ depth,
                                        float3* __restrict__ vertices,
                                        int width, int height,
                                        CameraIntrinsics intrinsics,
                                        KernelPose pose,
                                        KernelOptions options) {
  const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (col >= width || row >= height) {
    return;
  }
  const int index = row * width + col;

  const float sample = usable_depth_meters(depth[index], options);
  if (sample <= 0.0F) {
    vertices[index] = nan3();
    return;
  }
  const float3 camera_point{
      (static_cast<float>(col) - intrinsics.cx) * sample / intrinsics.fx,
      (static_cast<float>(row) - intrinsics.cy) * sample / intrinsics.fy,
      sample};
  vertices[index] = transform(pose, camera_point);
}

__global__ void normals_kernel(const float3* __restrict__ vertices,
                               float3* __restrict__ normals, int width,
                               int height, KernelOptions options) {
  const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (col >= width || row >= height) {
    return;
  }
  const int index = row * width + col;

  normals[index] = nan3();
  if (col == 0 || col + 1 >= width || row == 0 || row + 1 >= height) {
    return;
  }
  const float3 center = vertices[index];
  const float3 left = vertices[index - 1];
  const float3 right = vertices[index + 1];
  const float3 top = vertices[index - width];
  const float3 bottom = vertices[index + width];
  if (!is_finite(center) || !is_finite(left) || !is_finite(right) ||
      !is_finite(top) || !is_finite(bottom)) {
    return;
  }
  if (fabsf(right.z - left.z) > options.max_normal_depth_jump ||
      fabsf(bottom.z - top.z) > options.max_normal_depth_jump) {
    return;
  }
  const float3 normal = cross(bottom - top, right - left);
  const float length = norm(normal);
  if (length > 0.0F) {
    normals[index] =
        float3{normal.x / length, normal.y / length, normal.z / length};
  }
}

// ---------------------------------------------------------------------------

struct Workspace {
  void ensure_levels(std::size_t levels) {
    if (depth_levels.size() < levels) {
      depth_levels.resize(levels);
      vertex_levels.resize(levels);
      normal_levels.resize(levels);
    }
  }

  Stream stream;
  // Depth or vertex input; no entry point uploads both.
  DeviceBuffer input;
  PinnedBuffer input_staging;
  PinnedBuffer output_staging;
  std::vector<DeviceBuffer> depth_levels;
  std::vector<DeviceBuffer> vertex_levels;
  std::vector<DeviceBuffer> normal_levels;
};

Workspace& workspace() {
  static thread_local Workspace instance;
  return instance;
}

[[nodiscard]] dim3 grid_for(std::size_t width, std::size_t height) {
  return dim3{
      static_cast<unsigned int>((width + block_dim_x - 1) / block_dim_x),
      static_cast<unsigned int>((height + block_dim_y - 1) / block_dim_y)};
}

[[nodiscard]] constexpr dim3 block_dims() {
  return dim3{block_dim_x, block_dim_y};
}

template <typename Image>
[[nodiscard]] std::size_t image_bytes(const Image& image) {
  return image.data().size() * sizeof(image.data()[0]);
}

void to_device(const void* src, std::size_t bytes, Workspace& ws) {
  ws.input_staging.reserve(bytes);
  std::memcpy(ws.input_staging.as<void>(), src, bytes);
  ws.input.reserve(bytes);
  check(cudaMemcpyAsync(ws.input.as<void>(), ws.input_staging.as<void>(),
                        bytes, cudaMemcpyHostToDevice, ws.stream.get()),
        "input to device");
}

template <typename Image>
void to_device(const Image& image, Workspace& ws) {
  to_device(image.data().data(), image_bytes(image), ws);
}

void to_host(void* dest, const void* src, std::size_t bytes,
             Workspace& ws) {
  check(cudaMemcpyAsync(dest, src, bytes, cudaMemcpyDeviceToHost,
                        ws.stream.get()),
        "result to host");
  check(cudaStreamSynchronize(ws.stream.get()), "cudaStreamSynchronize");
}

template <typename Image>
void to_host(Image& img, const DeviceBuffer& src, Workspace& ws) {
  to_host(img.data().data(), src.as<void>(), image_bytes(img), ws);
}

}  // namespace

image_proc::Vector3fImage compute_normals_central_differences(
    const image_proc::Vector3fImage& vertices,
    const DepthProcessingOptions& options) {
  validate_options(options);
  const std::size_t width = vertices.width();
  const std::size_t height = vertices.height();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  image_proc::Vector3fImage normals{width, height,
                                    Eigen::Vector3f::Constant(nan)};
  if (width < 3U || height < 3U) {
    return normals;
  }

  Workspace& ws = workspace();
  ws.ensure_levels(1);
  to_device(vertices, ws);
  ws.normal_levels[0].reserve(image_bytes(normals));
  normals_kernel<<<grid_for(width, height), block_dims(), 0, ws.stream.get()>>>(
      ws.input.as<float3>(), ws.normal_levels[0].as<float3>(),
      static_cast<int>(width), static_cast<int>(height),
      make_kernel_options(options));
  check(cudaGetLastError(), "normals kernel launch");
  to_host(normals, ws.normal_levels[0], ws);
  return normals;
}

image_proc::DepthImage bilateral_filter_depth(
    const image_proc::DepthImage& depth_image,
    const DepthProcessingOptions& options) {
  validate_options(options);
  if (!options.bilateral_filter || options.bilateral_radius == 0) {
    return depth_image;
  }
  const std::size_t width = depth_image.width();
  const std::size_t height = depth_image.height();
  image_proc::DepthImage filtered{width, height};
  if (width == 0U || height == 0U) {
    return filtered;
  }

  Workspace& ws = workspace();
  ws.ensure_levels(1);
  to_device(depth_image, ws);
  ws.depth_levels[0].reserve(image_bytes(filtered));
  bilateral_filter_kernel<<<grid_for(width, height), block_dims(), 0,
                            ws.stream.get()>>>(
      ws.input.as<std::uint16_t>(), ws.depth_levels[0].as<std::uint16_t>(),
      static_cast<int>(width), static_cast<int>(height),
      make_kernel_options(options));
  check(cudaGetLastError(), "bilateral kernel launch");
  to_host(filtered, ws.depth_levels[0], ws);
  return filtered;
}

image_proc::DepthImage build_depth_pyramid_level(
    const image_proc::DepthImage& depth_image,
    const DepthProcessingOptions& options) {
  validate_options(options);
  const std::size_t width = depth_image.width() / downsample_factor;
  const std::size_t height = depth_image.height() / downsample_factor;
  image_proc::DepthImage downsampled{width, height};
  if (width == 0U || height == 0U) {
    return downsampled;
  }

  Workspace& ws = workspace();
  ws.ensure_levels(1);
  to_device(depth_image, ws);
  ws.depth_levels[0].reserve(image_bytes(downsampled));
  downsample_kernel<<<grid_for(width, height), block_dims(), 0, ws.stream.get()>>>(
      ws.input.as<std::uint16_t>(), ws.depth_levels[0].as<std::uint16_t>(),
      static_cast<int>(depth_image.width()), static_cast<int>(width),
      static_cast<int>(height), make_kernel_options(options));
  check(cudaGetLastError(), "downsample kernel launch");
  to_host(downsampled, ws.depth_levels[0], ws);
  return downsampled;
}

image_proc::Vector3fImage project_depth_to_vertices(
    const image_proc::DepthImage& depth_image,
    const CameraIntrinsics& intrinsics, const Eigen::Matrix4f& camera_pose,
    const DepthProcessingOptions& options) {
  validate_options(options);
  const std::size_t width = depth_image.width();
  const std::size_t height = depth_image.height();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  image_proc::Vector3fImage vertices{width, height,
                                     Eigen::Vector3f::Constant(nan)};
  if (width == 0U || height == 0U) {
    return vertices;
  }

  Workspace& ws = workspace();
  ws.ensure_levels(1);
  to_device(depth_image, ws);
  ws.vertex_levels[0].reserve(image_bytes(vertices));
  project_vertices_kernel<<<grid_for(width, height), block_dims(), 0,
                            ws.stream.get()>>>(
      ws.input.as<std::uint16_t>(), ws.vertex_levels[0].as<float3>(),
      static_cast<int>(width), static_cast<int>(height), intrinsics,
      make_kernel_pose(camera_pose), make_kernel_options(options));
  check(cudaGetLastError(), "project kernel launch");
  to_host(vertices, ws.vertex_levels[0], ws);
  return vertices;
}

std::vector<DepthProcessingLevel> build_surface_pyramid(
    const image_proc::DepthImage& depth_image,
    const CameraIntrinsics& intrinsics, const Eigen::Matrix4f& camera_pose,
    const DepthProcessingOptions& options) {
  validate_options(options);
  std::vector<DepthProcessingLevel> pyramid;

  // Level level_sizes, halving until a dimension collapses (CPU semantics).
  std::vector<std::pair<std::size_t, std::size_t>> level_sizes;
  for (std::size_t w = depth_image.width(), h = depth_image.height();
       level_sizes.size() < options.levels && w != 0U && h != 0U;
       w /= downsample_factor, h /= downsample_factor) {
    level_sizes.emplace_back(w, h);
  }
  if (level_sizes.empty()) {
    return pyramid;
  }

  Workspace& ws = workspace();
  const std::size_t levels = level_sizes.size();
  ws.ensure_levels(levels);
  const KernelOptions kernel_options = make_kernel_options(options);
  const KernelPose kernel_pose = make_kernel_pose(camera_pose);

  to_device(depth_image, ws);

  const bool filter = options.bilateral_filter && options.bilateral_radius > 0;
  std::vector<const std::uint16_t*> level_depths(levels);
  level_depths[0] = ws.input.as<std::uint16_t>();
  if (filter) {
    const auto [width, height] = level_sizes.front();
    ws.depth_levels[0].reserve(width * height * sizeof(std::uint16_t));
    bilateral_filter_kernel<<<grid_for(width, height), block_dims(), 0,
                              ws.stream.get()>>>(
        ws.input.as<std::uint16_t>(), ws.depth_levels[0].as<std::uint16_t>(),
        static_cast<int>(width), static_cast<int>(height), kernel_options);
    check(cudaGetLastError(), "bilateral kernel launch");
    level_depths[0] = ws.depth_levels[0].as<std::uint16_t>();
  }

  // Enqueue every level's kernels, then the downloads, and synchronize once.
  for (std::size_t level = 0; level < levels; ++level) {
    const auto [width, height] = level_sizes[level];
    const std::size_t map_bytes = width * height * sizeof(float3);

    ws.vertex_levels[level].reserve(map_bytes);
    ws.normal_levels[level].reserve(map_bytes);
    project_vertices_kernel<<<grid_for(width, height), block_dims(), 0,
                              ws.stream.get()>>>(
        level_depths[level], ws.vertex_levels[level].as<float3>(),
        static_cast<int>(width), static_cast<int>(height),
        intrinsics.scaled(static_cast<unsigned int>(level)), kernel_pose,
        kernel_options);
    check(cudaGetLastError(), "project kernel launch");
    normals_kernel<<<grid_for(width, height), block_dims(), 0, ws.stream.get()>>>(
        ws.vertex_levels[level].as<float3>(),
        ws.normal_levels[level].as<float3>(), static_cast<int>(width),
        static_cast<int>(height), kernel_options);
    check(cudaGetLastError(), "normals kernel launch");

    if (level + 1U < levels) {
      const auto [next_width, next_height] = level_sizes[level + 1];
      ws.depth_levels[level + 1].reserve(next_width * next_height *
                                         sizeof(std::uint16_t));
      downsample_kernel<<<grid_for(next_width, next_height), block_dims(), 0,
                          ws.stream.get()>>>(
          level_depths[level], ws.depth_levels[level + 1].as<std::uint16_t>(),
          static_cast<int>(width), static_cast<int>(next_width),
          static_cast<int>(next_height), kernel_options);
      check(cudaGetLastError(), "downsample kernel launch");
      level_depths[level + 1] = ws.depth_levels[level + 1].as<std::uint16_t>();
    }
  }

  std::size_t staging_bytes = 0;
  for (std::size_t level = 0; level < levels; ++level) {
    const auto [width, height] = level_sizes[level];
    if (level > 0 || filter) {
      staging_bytes += width * height * sizeof(std::uint16_t);
    }
    staging_bytes += 2U * width * height * sizeof(float3);
  }
  ws.output_staging.reserve(staging_bytes);
  unsigned char* staging = ws.output_staging.as<unsigned char>();

  std::size_t offset = 0;
  for (std::size_t level = 0; level < levels; ++level) {
    const auto [width, height] = level_sizes[level];
    const std::size_t map_bytes = width * height * sizeof(float3);
    if (level > 0 || filter) {
      const std::size_t depth_bytes = width * height * sizeof(std::uint16_t);
      check(cudaMemcpyAsync(staging + offset, level_depths[level], depth_bytes,
                            cudaMemcpyDeviceToHost, ws.stream.get()),
            "depth download");
      offset += depth_bytes;
    }
    check(cudaMemcpyAsync(staging + offset, ws.vertex_levels[level].as<void>(),
                          map_bytes, cudaMemcpyDeviceToHost, ws.stream.get()),
          "vertex download");
    offset += map_bytes;
    check(cudaMemcpyAsync(staging + offset, ws.normal_levels[level].as<void>(),
                          map_bytes, cudaMemcpyDeviceToHost, ws.stream.get()),
          "normal download");
    offset += map_bytes;
  }

  std::vector<image_proc::DepthImage> depth_maps;
  std::vector<image_proc::Vector3fImage> vertex_maps;
  std::vector<image_proc::Vector3fImage> normal_maps;
  depth_maps.reserve(levels);
  vertex_maps.reserve(levels);
  normal_maps.reserve(levels);

  for (unsigned int level = 0; level < levels; ++level) {
    const auto [width, height] = level_sizes[level];
    if (level == 0 && !filter) {
      depth_maps.push_back(depth_image);
    } else {
      depth_maps.emplace_back(width, height);
    }
    vertex_maps.emplace_back(width, height);
    normal_maps.emplace_back(width, height);
  }
  check(cudaStreamSynchronize(ws.stream.get()), "cudaStreamSynchronize");

  pyramid.reserve(levels);
  offset = 0;
  for (std::size_t level = 0; level < levels; ++level) {
    const auto [width, height] = level_sizes[level];
    const std::size_t map_bytes = width * height * sizeof(float3);
    if (level > 0 || filter) {
      const std::size_t depth_bytes = width * height * sizeof(std::uint16_t);
      std::memcpy(depth_maps[level].data().data(), staging + offset,
                  depth_bytes);
      offset += depth_bytes;
    }
    std::memcpy(vertex_maps[level].data().data(), staging + offset, map_bytes);
    offset += map_bytes;
    std::memcpy(normal_maps[level].data().data(), staging + offset, map_bytes);
    offset += map_bytes;
    pyramid.emplace_back(std::move(depth_maps[level]),
                         intrinsics.scaled(static_cast<unsigned int>(level)),
                         std::move(vertex_maps[level]),
                         std::move(normal_maps[level]));
  }
  return pyramid;
}

}  // namespace kinectfusion::cuda
