#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <kinectfusion/cuda/check.cuh>
#include <kinectfusion/cuda/launch.cuh>
#include <kinectfusion/registered_storages.hpp>
#include <kinectfusion/tsdf_integration.cuh>
#include <variant>

#include "band_walk.cuh"

namespace kinectfusion {

namespace {

// Perspective projection maps 3D lines to 2D lines. Thus a z-column whose
// two endpoints both land behind the camera or beyond the same image edge
// misses the depth image entirely: every observe() in it would be nullopt.
// The cull is exact (bit-identical output) and removes most of the volume
// every frame from the projection-bound integrate kernel.
template <VoxelGridView VolumeViewT>
[[nodiscard]] __device__ bool column_visible(
    const VolumeViewT& volume, const DeviceIntegrationContext& context,
    std::size_t x, std::size_t y) {
  const Size3 resolution = volume.resolution();
  Vec3f base = context.to_camera(volume.cell_center(x, y, 0));
  Vec3f top = context.to_camera(volume.cell_center(x, y, resolution.z - 1));
  // Clip to just in front of the camera plane. The clip is exact: measured
  // depth is at least min_depth, so the signed distance of a closer voxel
  // exceeds the truncation band and it can never fuse.
  constexpr float kNearClip = 1.0e-3F;
  if (base.z <= kNearClip && top.z <= kNearClip) {
    return false;
  }

  if (base.z < kNearClip) {
    base = top + ((base - top) * ((top.z - kNearClip) / (top.z - base.z)));
  } else if (top.z < kNearClip) {
    top = base + ((top - base) * ((base.z - kNearClip) / (base.z - top.z)));
  }

  const auto& intrinsics = context.frame().intrinsics;
  const Vec2f from = intrinsics.project(base);
  const Vec2f to = intrinsics.project(top);
  const auto width = static_cast<float>(context.frame().depth.width);
  const auto height = static_cast<float>(context.frame().depth.height);
  // Half-pixel margin matches project_to_pixel's lround bounds check.
  const float margin = 0.5F;
  if ((from.x < -margin && to.x < -margin) ||
      (from.y < -margin && to.y < -margin) ||
      (from.x >= width - margin && to.x >= width - margin) ||
      (from.y >= height - margin && to.y >= height - margin)) {
    return false;
  }

  return true;
}

// One thread per (x, y) column sweeping z
template <VoxelGridView VolumeViewT, TsdfUpdateRule<MemorySpace::kDevice> Rule>
__global__ void integrate_kernel(VolumeViewT volume,
                                 DeviceIntegrationContext context, Rule rule) {
  const std::size_t x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const std::size_t y = (blockIdx.y * blockDim.y) + threadIdx.y;
  const Size3 resolution = volume.resolution();
  if (x >= resolution.x || y >= resolution.y) {
    return;
  }
  if (!column_visible(volume, context, x, y)) {
    return;
  }
  // Camera depth is affine along the column, and a voxel can only fuse
  // while its depth lies in (~0, max_depth + truncation]. The clamp of the
  // z sweep is exact and skips the far tail of each visible column.
  const TsdfIntegrationOptions& options = context.options();

  const float depth_first = context.to_camera(volume.cell_center(x, y, 0)).z;
  const float depth_last =
      context.to_camera(volume.cell_center(x, y, resolution.z - 1)).z;

  const float steps = static_cast<float>(resolution.z - 1);
  const float slope = (depth_last - depth_first) / steps;

  const float truncation =
      options.truncation_for(volume.truncation_distance(), options.max_depth);
  const float max_fusable = options.max_depth + truncation;

  float z_begin = 0.0F;
  float z_end = steps;

  if (slope > 1.0e-12F || slope < -1.0e-12F) {
    const float at_near = (1.0e-3F - depth_first) / slope;
    const float at_far = (max_fusable - depth_first) / slope;
    z_begin = compat::max(0.0F, compat::min(at_near, at_far));
    z_end = compat::min(steps, compat::max(at_near, at_far));
  } else if (depth_first <= 1.0e-3F || depth_first > max_fusable) {
    return;
  }

  const auto z_first = static_cast<std::size_t>(compat::max(z_begin, 0.0F));
  const auto z_last = static_cast<std::size_t>(z_end);

  for (std::size_t z = z_first; z <= z_last && z < resolution.z; ++z) {
    rule.update(volume, context, x, y, z);
  }
}

// A null-data source empties the image.
template <typename Img, typename View>
  requires image_proc::RefillableFrom<Img, View>
void assign_image(Img& image, const View& source) {
  if (source.data == nullptr) {
    image = {};
    return;
  }
  image.ensure_extent(source.width, source.height);
  image.copy_from(source);
}

}  // namespace

void DeviceDepthFrame::assign(const DepthFrame& frame) {
  const HostDepthFrameView host = frame.view();
  intrinsics_ = host.intrinsics;
  world_to_camera_ = host.world_to_camera;

  assign_image(depth_, host.depth);
  assign_image(color_, host.color);
  assign_image(normals_, host.normals);
}

void DeviceDepthFrame::assign_from_pyramid(const DepthFrame& frame,
                                           const DeviceDepthImg& raw_depth,
                                           const DeviceVec3fImg& normals) {
  intrinsics_ = frame.intrinsics;
  world_to_camera_ = from_eigen(frame.world_to_camera);

  assign_image(depth_, raw_depth.view());

  if (frame.color != nullptr) {
    const auto source = frame.color->view();
    if (source.data == nullptr) {
      color_ = {};
    } else {
      color_.ensure_extent(source.width, source.height);
      color_.copy_from_staged(source, color_staging_);
    }
  } else {
    color_ = {};
  }

  assign_image(normals_, normals.view());
}

DeviceDepthFrameView DeviceDepthFrame::view() const {
  return DeviceDepthFrameView{.depth = depth_.view(),
                              .color = color_.view(),
                              .normals = normals_.view(),
                              .intrinsics = intrinsics_,
                              .world_to_camera = world_to_camera_};
}

template <VoxelGridView VolumeViewT>
void IntegrationSweep<MemorySpace::kDevice>::run(
    const VolumeViewT& volume, const DeviceIntegrationContext& context,
    const TsdfRuleVariant& rule) {
  constexpr dim3 kBlock{16, 16};
  const dim3 grid{cuda::ceil_div(volume.resolution().x, kBlock.x),
                  cuda::ceil_div(volume.resolution().y, kBlock.y)};

  std::visit(
      [&](const auto& concrete) {
        integrate_kernel<<<grid, kBlock>>>(volume, context, concrete);
      },
      rule);
  cuda::check(cudaGetLastError(), "integrate_kernel launch");
}

// One instantiation per registered storage combination.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KINECTFUSION_INSTANTIATE(GeomVoxel, Color)                      \
  template void IntegrationSweep<MemorySpace::kDevice>::run(            \
      const VolumeView<MemorySpace::kDevice, false, GeomVoxel, Color>&, \
      const DeviceIntegrationContext&, const TsdfRuleVariant&);
KINECTFUSION_FOR_EACH_REGISTERED_STORAGE(KINECTFUSION_INSTANTIATE)
#undef KINECTFUSION_INSTANTIATE

namespace {

// Enumerate set bits into the compact list: one iteration per marked block.
__global__ void compact_blocks_kernel(const std::uint32_t* bitmap,
                                      std::size_t word_count,
                                      std::uint32_t* list,
                                      unsigned int* count) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t first =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  for (std::size_t word = first; word < word_count; word += stride) {
    BlockBitmapOps::for_each_set_bit(
        bitmap[word], word * kBitmapWordBits, [&](std::size_t block) {
          list[atomicAdd(count, 1U)] = static_cast<std::uint32_t>(block);
        });
  }
}

}  // namespace

struct BandIntegrationSweep<MemorySpace::kDevice>::Scratch {
  cuda::DeviceBuffer<std::uint32_t> bitmap;
  cuda::DeviceBuffer<std::uint32_t> list;
  cuda::DeviceBuffer<unsigned int> count{1};
  std::size_t word_count{};
  BlockGrid blocks{};

  explicit Scratch(const Size3& resolution)
      : blocks(BlockGrid::for_resolution(resolution)) {
    word_count = BlockBitmapOps::word_count(blocks.count());
    bitmap = cuda::DeviceBuffer<std::uint32_t>{word_count};
    list = cuda::DeviceBuffer<std::uint32_t>{blocks.count()};
  }
};

BandIntegrationSweep<MemorySpace::kDevice>::BandIntegrationSweep() = default;
BandIntegrationSweep<MemorySpace::kDevice>::~BandIntegrationSweep() = default;
BandIntegrationSweep<MemorySpace::kDevice>::BandIntegrationSweep(
    BandIntegrationSweep&&) noexcept = default;
BandIntegrationSweep<MemorySpace::kDevice>& BandIntegrationSweep<
    MemorySpace::kDevice>::operator=(BandIntegrationSweep&&) noexcept = default;

template <VoxelGridView VolumeViewT>
void BandIntegrationSweep<MemorySpace::kDevice>::run(
    const VolumeViewT& volume, const DeviceIntegrationContext& context,
    const TsdfRuleVariant& rule) {
  if (!scratch_) {
    scratch_ = std::make_unique<Scratch>(volume.resolution());
  }
  Scratch& scratch = *scratch_;
  cuda::check(cudaMemsetAsync(scratch.bitmap.data(), 0,
                              scratch.word_count * sizeof(std::uint32_t)),
              "band bitmap clear");
  cuda::check(cudaMemsetAsync(scratch.count.data(), 0, sizeof(unsigned int)),
              "band count clear");

  const auto& frame = context.frame();
  constexpr dim3 kBlock2d{16, 16};
  const dim3 mark_grid{cuda::ceil_div(frame.depth.width, kBlock2d.x),
                       cuda::ceil_div(frame.depth.height, kBlock2d.y)};
  band_walk::mark_band_blocks_kernel<<<mark_grid, kBlock2d>>>(
      context, inverse(frame.world_to_camera), volume.geometry, scratch.blocks,
      band_walk::WarpAggregatedSet{scratch.bitmap.data()});
  cuda::check(cudaGetLastError(), "mark_band_blocks_kernel launch");

  const unsigned int compact_grid =
      std::min(cuda::ceil_div(scratch.word_count, band_walk::kLinearBlock),
               band_walk::kMaxGrid);
  compact_blocks_kernel<<<compact_grid, band_walk::kLinearBlock>>>(
      scratch.bitmap.data(), scratch.word_count, scratch.list.data(),
      scratch.count.data());
  cuda::check(cudaGetLastError(), "compact_blocks_kernel launch");

  std::visit(
      [&](const auto& concrete) {
        band_walk::integrate_listed_blocks_kernel<<<band_walk::kMaxGrid,
                                                    band_walk::kBlockThreads>>>(
            volume, context, concrete, scratch.blocks, scratch.list.data(),
            scratch.count.data());
      },
      rule);
  cuda::check(cudaGetLastError(), "integrate_listed_blocks_kernel launch");
}

// One instantiation per registered storage combination.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KINECTFUSION_INSTANTIATE(GeomVoxel, Color)                      \
  template void BandIntegrationSweep<MemorySpace::kDevice>::run(        \
      const VolumeView<MemorySpace::kDevice, false, GeomVoxel, Color>&, \
      const DeviceIntegrationContext&, const TsdfRuleVariant&);
KINECTFUSION_FOR_EACH_REGISTERED_STORAGE(KINECTFUSION_INSTANTIATE)
#undef KINECTFUSION_INSTANTIATE

}  // namespace kinectfusion
