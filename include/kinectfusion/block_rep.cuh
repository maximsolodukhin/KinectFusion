#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_BLOCK_REP_CUH
#define KINECTFUSION_INCLUDE_KINECTFUSION_BLOCK_REP_CUH

#include <cstddef>
#include <cstdint>
#include <kinectfusion/block_rep.hpp>
#include <kinectfusion/cuda/device_buffer.cuh>
#include <kinectfusion/device_volume.cuh>
#include <optional>

namespace kinectfusion {

// The CUDA backend defines the methods and instantiates the class for each
// registered storage combination.
template <typename GeomVoxel, typename Color>
class BlockRep<MemorySpace::kDevice, GeomVoxel, Color> {
  static constexpr MemorySpace kSpace = MemorySpace::kDevice;

 public:
  using Sampler = SparseVolumeSampler<kSpace, GeomVoxel, Color>;
  using View = SparseVolumeView<kSpace, false, GeomVoxel, Color>;
  using ConstView = SparseVolumeView<kSpace, true, GeomVoxel, Color>;

  explicit BlockRep(const VolumeGeometry& geometry,
                    std::size_t block_capacity = 0);

  [[nodiscard]] Sampler sampler() const { return Sampler{view()}; }
  [[nodiscard]] const VolumeGeometry& geometry() const { return geometry_; }

  [[nodiscard]] ConstView view() const {
    return {.grid = grid_.data(),
            .pool = pool_.data(),
            .color_pool = color_pool_.data(),
            .geometry = geometry_,
            .blocks = blocks_,
            .capacity = static_cast<std::uint32_t>(capacity_)};
  }
  [[nodiscard]] View view() {
    return {.grid = grid_.data(),
            .pool = pool_.data(),
            .color_pool = color_pool_.data(),
            .geometry = geometry_,
            .blocks = blocks_,
            .capacity = static_cast<std::uint32_t>(capacity_)};
  }

  [[nodiscard]] std::size_t observed_voxel_count() const;

  void integrate(const DepthFrameView<kSpace>& frame,
                 const TsdfIntegrationOptions& options,
                 const TsdfRuleVariant& rule);

  [[nodiscard]] ConstHostVolumeView host_dense_view(
      std::optional<HostVolume>& staging) const;

 private:
  VolumeGeometry geometry_;
  BlockGrid blocks_;
  std::size_t capacity_;
  std::size_t word_count_;
  cuda::DeviceBuffer<std::uint32_t> grid_;
  cuda::DeviceBuffer<GeomVoxel> pool_;
  cuda::DeviceBuffer<typename Color::Voxel> color_pool_;
  cuda::DeviceBuffer<std::uint32_t> bitmap_;
  cuda::DeviceBuffer<std::uint32_t> work_list_;
  cuda::DeviceBuffer<unsigned int> work_count_{1};
  cuda::DeviceBuffer<std::uint32_t> new_list_;
  cuda::DeviceBuffer<unsigned int> new_count_{1};
  cuda::DeviceBuffer<unsigned int> allocated_{1};
};

using DeviceBlockRep = BlockRep<MemorySpace::kDevice>;

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_BLOCK_REP_CUH */
