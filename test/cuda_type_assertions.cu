#include <concepts>
#include <cstddef>
#include <cstdint>
#include <kinectfusion/cuda/device_buffer.cuh>
#include <kinectfusion/depth_processing.cuh>
#include <kinectfusion/device_volume.cuh>
#include <kinectfusion/icp_correspondence.hpp>
#include <kinectfusion/image_proc/device_image.cuh>
#include <kinectfusion/raycasting.hpp>
#include <kinectfusion/tsdf_integration.hpp>
#include <kinectfusion/volume.hpp>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using kinectfusion::BasicVolume;
using kinectfusion::DepthProcessingLevel;
using kinectfusion::DepthProcessor;
using kinectfusion::DeviceVolume;
using kinectfusion::MemorySpace;
using kinectfusion::SpaceTraits;
using kinectfusion::VolumeGeometry;
using kinectfusion::Voxel;
using kinectfusion::cuda::DeviceBuffer;
using kinectfusion::image_proc::DepthImageFor;
using kinectfusion::image_proc::DeviceDepthImage;
using kinectfusion::image_proc::DeviceImageView;

static_assert(
    std::same_as<DeviceDepthImage, DepthImageFor<MemorySpace::kDevice>>);
static_assert(DeviceDepthImage::kMemorySpace == MemorySpace::kDevice);
static_assert(!std::copy_constructible<DeviceDepthImage>);
static_assert(std::movable<DeviceDepthImage>);
static_assert(std::same_as<decltype(std::declval<DeviceDepthImage&>().view()),
                           DeviceImageView<std::uint16_t>>);
static_assert(
    std::same_as<decltype(std::declval<const DeviceDepthImage&>().view()),
                 DeviceImageView<const std::uint16_t>>);
// Mutable views convert to read-only views, like std::span.
static_assert(std::convertible_to<DeviceImageView<std::uint16_t>,
                                  DeviceImageView<const std::uint16_t>>);

using DeviceLevel = DepthProcessingLevel<MemorySpace::kDevice>;
static_assert(DeviceLevel::depth_image_type::kMemorySpace ==
              MemorySpace::kDevice);

using DeviceProcessor = DepthProcessor<MemorySpace::kDevice>;
static_assert(std::same_as<typename DeviceProcessor::SurfacePyramid,
                           std::vector<DeviceLevel>>);

// The one RAII owner of device allocations: move-only, shared by the device
// image and (through SpaceTraits) the device volume.
static_assert(!std::copy_constructible<DeviceBuffer<Voxel>>);
static_assert(std::movable<DeviceBuffer<Voxel>>);
static_assert(
    std::same_as<typename SpaceTraits<MemorySpace::kDevice>::Buffer<Voxel>,
                 DeviceBuffer<Voxel>>);
static_assert(std::same_as<DeviceVolume, BasicVolume<MemorySpace::kDevice>>);
static_assert(std::constructible_from<DeviceVolume, const VolumeGeometry&>);
static_assert(!std::copy_constructible<DeviceVolume>);
static_assert(
    std::same_as<decltype(std::declval<DeviceVolume&>().view()),
                 kinectfusion::VolumeView<MemorySpace::kDevice, false>>);
static_assert(
    std::same_as<decltype(std::declval<const DeviceVolume&>().view()),
                 kinectfusion::VolumeView<MemorySpace::kDevice, true>>);

// The per-element TSDF/raycast layer is space-generic; kernels reuse it.
static_assert(kinectfusion::TsdfUpdateRule<kinectfusion::ClassicTsdf,
                                           MemorySpace::kDevice>);
static_assert(kinectfusion::TsdfUpdateRule<kinectfusion::AngleWeightedTsdf,
                                           MemorySpace::kDevice>);
static_assert(
    std::same_as<decltype(kinectfusion::DeviceDepthFrameView::depth),
                 kinectfusion::image_proc::DeviceImageView<const std::uint16_t>>);

// The ICP per-pixel layer is space-generic; its accumulator is the reduction
// payload the correspondence kernel hands back to the host solve.
static_assert(std::is_trivially_copyable_v<kinectfusion::IcpNormalEquations>);
static_assert(
    std::same_as<
        decltype(std::declval<
                     const kinectfusion::CorrespondenceSearch<
                         MemorySpace::kDevice>&>()
                     .match(std::size_t{}, std::size_t{})),
        std::optional<kinectfusion::IcpCorrespondence>>);
static_assert(std::constructible_from<
              kinectfusion::CorrespondenceSearch<MemorySpace::kDevice>,
              const kinectfusion::ConstDeviceVertexNormalMapsView&,
              const kinectfusion::ConstDeviceVertexNormalMapsView&,
              const kinectfusion::CameraIntrinsics&,
              const kinectfusion::IcpIterationTransforms&,
              const kinectfusion::CorrespondenceGates&>);

}  // namespace
