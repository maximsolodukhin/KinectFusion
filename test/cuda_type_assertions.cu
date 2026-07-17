// Also the nvcc canary: every kernel-reachable header is included here
// explicitly so a header nvcc cannot compile fails this TU, not a later port.
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <kinectfusion/cuda/device_buffer.cuh>
#include <kinectfusion/cuda_compat.hpp>
#include <kinectfusion/depth_processing.cuh>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/device_volume.cuh>
#include <kinectfusion/grid.hpp>
#include <kinectfusion/icp_correspondence.hpp>
#include <kinectfusion/image_proc/device_image.cuh>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/raycasting.cuh>
#include <kinectfusion/raycasting.hpp>
#include <kinectfusion/rgbd.hpp>
#include <kinectfusion/tsdf_integration.cuh>
#include <kinectfusion/tsdf_integration.hpp>
#include <kinectfusion/validation.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <kinectfusion/volume_sampler.hpp>
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
using kinectfusion::image_proc::AnyDeviceImage;
using kinectfusion::image_proc::DepthImageFor;
using kinectfusion::image_proc::DeviceDepthImage;
using kinectfusion::image_proc::DeviceImageView;
using kinectfusion::image_proc::HostImageView;
using kinectfusion::image_proc::RefillableFrom;

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

// A device image refills only from a same-pixel view; either space, since
// frames arrive from a host upload or a device pyramid.
static_assert(AnyDeviceImage<DeviceDepthImage>);
static_assert(!AnyDeviceImage<DepthImageFor<MemorySpace::kHost>>);
static_assert(!AnyDeviceImage<int>);
static_assert(
    RefillableFrom<DeviceDepthImage, HostImageView<const std::uint16_t>>);
static_assert(
    RefillableFrom<DeviceDepthImage, DeviceImageView<const std::uint16_t>>);
static_assert(!RefillableFrom<DeviceDepthImage, DeviceImageView<const float>>);
// Mutable sources arrive through the span-like read-only conversion.
static_assert(RefillableFrom<DeviceDepthImage, DeviceImageView<std::uint16_t>>);

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
static_assert(std::same_as<
              decltype(kinectfusion::DeviceDepthFrameView::depth),
              kinectfusion::image_proc::DeviceImageView<const std::uint16_t>>);

// Kernel arguments are passed by value through the launch; everything the
// per-element layer closes over must be trivially copyable.
static_assert(std::is_trivially_copyable_v<
              kinectfusion::BilateralFilter<MemorySpace::kDevice>>);
static_assert(std::is_trivially_copyable_v<
              kinectfusion::BlockDownsample<MemorySpace::kDevice>>);
static_assert(std::is_trivially_copyable_v<
              kinectfusion::VertexProjection<MemorySpace::kDevice>>);
static_assert(std::is_trivially_copyable_v<
              kinectfusion::NormalEstimation<MemorySpace::kDevice>>);
static_assert(std::is_trivially_copyable_v<
              kinectfusion::DepthFrameView<MemorySpace::kDevice>>);
static_assert(std::is_trivially_copyable_v<
              kinectfusion::IntegrationContext<MemorySpace::kDevice>>);
static_assert(std::is_trivially_copyable_v<
              kinectfusion::VolumeView<MemorySpace::kDevice>>);
static_assert(std::is_trivially_copyable_v<
              kinectfusion::VolumeSampler<MemorySpace::kDevice>>);
static_assert(kinectfusion::TsdfVolumeSampler<
              kinectfusion::VolumeSampler<MemorySpace::kDevice>>);
static_assert(std::is_trivially_copyable_v<kinectfusion::DeviceSurfaceRaycast>);

// Device surface maps are a move-only owner like the device images they hold.
static_assert(!std::copy_constructible<kinectfusion::DeviceSurfaceMaps>);
static_assert(std::movable<kinectfusion::DeviceSurfaceMaps>);
static_assert(std::same_as<
              decltype(std::declval<kinectfusion::DeviceSurfaceMaps&>().view()),
              kinectfusion::DeviceSurfaceMapsView>);

// The ICP per-pixel layer is space-generic; its accumulator is the reduction
// payload the correspondence kernel hands back to the host solve.
static_assert(std::is_trivially_copyable_v<kinectfusion::IcpNormalEquations>);
static_assert(
    std::is_trivially_copyable_v<kinectfusion::DeviceCorrespondenceSearch>);
static_assert(
    std::same_as<
        decltype(std::declval<const kinectfusion::DeviceCorrespondenceSearch&>()
                     .match(std::size_t{}, std::size_t{})),
        kinectfusion::compat::optional<kinectfusion::IcpCorrespondence>>);
static_assert(
    std::constructible_from<kinectfusion::DeviceCorrespondenceSearch,
                            const kinectfusion::DeviceTrackingSurfaces&,
                            const kinectfusion::CameraIntrinsics&,
                            const kinectfusion::IcpIterationTransforms&,
                            const kinectfusion::CorrespondenceGates&>);

// Both graph-build variants of the sweep are default-constructible, move-only,
// and reduce to the same normal-equations type; they only differ in how the
// replayed graph is built.
static_assert(
    std::default_initializable<kinectfusion::DeviceCorrespondenceSweep>);
static_assert(
    std::default_initializable<kinectfusion::CapturedCorrespondenceSweep>);
static_assert(std::movable<kinectfusion::CapturedCorrespondenceSweep>);
static_assert(
    !std::copy_constructible<kinectfusion::CapturedCorrespondenceSweep>);
static_assert(
    std::same_as<
        decltype(std::declval<kinectfusion::CapturedCorrespondenceSweep&>().run(
            std::declval<const kinectfusion::DeviceCorrespondenceSearch&>())),
        kinectfusion::IcpNormalEquations>);

}  // namespace
