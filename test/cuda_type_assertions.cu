#include <concepts>
#include <cstdint>
#include <kinectfusion/depth_processing.cuh>
#include <kinectfusion/image_proc/device_image.cuh>
#include <utility>
#include <vector>

namespace {

using kinectfusion::DepthProcessingLevel;
using kinectfusion::DepthProcessor;
using kinectfusion::MemorySpace;
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

using DeviceLevel = DepthProcessingLevel<MemorySpace::kDevice>;
static_assert(DeviceLevel::depth_image_type::kMemorySpace ==
              MemorySpace::kDevice);

using DeviceProcessor = DepthProcessor<MemorySpace::kDevice>;
static_assert(std::same_as<typename DeviceProcessor::SurfacePyramid,
                           std::vector<DeviceLevel>>);

}  // namespace
