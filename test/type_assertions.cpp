#include <concepts>
#include <cstdint>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/rgbd.hpp>
#include <kinectfusion/vector.hpp>
#include <utility>
#include <vector>

namespace {

using kinectfusion::DepthProcessingLevel;
using kinectfusion::DepthProcessor;
using kinectfusion::MemorySpace;
using kinectfusion::image_proc::DepthImage;
using kinectfusion::image_proc::DepthImageFor;
using kinectfusion::image_proc::HostImageView;

static_assert(std::same_as<DepthImage, DepthImageFor<MemorySpace::kHost>>);
static_assert(DepthImage::kMemorySpace == MemorySpace::kHost);
static_assert(std::copy_constructible<DepthImage>);
static_assert(std::movable<DepthImage>);
static_assert(std::same_as<decltype(std::declval<DepthImage&>().view()),
                           HostImageView<std::uint16_t>>);
static_assert(std::same_as<decltype(std::declval<const DepthImage&>().view()),
                           HostImageView<const std::uint16_t>>);

using HostLevel = DepthProcessingLevel<MemorySpace::kHost>;
static_assert(HostLevel::depth_image_type::kMemorySpace == MemorySpace::kHost);

using HostProcessor = DepthProcessor<MemorySpace::kHost>;
static_assert(
    std::same_as<
        decltype(std::declval<const HostProcessor&>().build_surface_pyramid(
            std::declval<const DepthImage&>(),
            std::declval<const kinectfusion::CameraIntrinsics&>())),
        std::vector<HostLevel>>);

}  // namespace
