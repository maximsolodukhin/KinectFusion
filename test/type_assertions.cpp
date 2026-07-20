#include <concepts>
#include <cstdint>
#include <kinectfusion/block_rep.hpp>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/empty_space_skip.hpp>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/occupancy.hpp>
#include <kinectfusion/raycasting.hpp>
#include <kinectfusion/rgbd.hpp>
#include <kinectfusion/trilinear.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <kinectfusion/volume_representation.hpp>
#include <kinectfusion/volume_sampler.hpp>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using kinectfusion::DepthProcessingLevel;
using kinectfusion::DepthProcessor;
using kinectfusion::MemorySpace;
using kinectfusion::image_proc::DepthImage;
using kinectfusion::image_proc::DepthImageFor;
using kinectfusion::image_proc::HostImageView;

// The representation contract: DenseRep models it, and the sampler of each
// registered storage combination satisfies the raycast contract.
using kinectfusion::DenseRep;
using kinectfusion::FloatColorFacet;
using kinectfusion::NoColorFacet;
using kinectfusion::TsdfVolumeSampler;
using kinectfusion::VolumeRepresentation;
using kinectfusion::VolumeSampler;
using QuantizedVoxel = kinectfusion::BasicVoxel<kinectfusion::QuantizedTsdf,
                                                kinectfusion::QuantizedWeight>;

static_assert(
    VolumeRepresentation<DenseRep<MemorySpace::kHost>, MemorySpace::kHost>);
static_assert(
    TsdfVolumeSampler<VolumeSampler<MemorySpace::kHost, QuantizedVoxel>>);
static_assert(
    TsdfVolumeSampler<
        VolumeSampler<MemorySpace::kHost, kinectfusion::Voxel, NoColorFacet>>);
static_assert(FloatColorFacet::kEnabled && !NoColorFacet::kEnabled);
static_assert(TsdfVolumeSampler<
              VolumeSampler<MemorySpace::kHost, kinectfusion::Bf16Voxel>>);
static_assert(std::is_trivially_copyable_v<kinectfusion::OccupancyView>);
static_assert(std::is_trivially_copyable_v<kinectfusion::BitmapSkip>);

// The view/voxel/skip contracts: dense views model the flat refinement,
// sparse views only the coordinate one.
using kinectfusion::BlockBitmapView;
using kinectfusion::DenseVoxelGridView;
using kinectfusion::SkipPolicy;
using kinectfusion::SparseVolumeView;
using kinectfusion::TsdfVoxel;
using kinectfusion::VoxelGridView;

static_assert(TsdfVoxel<kinectfusion::Voxel>);
static_assert(TsdfVoxel<QuantizedVoxel>);
static_assert(TsdfVoxel<kinectfusion::Bf16Voxel>);
static_assert(!TsdfVoxel<kinectfusion::ColorVoxel>);

static_assert(DenseVoxelGridView<kinectfusion::HostVolumeView>);
static_assert(DenseVoxelGridView<kinectfusion::ConstHostVolumeView>);
static_assert(VoxelGridView<SparseVolumeView<>>);
static_assert(!DenseVoxelGridView<SparseVolumeView<>>);

static_assert(BlockBitmapView<kinectfusion::OccupancyView>);
static_assert(BlockBitmapView<kinectfusion::BandView>);

static_assert(SkipPolicy<kinectfusion::NoSkip>);
static_assert(SkipPolicy<kinectfusion::BitmapSkip>);
static_assert(SkipPolicy<kinectfusion::BandSkip>);

static_assert(kinectfusion::PixelRenderer<kinectfusion::HostSurfaceRaycast,
                                          MemorySpace::kHost>);

static_assert(std::same_as<DepthImage, DepthImageFor<MemorySpace::kHost>>);
static_assert(DepthImage::kMemorySpace == MemorySpace::kHost);
static_assert(std::copy_constructible<DepthImage>);
static_assert(std::movable<DepthImage>);
static_assert(std::same_as<decltype(std::declval<DepthImage&>().view()),
                           HostImageView<std::uint16_t>>);
static_assert(std::same_as<decltype(std::declval<const DepthImage&>().view()),
                           HostImageView<const std::uint16_t>>);
// Mutable views convert to read-only views, like std::span.
static_assert(std::convertible_to<HostImageView<std::uint16_t>,
                                  HostImageView<const std::uint16_t>>);
static_assert(!std::convertible_to<HostImageView<const std::uint16_t>,
                                   HostImageView<std::uint16_t>>);

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
