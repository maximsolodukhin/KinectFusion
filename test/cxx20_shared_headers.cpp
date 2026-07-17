// Kernel-reachable headers must stay C++20-compatible: nvcc caps CUDA TUs at
// C++20 while host TUs build as C++23. This TU compiles every shared header
// at C++20 so leakage of newer std features fails locally, before the remote
// CUDA build.
// NOLINTBEGIN(misc-include-cleaner)
#include <kinectfusion/cuda_compat.hpp>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/grid.hpp>
#include <kinectfusion/icp_correspondence.hpp>
#include <kinectfusion/image_proc/image.hpp>
#include <kinectfusion/raycasting.hpp>
#include <kinectfusion/rgbd.hpp>
#include <kinectfusion/tsdf_integration.hpp>
#include <kinectfusion/validation.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <kinectfusion/volume_sampler.hpp>
// NOLINTEND(misc-include-cleaner)

static_assert(__cplusplus == 202002L);

namespace {

using kinectfusion::compat::clamp;
using kinectfusion::compat::max;
using kinectfusion::compat::min;

static_assert(min(2, 3) == 2);
static_assert(min(3.0F, 2.0F) == 2.0F);
static_assert(max(2, 3) == 3);
static_assert(max(3.0F, 2.0F) == 3.0F);
static_assert(clamp(5, 0, 3) == 3);
static_assert(clamp(-1.0F, 0.0F, 3.0F) == 0.0F);
static_assert(clamp(1U, 0U, 3U) == 1U);

}  // namespace
