#ifndef KINECTFUSION_SRC_APP_FRAME_OUTPUT_HPP
#define KINECTFUSION_SRC_APP_FRAME_OUTPUT_HPP

#include <kinectfusion/volume.hpp>

#include "app_options.hpp"

namespace app {

// Writes the enabled per-frame artifacts (raycast PNG and/or point-cloud PLY)
// for `frame_index` into app_options.output_dir. No-op if both outputs are
// disabled.
void write_outputs(const AppOptions& app_options,
                   const kinectfusion::SurfaceMaps& maps, int frame_index);

}  // namespace app

#endif /* KINECTFUSION_SRC_APP_FRAME_OUTPUT_HPP */
