#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_IMAGE_PROC_READ_PNG_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_IMAGE_PROC_READ_PNG_HPP

#include <string>

#include "image.hpp"

namespace kinectfusion::image_proc {

template <class T>
T read_png(const std::string& filename);

extern template DepthImage read_png<DepthImage>(const std::string& filename);
extern template ColorImage read_png<ColorImage>(const std::string& filename);

}  // namespace kinectfusion::image_proc

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_IMAGE_PROC_READ_PNG_HPP */
