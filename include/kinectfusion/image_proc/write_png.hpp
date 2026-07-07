#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_IMAGE_PROC_WRITE_PNG_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_IMAGE_PROC_WRITE_PNG_HPP

#include <kinectfusion/image_proc/image.hpp>
#include <string>

namespace kinectfusion::image_proc {

void write_png(const ColorImage& image, const std::string& filename);

}  // namespace kinectfusion::image_proc

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_IMAGE_PROC_WRITE_PNG_HPP */
