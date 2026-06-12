#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_IMAGE_PROC_UTIL_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_IMAGE_PROC_UTIL_HPP

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace kinectfusion::image_proc {
inline std::vector<char> read_file(const std::string& filename) {
  std::ifstream file{filename, std::ios::binary};
  if (!file) {
    throw std::runtime_error("Failed to open " + filename);
  }
  return std::vector<char>{std::istreambuf_iterator<char>{file},
                           std::istreambuf_iterator<char>{}};
}
}  // namespace kinectfusion::image_proc

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_IMAGE_PROC_UTIL_HPP */
