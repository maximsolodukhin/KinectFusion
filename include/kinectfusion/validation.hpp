#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_VALIDATION_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_VALIDATION_HPP

#include <stdexcept>

namespace kinectfusion {

// Shared precondition check for user-facing options structs.
inline void require(bool condition, const char* message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_VALIDATION_HPP */
