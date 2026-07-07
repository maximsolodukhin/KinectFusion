#ifndef KINECTFUSION_SAMPLE_LIBRARY_HPP
#define KINECTFUSION_SAMPLE_LIBRARY_HPP

#include <kinectfusion/sample_library_export.hpp>

[[nodiscard]] SAMPLE_LIBRARY_EXPORT int factorial(int input) noexcept;

[[nodiscard]] constexpr int factorial_constexpr(int input) noexcept {
  if (input == 0) {
    return 1;
  }

  return input * factorial_constexpr(input - 1);
}

#endif
