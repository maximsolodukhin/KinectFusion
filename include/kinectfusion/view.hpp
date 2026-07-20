#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_VIEW_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_VIEW_HPP

#include <bit>
#include <kinectfusion/cuda_compat.hpp>
#include <type_traits>

namespace kinectfusion {

template <bool IsConst, typename T>
using Pointee = std::conditional_t<IsConst, const T, T>;

// Const and mutable views have the same layout.
struct ViewCast {
  template <typename ConstView, typename View>
    requires(std::is_trivially_copyable_v<View> &&
             std::is_trivially_copyable_v<ConstView> &&
             sizeof(View) == sizeof(ConstView))
  [[nodiscard]] KINECTFUSION_HOST_DEVICE static ConstView to_const(
      const View& view) {
    return std::bit_cast<ConstView>(view);
  }
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_VIEW_HPP */
