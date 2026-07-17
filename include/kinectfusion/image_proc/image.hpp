#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_IMAGE_PROC_IMAGE_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_IMAGE_PROC_IMAGE_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <kinectfusion/vector.hpp>
#include <type_traits>
#include <vector>

namespace kinectfusion::image_proc {

template <typename PixelT, MemorySpace Space = MemorySpace::kHost>
struct ImageView {
  PixelT* data{};
  std::size_t width{};
  std::size_t height{};

  static constexpr MemorySpace kMemorySpace = Space;

  // ImageView deliberately holds a raw pointer so it can be passed to CUDA
  // kernels. Subscripting a raw pointer here is unavoidable,
  // hence the NOLINTs on the two lookup expressions below.
  [[nodiscard]] KINECTFUSION_HOST_DEVICE PixelT& at(std::size_t x,
                                                    std::size_t y) {
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return data[(y * width) + x];
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  }

  [[nodiscard]] KINECTFUSION_HOST_DEVICE const PixelT& at(std::size_t x,
                                                          std::size_t y) const {
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return data[(y * width) + x];
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  }

  // Mutable views convert to read-only views implicitly, like std::span
  // (and like VolumeView). A member template so the const instantiation
  // never declares a self-conversion (nvcc warns on those).
  template <typename T = PixelT>
    requires(std::is_same_v<T, PixelT> && !std::is_const_v<T>)
  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  [[nodiscard]] KINECTFUSION_HOST_DEVICE operator ImageView<const T, Space>()
      const {
    return {.data = data, .width = width, .height = height};
  }
};

template <typename PixelT>
using HostImageView = ImageView<PixelT, MemorySpace::kHost>;

template <typename PixelT>
using DeviceImageView = ImageView<PixelT, MemorySpace::kDevice>;

// Readable as a view of PixelT pixels in either memory space; mutable views
// qualify through their read-only conversion.
template <typename V, typename PixelT>
concept ImageViewOf = std::convertible_to<V, HostImageView<const PixelT>> ||
                      std::convertible_to<V, DeviceImageView<const PixelT>>;

template <typename PixelT, MemorySpace Space = MemorySpace::kHost>
class Image;

// Host images retain value semantics and contiguous std::vector storage.
// The device specialization is defined separately in device_image.cuh so
// CPU-only translation units do not depend on the CUDA runtime headers.
template <typename PixelT>
class Image<PixelT, MemorySpace::kHost> {
 public:
  using value_type = PixelT;

  static constexpr MemorySpace kMemorySpace = MemorySpace::kHost;

  Image() = default;
  Image(std::size_t width, std::size_t height)
      : width_(width), height_(height) {
    data_.resize(width * height);
  }
  Image(std::size_t width, std::size_t height, const PixelT& init)
      : width_(width), height_(height), data_(width * height, init) {}

  [[nodiscard]] const PixelT& at(std::size_t x, std::size_t y) const {
    return data_[(y * width_) + x];
  }
  [[nodiscard]] PixelT& at(std::size_t x, std::size_t y) {
    return data_[(y * width_) + x];
  }

  [[nodiscard]] std::size_t width() const { return width_; }
  [[nodiscard]] std::size_t height() const { return height_; }

  [[nodiscard]] HostImageView<PixelT> view() {
    return HostImageView<PixelT>{
        .data = data_.data(), .width = width_, .height = height_};
  }

  [[nodiscard]] HostImageView<const PixelT> view() const {
    return HostImageView<const PixelT>{
        .data = data_.data(), .width = width_, .height = height_};
  }

  [[nodiscard]] std::vector<PixelT>& data() { return data_; }
  [[nodiscard]] const std::vector<PixelT>& data() const { return data_; }

 private:
  std::size_t width_{};
  std::size_t height_{};
  std::vector<PixelT> data_;
};

template <MemorySpace Space>
using DepthImageFor = Image<std::uint16_t, Space>;

template <MemorySpace Space>
using ColorImageFor = Image<std::uint32_t, Space>;

template <MemorySpace Space>
using Vector3fImageFor = Image<kinectfusion::Vec3f, Space>;

using DepthImage = DepthImageFor<MemorySpace::kHost>;
using ColorImage = ColorImageFor<MemorySpace::kHost>;
using Vector3fImage = Vector3fImageFor<MemorySpace::kHost>;

}  // namespace kinectfusion::image_proc

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_IMAGE_PROC_IMAGE_HPP */
