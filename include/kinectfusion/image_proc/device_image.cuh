#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_IMAGE_PROC_DEVICE_IMAGE_CUH
#define KINECTFUSION_INCLUDE_KINECTFUSION_IMAGE_PROC_DEVICE_IMAGE_CUH

#include <cstddef>
#include <kinectfusion/cuda/device_buffer.cuh>
#include <kinectfusion/image_proc/image.hpp>
#include <limits>
#include <stdexcept>
#include <utility>

namespace kinectfusion::image_proc {

// Device specialization of Image: 2D pixel semantics over a
// cuda::DeviceBuffer, which owns the allocation and the transfers. It is
// move-only because copying device storage must be explicit about direction
// and synchronization. Pixels are zero-initialized on construction, matching
// the host specialization.
template <typename PixelT>
class Image<PixelT, MemorySpace::kDevice> {
 public:
  using value_type = PixelT;

  static constexpr MemorySpace kMemorySpace = MemorySpace::kDevice;

  Image() = default;

  Image(std::size_t width, std::size_t height)
      : width_(width),
        height_(height),
        buffer_(checked_pixel_count(width, height)) {}

  ~Image() = default;

  Image(const Image&) = delete;
  Image& operator=(const Image&) = delete;

  Image(Image&& other) noexcept { swap(other); }

  Image& operator=(Image&& other) noexcept {
    if (this != &other) {
      Image temporary{std::move(other)};
      swap(temporary);
    }
    return *this;
  }

  [[nodiscard]] DeviceImageView<PixelT> view() noexcept {
    return {.data = buffer_.data(), .width = width_, .height = height_};
  }

  [[nodiscard]] DeviceImageView<const PixelT> view() const noexcept {
    return {.data = buffer_.data(), .width = width_, .height = height_};
  }

  [[nodiscard]] std::size_t width() const noexcept { return width_; }
  [[nodiscard]] std::size_t height() const noexcept { return height_; }
  [[nodiscard]] bool empty() const noexcept { return buffer_.empty(); }

  void copy_from(HostImageView<const PixelT> source) {
    require_same_dimensions(source.width, source.height);
    buffer_.copy_from_host(source.data, buffer_.size());
  }

  void copy_from(DeviceImageView<const PixelT> source) {
    require_same_dimensions(source.width, source.height);
    buffer_.copy_from_device(source.data, buffer_.size());
  }

  void copy_to(HostImageView<PixelT> destination) const {
    require_same_dimensions(destination.width, destination.height);
    buffer_.copy_to_host(destination.data, buffer_.size());
  }

  void fill_zero() { buffer_.fill_zero(); }

  void swap(Image& other) noexcept {
    using std::swap;
    swap(width_, other.width_);
    swap(height_, other.height_);
    buffer_.swap(other.buffer_);
  }

 private:
  [[nodiscard]] static std::size_t checked_pixel_count(std::size_t width,
                                                       std::size_t height) {
    if (height != 0U &&
        width > std::numeric_limits<std::size_t>::max() / height) {
      throw std::overflow_error("Device image dimensions overflow");
    }
    return width * height;
  }

  void require_same_dimensions(std::size_t width, std::size_t height) const {
    if (width != width_ || height != height_) {
      throw std::invalid_argument("Device image dimensions do not match");
    }
  }

  std::size_t width_{};
  std::size_t height_{};
  cuda::DeviceBuffer<PixelT> buffer_;
};

template <typename PixelT>
void swap(Image<PixelT, MemorySpace::kDevice>& lhs,
          Image<PixelT, MemorySpace::kDevice>& rhs) noexcept {
  lhs.swap(rhs);
}

using DeviceDepthImage = DepthImageFor<MemorySpace::kDevice>;
using DeviceColorImage = ColorImageFor<MemorySpace::kDevice>;
using DeviceVector3fImage = Vector3fImageFor<MemorySpace::kDevice>;

}  // namespace kinectfusion::image_proc

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_IMAGE_PROC_DEVICE_IMAGE_CUH */
