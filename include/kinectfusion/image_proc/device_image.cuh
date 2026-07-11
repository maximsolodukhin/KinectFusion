#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_IMAGE_PROC_DEVICE_IMAGE_CUH
#define KINECTFUSION_INCLUDE_KINECTFUSION_IMAGE_PROC_DEVICE_IMAGE_CUH

#include <cuda_runtime_api.h>

#include <cstddef>
#include <kinectfusion/cuda/check.cuh>
#include <kinectfusion/image_proc/image.hpp>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace kinectfusion::image_proc {

// Device specialization of Image. It is move-only because copying device
// storage must be explicit about direction and synchronization. Pixels are
// zero-initialized on construction, matching the host specialization.
template <typename PixelT>
class Image<PixelT, MemorySpace::kDevice> {
  static_assert(std::is_trivially_copyable_v<PixelT>,
                "Device image pixels must be trivially copyable");

 public:
  using value_type = PixelT;

  static constexpr MemorySpace kMemorySpace = MemorySpace::kDevice;

  Image() = default;

  Image(std::size_t width, std::size_t height)
      : width_(width), height_(height) {
    if (height_ != 0U &&
        width_ > std::numeric_limits<std::size_t>::max() / height_) {
      throw std::overflow_error("Device image dimensions overflow");
    }
    const std::size_t pixel_count = width_ * height_;
    if (pixel_count >
        std::numeric_limits<std::size_t>::max() / sizeof(PixelT)) {
      throw std::overflow_error("Device image allocation size overflows");
    }
    if (pixel_count != 0U) {
      cuda::check(cudaMalloc(reinterpret_cast<void**>(&data_),
                             pixel_count * sizeof(PixelT)),
                  "cudaMalloc(DeviceImage)");
      try {
        fill_zero();
      } catch (...) {
        // The destructor does not run when a constructor throws.
        release();
        throw;
      }
    }
  }

  ~Image() { release(); }

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
    return {.data = data_, .width = width_, .height = height_};
  }

  [[nodiscard]] DeviceImageView<const PixelT> view() const noexcept {
    return {.data = data_, .width = width_, .height = height_};
  }

  [[nodiscard]] std::size_t width() const noexcept { return width_; }
  [[nodiscard]] std::size_t height() const noexcept { return height_; }
  [[nodiscard]] bool empty() const noexcept { return data_ == nullptr; }

  void copy_from(HostImageView<const PixelT> source) {
    require_same_dimensions(source.width, source.height);
    if (!empty()) {
      cuda::check(
          cudaMemcpy(data_, source.data, size_bytes(), cudaMemcpyHostToDevice),
          "cudaMemcpy(DeviceImage host to device)");
    }
  }

  void copy_from(DeviceImageView<const PixelT> source) {
    require_same_dimensions(source.width, source.height);
    if (!empty()) {
      cuda::check(cudaMemcpy(data_, source.data, size_bytes(),
                             cudaMemcpyDeviceToDevice),
                  "cudaMemcpy(DeviceImage device to device)");
    }
  }

  void copy_to(HostImageView<PixelT> destination) const {
    require_same_dimensions(destination.width, destination.height);
    if (!empty()) {
      cuda::check(cudaMemcpy(destination.data, data_, size_bytes(),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(DeviceImage device to host)");
    }
  }

  void fill_zero() {
    if (!empty()) {
      cuda::check(cudaMemset(data_, 0, size_bytes()),
                  "cudaMemset(DeviceImage)");
    }
  }

  void swap(Image& other) noexcept {
    using std::swap;
    swap(data_, other.data_);
    swap(width_, other.width_);
    swap(height_, other.height_);
  }

 private:
  [[nodiscard]] std::size_t size_bytes() const noexcept {
    return width_ * height_ * sizeof(PixelT);
  }

  void require_same_dimensions(std::size_t width, std::size_t height) const {
    if (width != width_ || height != height_) {
      throw std::invalid_argument("Device image dimensions do not match");
    }
  }

  void release() noexcept {
    if (data_ != nullptr) {
      // Destructors cannot report CUDA teardown errors safely.
      static_cast<void>(cudaFree(data_));
    }
    data_ = nullptr;
    width_ = 0U;
    height_ = 0U;
  }

  PixelT* data_{};
  std::size_t width_{};
  std::size_t height_{};
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
