#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_CUDA_PINNED_BUFFER_CUH
#define KINECTFUSION_INCLUDE_KINECTFUSION_CUDA_PINNED_BUFFER_CUH

#include <cuda_runtime_api.h>

#include <cstddef>
#include <kinectfusion/cuda/check.cuh>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace kinectfusion::cuda {

// Move-only RAII ownership of page-locked host memory.
// Fast memory transfers between host and device, for example for ICP.
template <typename T>
class PinnedBuffer {
  static_assert(std::is_trivially_copyable_v<T>,
                "Pinned buffers hold trivially copyable elements");

 public:
  using value_type = T;

  PinnedBuffer() = default;

  explicit PinnedBuffer(std::size_t count) : count_(count) {
    if (count_ > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::overflow_error("Pinned buffer allocation size overflows");
    }
    if (count_ != 0U) {
      check(cudaHostAlloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T),
                          cudaHostAllocDefault),
            "cudaHostAlloc(PinnedBuffer)");
    }
  }

  ~PinnedBuffer() { release(); }

  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  PinnedBuffer(PinnedBuffer&& other) noexcept { swap(other); }

  PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
    if (this != &other) {
      PinnedBuffer temporary{std::move(other)};
      swap(temporary);
    }
    return *this;
  }

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }
  [[nodiscard]] bool empty() const noexcept { return data_ == nullptr; }

  void swap(PinnedBuffer& other) noexcept {
    using std::swap;
    swap(data_, other.data_);
    swap(count_, other.count_);
  }

 private:
  void release() noexcept {
    if (data_ != nullptr) {
      cudaFreeHost(data_);
      data_ = nullptr;
    }
    count_ = 0;
  }

  T* data_{nullptr};
  std::size_t count_{0};
};

template <typename T>
void swap(PinnedBuffer<T>& lhs, PinnedBuffer<T>& rhs) noexcept {
  lhs.swap(rhs);
}

}  // namespace kinectfusion::cuda

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_CUDA_PINNED_BUFFER_CUH */
