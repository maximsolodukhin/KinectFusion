#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_CUDA_DEVICE_BUFFER_CUH
#define KINECTFUSION_INCLUDE_KINECTFUSION_CUDA_DEVICE_BUFFER_CUH

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <kinectfusion/cuda/check.cuh>
#include <kinectfusion/cuda/launch.cuh>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace kinectfusion::cuda {

namespace detail {

// std::fill_n
template <typename T>
__global__ void fill_kernel(T* data, std::size_t count, T value) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t first =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  for (std::size_t index = first; index < count; index += stride) {
    data[index] = value;
  }
}

}  // namespace detail

// Move-only RAII ownership of one device allocation — the single place that
// calls cudaMalloc/cudaFree. Owners with richer semantics (device images,
// device volumes via SpaceTraits) compose a buffer and hand out non-owning
// views. Elements are zero-initialized on construction, matching the host
// containers; copies are direction-labelled and never implicit.
template <typename T>
class DeviceBuffer {
  static_assert(std::is_trivially_copyable_v<T>,
                "Device buffers hold trivially copyable elements");

 public:
  using value_type = T;

  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t count) : count_(count) {
    allocate();
    try {
      fill_zero();
    } catch (...) {
      // The destructor does not run when a constructor throws.
      release();
      throw;
    }
  }

  // Skips the zero initialization for buffers that are immediately
  // overwritten
  [[nodiscard]] static DeviceBuffer uninitialized(std::size_t count) {
    DeviceBuffer buffer;
    buffer.count_ = count;
    buffer.allocate();
    return buffer;
  }

  ~DeviceBuffer() { release(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      DeviceBuffer temporary{std::move(other)};
      swap(temporary);
    }
    return *this;
  }

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }
  [[nodiscard]] bool empty() const noexcept { return data_ == nullptr; }

  // Direction-labelled element copies; element counts must match exactly.
  void copy_from_host(const T* source, std::size_t count) {
    require_same_count(count);
    if (!empty()) {
      check(cudaMemcpy(data_, source, size_bytes(), cudaMemcpyHostToDevice),
            "cudaMemcpy(DeviceBuffer host to device)");
    }
  }

  void copy_from_device(const T* source, std::size_t count) {
    require_same_count(count);
    if (!empty()) {
      check(cudaMemcpy(data_, source, size_bytes(), cudaMemcpyDeviceToDevice),
            "cudaMemcpy(DeviceBuffer device to device)");
    }
  }

  void copy_to_host(T* destination, std::size_t count) const {
    require_same_count(count);
    if (!empty()) {
      check(
          cudaMemcpy(destination, data_, size_bytes(), cudaMemcpyDeviceToHost),
          "cudaMemcpy(DeviceBuffer device to host)");
    }
  }

  void fill_zero() {
    if (!empty()) {
      check(cudaMemset(data_, 0, size_bytes()), "cudaMemset(DeviceBuffer)");
    }
  }

  // Synchronous element-wise fill.
  void fill(const T& value) {
    if (empty()) {
      return;
    }
    constexpr unsigned int kBlock = 256;
    constexpr unsigned int kMaxGrid = 256;
    const unsigned int grid = std::min(ceil_div(count_, kBlock), kMaxGrid);
    detail::fill_kernel<<<grid, kBlock>>>(data_, count_, value);
    check(cudaGetLastError(), "fill_kernel launch");
    check(cudaDeviceSynchronize(), "fill_kernel");
  }

  void swap(DeviceBuffer& other) noexcept {
    using std::swap;
    swap(data_, other.data_);
    swap(count_, other.count_);
  }

 private:
  void allocate() {
    if (count_ > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::overflow_error("Device buffer allocation size overflows");
    }
    if (count_ != 0U) {
      check(cudaMalloc(reinterpret_cast<void**>(&data_), size_bytes()),
            "cudaMalloc(DeviceBuffer)");
    }
  }

  [[nodiscard]] std::size_t size_bytes() const noexcept {
    return count_ * sizeof(T);
  }

  void require_same_count(std::size_t count) const {
    if (count != count_) {
      throw std::invalid_argument("Device buffer element counts do not match");
    }
  }

  void release() noexcept {
    if (data_ != nullptr) {
      // Destructors cannot report CUDA teardown errors safely.
      static_cast<void>(cudaFree(data_));
    }
    data_ = nullptr;
    count_ = 0U;
  }

  T* data_{};
  std::size_t count_{};
};

template <typename T>
void swap(DeviceBuffer<T>& lhs, DeviceBuffer<T>& rhs) noexcept {
  lhs.swap(rhs);
}

}  // namespace kinectfusion::cuda

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_CUDA_DEVICE_BUFFER_CUH */
