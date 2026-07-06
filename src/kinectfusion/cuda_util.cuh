#ifndef KINECTFUSION_SRC_KINECTFUSION_CUDA_UTIL_CUH
#define KINECTFUSION_SRC_KINECTFUSION_CUDA_UTIL_CUH

// Plumbing shared by the CUDA backends: error checking, device float3 math,
// the kernel-side camera pose, and reusable device/pinned host buffers.
//
// This header pulls in Eigen; translation units whose device code never runs
// Eigen should define EIGEN_NO_CUDA before their first Eigen include.

#include <cuda_runtime.h>

#include <Eigen/Core>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace kinectfusion::cuda {

inline void check(cudaError_t error, const char* what) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string{what} + " failed: " +
                             cudaGetErrorString(error));
  }
}

// Top three rows of the camera pose; the fourth row is always (0, 0, 0, 1).
struct KernelPose {
  float row[3][4];
};

[[nodiscard]] inline KernelPose make_kernel_pose(const Eigen::Matrix4f& pose) {
  KernelPose result{};
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 4; ++col) {
      result.row[row][col] = pose(row, col);
    }
  }
  return result;
}

__device__ __forceinline__ float device_nan() {
  return __int_as_float(0x7FC00000);
}

__device__ __forceinline__ float3 nan3() {
  const float nan = device_nan();
  return float3{nan, nan, nan};
}

__device__ __forceinline__ bool is_finite(const float3& v) {
  return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
  return float3{a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ __forceinline__ float3 cross(const float3& a, const float3& b) {
  return float3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x};
}

__device__ __forceinline__ float norm(const float3& v) {
  return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __forceinline__ float3 transform(const KernelPose& pose,
                                            const float3& point) {
  return float3{pose.row[0][0] * point.x + pose.row[0][1] * point.y +
                    pose.row[0][2] * point.z + pose.row[0][3],
                pose.row[1][0] * point.x + pose.row[1][1] * point.y +
                    pose.row[1][2] * point.z + pose.row[1][3],
                pose.row[2][0] * point.x + pose.row[2][1] * point.y +
                    pose.row[2][2] * point.z + pose.row[2][3]};
}

// Owns the CUDA stream a workspace queues its work on.
class Stream {
 public:
  Stream() { check(cudaStreamCreate(&stream_), "cudaStreamCreate"); }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  ~Stream() {
    if (stream_ != nullptr) {
      (void)cudaStreamDestroy(stream_);
    }
  }

  [[nodiscard]] cudaStream_t get() const { return stream_; }

 private:
  cudaStream_t stream_{};
};

struct DeviceAllocator {
  static void* allocate(std::size_t bytes) {
    void* ptr = nullptr;
    check(cudaMalloc(&ptr, bytes), "cudaMalloc");
    return ptr;
  }
  static void free(void* ptr) noexcept { (void)cudaFree(ptr); }
};

// Page-locked host memory, required to keep cudaMemcpyAsync asynchronous.
struct PinnedAllocator {
  static void* allocate(std::size_t bytes) {
    void* ptr = nullptr;
    check(cudaMallocHost(&ptr, bytes), "cudaMallocHost");
    return ptr;
  }
  static void free(void* ptr) noexcept { (void)cudaFreeHost(ptr); }
};

// Grow-only allocation, so buffers reused across calls stop hitting
// cudaMalloc/cudaMallocHost once they reach their steady-state size.
template <typename Allocator>
class Buffer {
 public:
  Buffer() = default;
  Buffer(Buffer&& other) noexcept
      : ptr_{std::exchange(other.ptr_, nullptr)},
        capacity_{std::exchange(other.capacity_, 0)} {}
  Buffer& operator=(Buffer&&) = delete;
  ~Buffer() { Allocator::free(ptr_); }

  void reserve(std::size_t bytes) {
    if (bytes <= capacity_) {
      return;
    }
    Allocator::free(ptr_);
    ptr_ = nullptr;
    capacity_ = 0;
    ptr_ = Allocator::allocate(bytes);
    capacity_ = bytes;
  }

  template <typename T>
  [[nodiscard]] T* as() const {
    return static_cast<T*>(ptr_);
  }

 private:
  void* ptr_{nullptr};
  std::size_t capacity_{0};
};

using DeviceBuffer = Buffer<DeviceAllocator>;
using PinnedBuffer = Buffer<PinnedAllocator>;

}  // namespace kinectfusion::cuda

#endif  // KINECTFUSION_SRC_KINECTFUSION_CUDA_UTIL_CUH
