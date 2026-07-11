#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_CUDA_CHECK_CUH
#define KINECTFUSION_INCLUDE_KINECTFUSION_CUDA_CHECK_CUH

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace kinectfusion::cuda {

inline void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string{operation} +
                             " failed: " + cudaGetErrorString(status));
  }
}

}  // namespace kinectfusion::cuda

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_CUDA_CHECK_CUH */
