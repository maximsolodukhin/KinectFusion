#include <algorithm>
#include <array>
#include <cstddef>
#include <kinectfusion/cuda/check.cuh>
#include <kinectfusion/cuda/device_buffer.cuh>
#include <kinectfusion/cuda/launch.cuh>
#include <kinectfusion/cuda/pinned_buffer.cuh>
#include <kinectfusion/icp_correspondence.hpp>
#include <memory>
#include <type_traits>
#include <vector>

namespace kinectfusion {

namespace {

static_assert(sizeof(std::size_t) == sizeof(unsigned long long));

constexpr unsigned int kWarpSize = 32;
constexpr unsigned int kFullWarpMask = 0xFFFFFFFFU;
constexpr std::size_t kFloatSums = kIcpUpperTriangleSize + kIcpDof + 1;
constexpr unsigned int kReduceBlock = 256;
constexpr unsigned int kMaxReduceGrid = 1024;

// Sums every lane's float sums into lane 0.
// __shfl_down_sync - mask var delta width
// so basically during first iter:
// lane 0 = lane 0 + lane 16
// lane 1 = lane 1 + lane 17
// lane 2 = lane 2 + lane 18
// ...
// lane 15 = lane 15 + lane 31
// second iter:
// lane 0 = lane 0 + lane 8 ie lane 0 = lane 0 + lane 16 + lane 8 + lane 24
// lane 1 = lane 1 + lane 9 ie lane 1 = lane 1 + lane 17 + lane 9 + lane 25
// ...
__device__ void warp_reduce(IcpNormalEquations& local) {
  for (unsigned int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    for (std::size_t entry = 0; entry < kIcpUpperTriangleSize; ++entry) {
      local.jtj[entry] +=
          __shfl_down_sync(kFullWarpMask, local.jtj[entry], offset);
    }
    for (std::size_t entry = 0; entry < kIcpDof; ++entry) {
      local.jtr[entry] +=
          __shfl_down_sync(kFullWarpMask, local.jtr[entry], offset);
    }
    local.distance_sum +=
        __shfl_down_sync(kFullWarpMask, local.distance_sum, offset);
  }
}

// Grid-stride over the live image
// Each thread accumulates its matches locally
// warps reduce via shuffles(warp_reduce)
// warp leaders merge into shared memory, and one thread per block touches the
// global system. Every lane runs the same number of rounds so the ballot sees
// the full warp. The search comes in via device memory so the launch config is
// static and the whole iteration replays as one CUDA graph.
// ballot_sync evaluates condition for every lane in the warp and returns a
// bitmask where each bit corresponds to a lane. The number of set bits is the
// number of matches in the warp, which can be easily computed using popcount
// The warp leaders then atomically add the local sums to shared memory, and one
// thread per block atomically adds the shared memory sums to the global system.
// Is an improvement over previous atomic add per match, which caused 90% of the
// time to be spent in atomicAdd.
__global__ void reduce_correspondences_kernel(
    const DeviceCorrespondenceSearch* search_parameters,
    IcpNormalEquations* result, const DeviceIcpLoopResult* loop_state) {
  // After convergence or failure, the remaining iterations of the loop
  // graph collapse to this one broadcast load per thread.
  if (loop_state != nullptr && loop_state->status != 0) {
    return;
  }
  const DeviceCorrespondenceSearch& search = *search_parameters;
  __shared__ float block_sums[kFloatSums];
  __shared__ unsigned long long block_count;

  if (threadIdx.x == 0) {
    block_count = 0;
  }
  for (std::size_t entry = threadIdx.x; entry < kFloatSums;
       entry += blockDim.x) {
    block_sums[entry] = 0.0F;
  }
  __syncthreads();

  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t first =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;

  const std::size_t width = search.live().vertices.width;
  const std::size_t pixel_count = width * search.live().vertices.height;
  const std::size_t rounds = (pixel_count + stride - 1) / stride;

  IcpNormalEquations local;
  unsigned long long warp_count = 0;
  for (std::size_t round = 0; round < rounds; ++round) {
    const std::size_t index = first + (round * stride);
    compat::optional<IcpCorrespondence> match;
    if (index < pixel_count) {
      match = search.match(index % width, index / width);
    }

    warp_count += __popc(__ballot_sync(kFullWarpMask, match.has_value()));

    if (match) {
      local.accumulate(*match);
    }
  }

  warp_reduce(local);
  if (threadIdx.x % kWarpSize == 0 && warp_count != 0) {
    std::size_t slot = 0;

    for (std::size_t entry = 0; entry < kIcpUpperTriangleSize; ++entry) {
      atomicAdd(&block_sums[slot++], local.jtj[entry]);
    }

    for (std::size_t entry = 0; entry < kIcpDof; ++entry) {
      atomicAdd(&block_sums[slot++], local.jtr[entry]);
    }

    atomicAdd(&block_sums[slot], local.distance_sum);
    atomicAdd(&block_count, warp_count);
  }
  __syncthreads();

  if (threadIdx.x == 0 && block_count != 0) {
    std::size_t slot = 0;
    for (std::size_t entry = 0; entry < kIcpUpperTriangleSize; ++entry) {
      atomicAdd(&result->jtj[entry], block_sums[slot++]);
    }

    for (std::size_t entry = 0; entry < kIcpDof; ++entry) {
      atomicAdd(&result->jtr[entry], block_sums[slot++]);
    }

    atomicAdd(&result->distance_sum, block_sums[slot]);
    atomicAdd(reinterpret_cast<unsigned long long*>(&result->count),
              block_count);
  }
}

// One thread: build the 6x6 SPD system, solve it, and compose the Rodrigues
// increment onto the camera pose for the next reduce in the graph chain.
// Status latches: once set, later steps do nothing.
__global__ void solve_update_kernel(const IcpNormalEquations* equations,
                                    DeviceCorrespondenceSearch* search,
                                    DeviceIcpLoopResult* result,
                                    DeviceIcpLoopParams params) {
  if (result->status != 0) {
    return;
  }
  const IcpNormalEquations& eq = *equations;
  result->equations = eq;
  if (eq.count < params.min_correspondences) {
    result->status = 2;
    return;
  }

  float mat[kIcpDof][kIcpDof];
  std::size_t entry = 0;

  for (std::size_t row = 0; row < kIcpDof; ++row) {
    for (std::size_t col = row; col < kIcpDof; ++col) {
      mat[row][col] = eq.jtj[entry];
      mat[col][row] = eq.jtj[entry];
      ++entry;
    }
  }

  // In-place Cholesky (LLT); non-positive pivot = solve failure.
  float chol[kIcpDof][kIcpDof];
  for (std::size_t row = 0; row < kIcpDof; ++row) {
    for (std::size_t col = 0; col <= row; ++col) {
      float sum = mat[row][col];
      for (std::size_t k = 0; k < col; ++k) {
        sum -= chol[row][k] * chol[col][k];
      }
      if (row == col) {
        if (sum <= 0.0F) {
          result->status = 3;
          return;
        }
        chol[row][row] = sqrtf(sum);
      } else {
        chol[row][col] = sum / chol[col][col];
      }
    }
  }

  float solution[kIcpDof];

  for (std::size_t row = 0; row < kIcpDof; ++row) {  // forward: L y = b
    float sum = eq.jtr[row];
    for (std::size_t k = 0; k < row; ++k) {
      sum -= chol[row][k] * solution[k];
    }
    solution[row] = sum / chol[row][row];
  }

  for (std::size_t row = kIcpDof; row-- > 0;) {  // backward: L^T x = y
    float sum = solution[row];
    for (std::size_t k = row + 1; k < kIcpDof; ++k) {
      sum -= chol[k][row] * solution[k];
    }
    solution[row] = sum / chol[row][row];
  }

  const Vec3f axis_angle = make_vec3f(solution[0], solution[1], solution[2]);
  const Vec3f translation = make_vec3f(solution[3], solution[4], solution[5]);
  const float angle = norm(axis_angle);
  const float update_translation = norm(translation);

  result->update_translation = update_translation;
  result->update_rotation = angle;

  if (update_translation > params.max_update_translation ||
      angle > params.max_update_rotation) {
    result->status = 4;
    return;
  }

  // Rodrigues rotation from the angle-axis increment.
  Mat3f rot{.row_x = make_vec3f(1.0F, 0.0F, 0.0F),
            .row_y = make_vec3f(0.0F, 1.0F, 0.0F),
            .row_z = make_vec3f(0.0F, 0.0F, 1.0F)};
  if (angle > 1.0e-12F) {
    const Vec3f unit = axis_angle / angle;
    const float sin_a = sinf(angle);
    const float cos_a = cosf(angle);
    const float one_c = 1.0F - cos_a;
    rot.row_x = make_vec3f(cos_a + (unit.x * unit.x * one_c),
                           (unit.x * unit.y * one_c) - (unit.z * sin_a),
                           (unit.x * unit.z * one_c) + (unit.y * sin_a));
    rot.row_y = make_vec3f((unit.y * unit.x * one_c) + (unit.z * sin_a),
                           cos_a + (unit.y * unit.y * one_c),
                           (unit.y * unit.z * one_c) - (unit.x * sin_a));
    rot.row_z = make_vec3f((unit.z * unit.x * one_c) - (unit.y * sin_a),
                           (unit.z * unit.y * one_c) + (unit.x * sin_a),
                           cos_a + (unit.z * unit.z * one_c));
  }
  RigidTransform& camera = search->transforms_ref().camera;
  const Mat3f old = camera.rotation;
  Mat3f composed;

  composed.row_x = make_vec3f(
      dot(rot.row_x, make_vec3f(old.row_x.x, old.row_y.x, old.row_z.x)),
      dot(rot.row_x, make_vec3f(old.row_x.y, old.row_y.y, old.row_z.y)),
      dot(rot.row_x, make_vec3f(old.row_x.z, old.row_y.z, old.row_z.z)));
  composed.row_y = make_vec3f(
      dot(rot.row_y, make_vec3f(old.row_x.x, old.row_y.x, old.row_z.x)),
      dot(rot.row_y, make_vec3f(old.row_x.y, old.row_y.y, old.row_z.y)),
      dot(rot.row_y, make_vec3f(old.row_x.z, old.row_y.z, old.row_z.z)));
  composed.row_z = make_vec3f(
      dot(rot.row_z, make_vec3f(old.row_x.x, old.row_y.x, old.row_z.x)),
      dot(rot.row_z, make_vec3f(old.row_x.y, old.row_y.y, old.row_z.y)),
      dot(rot.row_z, make_vec3f(old.row_x.z, old.row_y.z, old.row_z.z)));

  camera.rotation = composed;
  camera.translation = (rot * camera.translation) + translation;
  result->transforms = search->transforms_ref();

  if (update_translation < params.min_update_translation &&
      angle < params.min_update_rotation) {
    result->status = 1;
  }
}

}  // namespace

// One iteration = one instantiated graph replay: upload the search
// parameters and zero the accumulator, reduce, download the system.
// Only the grid size differs between pyramid levels.
// The private stream is blocking, so the legacy-stream pyramid and
// raycast kernels order with the graph both ways.
template <GraphBuildStrategy Build>
struct BasicDeviceCorrespondenceSweep<Build>::Scratch {
  struct GraphEntry {
    unsigned int grid{};
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
  };

  struct LoopEntry {
    unsigned int grid{};
    unsigned int iterations{};
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
  };

  cuda::DeviceBuffer<IcpNormalEquations> accumulator{1};
  cuda::PinnedBuffer<IcpNormalEquations> staging{1};
  cuda::DeviceBuffer<DeviceCorrespondenceSearch> search_device{1};
  cuda::PinnedBuffer<DeviceCorrespondenceSearch> search_staging{1};
  cuda::DeviceBuffer<DeviceIcpLoopResult> loop_result_device{1};
  cuda::PinnedBuffer<DeviceIcpLoopResult> loop_result_staging{1};

  cudaStream_t stream{};
  std::vector<GraphEntry> graphs;
  std::vector<LoopEntry> loop_graphs;

  Scratch() {
    cuda::check(cudaStreamCreate(&stream), "cudaStreamCreate(ICP reduce)");
  }

  ~Scratch() {
    for (GraphEntry& entry : graphs) {
      cudaGraphExecDestroy(entry.executable);
      cudaGraphDestroy(entry.graph);
    }
    for (LoopEntry& entry : loop_graphs) {
      cudaGraphExecDestroy(entry.executable);
      cudaGraphDestroy(entry.graph);
    }
    cudaStreamDestroy(stream);
  }

  // The whole-level GN loop as one captured graph. Upload the search and a
  // zeroed result once, run memset + reduce + solve/update per iteration,
  // then download the result. One sync per level, not one per iteration.
  cudaGraphExec_t loop_executable_for(unsigned int grid,
                                      unsigned int iterations,
                                      const DeviceIcpLoopParams& params) {
    for (const LoopEntry& entry : loop_graphs) {
      if (entry.grid == grid && entry.iterations == iterations) {
        return entry.executable;
      }
    }
    LoopEntry entry{.grid = grid, .iterations = iterations};
    cuda::check(
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
        "cudaStreamBeginCapture(ICP loop)");
    cudaMemcpyAsync(search_device.data(), search_staging.data(),
                    sizeof(DeviceCorrespondenceSearch), cudaMemcpyHostToDevice,
                    stream);
    cudaMemsetAsync(loop_result_device.data(), 0, sizeof(DeviceIcpLoopResult),
                    stream);
    for (unsigned int iteration = 0; iteration < iterations; ++iteration) {
      cudaMemsetAsync(accumulator.data(), 0, sizeof(IcpNormalEquations),
                      stream);

      reduce_correspondences_kernel<<<grid, kReduceBlock, 0, stream>>>(
          search_device.data(), accumulator.data(), loop_result_device.data());
      solve_update_kernel<<<1, 1, 0, stream>>>(
          accumulator.data(), search_device.data(), loop_result_device.data(),
          params);
    }

    cudaMemcpyAsync(loop_result_staging.data(), loop_result_device.data(),
                    sizeof(DeviceIcpLoopResult), cudaMemcpyDeviceToHost,
                    stream);
    cuda::check(cudaStreamEndCapture(stream, &entry.graph),
                "cudaStreamEndCapture(ICP loop)");
    try {
      cuda::check(cudaGraphInstantiate(&entry.executable, entry.graph, 0),
                  "cudaGraphInstantiate(ICP loop)");
    } catch (...) {
      cudaGraphDestroy(entry.graph);
      throw;
    }
    loop_graphs.push_back(entry);
    return entry.executable;
  }

  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  Scratch(Scratch&&) = delete;
  Scratch& operator=(Scratch&&) = delete;

  cudaGraphExec_t executable_for(unsigned int grid) {
    for (const GraphEntry& entry : graphs) {
      if (entry.grid == grid) {
        return entry.executable;
      }
    }
    GraphEntry entry{.grid = grid};
    entry.graph = build_graph(grid);
    try {
      cuda::check(cudaGraphInstantiate(&entry.executable, entry.graph, 0),
                  "cudaGraphInstantiate(ICP)");
    } catch (...) {
      cudaGraphDestroy(entry.graph);
      throw;
    }
    graphs.push_back(entry);
    return entry.executable;
  }

  cudaGraph_t build_graph(unsigned int grid) {
    if constexpr (std::is_same_v<Build, CapturedGraphBuild>) {
      return capture_graph(grid);
    } else {
      return assemble_graph(grid);
    }
  }

  // Tried to ablate the explicit graph assembly and capturing(4.2 cuda
  // programming guide). Doesn't matter at all -_-.

  // Records the chain by capturing ordinary stream launches
  cudaGraph_t capture_graph(unsigned int grid) {
    cudaGraph_t graph{};
    cuda::check(
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
        "cudaStreamBeginCapture(ICP)");
    cudaMemsetAsync(accumulator.data(), 0, sizeof(IcpNormalEquations), stream);
    cudaMemcpyAsync(search_device.data(), search_staging.data(),
                    sizeof(DeviceCorrespondenceSearch), cudaMemcpyHostToDevice,
                    stream);

    reduce_correspondences_kernel<<<grid, kReduceBlock, 0, stream>>>(
        search_device.data(), accumulator.data(), nullptr);

    cudaMemcpyAsync(staging.data(), accumulator.data(),
                    sizeof(IcpNormalEquations), cudaMemcpyDeviceToHost, stream);

    cuda::check(cudaStreamEndCapture(stream, &graph),
                "cudaStreamEndCapture(ICP)");
    return graph;
  }

  // Spells out the same four nodes and their dependencies through the graph
  // API. memset and upload are independent roots the kernel joins.
  cudaGraph_t assemble_graph(unsigned int grid) {
    cudaGraph_t graph{};
    cuda::check(cudaGraphCreate(&graph, 0), "cudaGraphCreate(ICP)");
    try {
      static_assert(sizeof(IcpNormalEquations) % sizeof(unsigned int) == 0);

      cudaMemsetParams zero{};
      zero.dst = accumulator.data();
      zero.value = 0;
      zero.elementSize = sizeof(unsigned int);
      zero.width = sizeof(IcpNormalEquations) / sizeof(unsigned int);
      zero.height = 1;
      cudaGraphNode_t zero_node{};

      cuda::check(cudaGraphAddMemsetNode(&zero_node, graph, nullptr, 0, &zero),
                  "graph memset node(ICP)");

      cudaGraphNode_t upload_node{};
      cuda::check(cudaGraphAddMemcpyNode1D(
                      &upload_node, graph, nullptr, 0, search_device.data(),
                      search_staging.data(), sizeof(DeviceCorrespondenceSearch),
                      cudaMemcpyHostToDevice),
                  "graph upload node(ICP)");

      const DeviceCorrespondenceSearch* search_argument = search_device.data();
      IcpNormalEquations* result_argument = accumulator.data();
      const DeviceIcpLoopResult* loop_argument = nullptr;
      std::array<void*, 3> arguments{&search_argument, &result_argument,
                                     &loop_argument};
      cudaKernelNodeParams kernel{};
      kernel.func = reinterpret_cast<void*>(&reduce_correspondences_kernel);
      kernel.gridDim = dim3{grid, 1, 1};
      kernel.blockDim = dim3{kReduceBlock, 1, 1};
      kernel.sharedMemBytes = 0;
      kernel.kernelParams = arguments.data();
      kernel.extra = nullptr;
      const std::array<cudaGraphNode_t, 2> kernel_inputs{zero_node,
                                                         upload_node};
      cudaGraphNode_t kernel_node{};
      cuda::check(
          cudaGraphAddKernelNode(&kernel_node, graph, kernel_inputs.data(),
                                 kernel_inputs.size(), &kernel),
          "graph kernel node(ICP)");

      cudaGraphNode_t download_node{};
      cuda::check(cudaGraphAddMemcpyNode1D(
                      &download_node, graph, &kernel_node, 1, staging.data(),
                      accumulator.data(), sizeof(IcpNormalEquations),
                      cudaMemcpyDeviceToHost),
                  "graph download node(ICP)");
    } catch (...) {
      cudaGraphDestroy(graph);
      throw;
    }
    return graph;
  }
};

template <GraphBuildStrategy Build>
BasicDeviceCorrespondenceSweep<Build>::BasicDeviceCorrespondenceSweep() =
    default;
template <GraphBuildStrategy Build>
BasicDeviceCorrespondenceSweep<Build>::~BasicDeviceCorrespondenceSweep() =
    default;
template <GraphBuildStrategy Build>
BasicDeviceCorrespondenceSweep<Build>::BasicDeviceCorrespondenceSweep(
    BasicDeviceCorrespondenceSweep&&) noexcept = default;
template <GraphBuildStrategy Build>
BasicDeviceCorrespondenceSweep<Build>&
BasicDeviceCorrespondenceSweep<Build>::operator=(
    BasicDeviceCorrespondenceSweep&&) noexcept = default;

template <GraphBuildStrategy Build>
IcpNormalEquations BasicDeviceCorrespondenceSweep<Build>::run(
    const DeviceCorrespondenceSearch& search) {
  const std::size_t pixel_count =
      search.live().vertices.width * search.live().vertices.height;

  if (pixel_count == 0) {
    return {};
  }
  if (!scratch_) {
    scratch_ = std::make_unique<Scratch>();
  }

  const unsigned int grid =
      std::min(cuda::ceil_div(pixel_count, kReduceBlock), kMaxReduceGrid);
  *scratch_->search_staging.data() = search;

  cuda::check(cudaGraphLaunch(scratch_->executable_for(grid), scratch_->stream),
              "reduce_correspondences graph launch");

  // The sync waits for the download and reports kernel failures.
  cuda::check(cudaStreamSynchronize(scratch_->stream),
              "reduce_correspondences graph");

  return *scratch_->staging.data();
}

template <GraphBuildStrategy Build>
DeviceIcpLoopResult BasicDeviceCorrespondenceSweep<Build>::run_loop(
    const DeviceCorrespondenceSearch& search, unsigned int iterations,
    const DeviceIcpLoopParams& params) {
  const std::size_t pixel_count =
      search.live().vertices.width * search.live().vertices.height;
  if (pixel_count == 0 || iterations == 0) {
    return {};
  }
  if (!scratch_) {
    scratch_ = std::make_unique<Scratch>();
  }
  const unsigned int grid =
      std::min(cuda::ceil_div(pixel_count, kReduceBlock), kMaxReduceGrid);
  *scratch_->search_staging.data() = search;
  cuda::check(
      cudaGraphLaunch(scratch_->loop_executable_for(grid, iterations, params),
                      scratch_->stream),
      "icp loop graph launch");
  cuda::check(cudaStreamSynchronize(scratch_->stream), "icp loop graph");
  return *scratch_->loop_result_staging.data();
}

template class BasicDeviceCorrespondenceSweep<ExplicitGraphBuild>;
template class BasicDeviceCorrespondenceSweep<CapturedGraphBuild>;

}  // namespace kinectfusion
