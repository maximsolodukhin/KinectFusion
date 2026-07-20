#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_PIPELINE_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_PIPELINE_HPP

#include <cstddef>
#include <cstdint>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/icp_correspondence.hpp>
#include <kinectfusion/marching_cubes.hpp>
#include <kinectfusion/raycasting.hpp>
#include <kinectfusion/tsdf_integration.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <memory>
#include <optional>
#include <string>
#include <variant>

namespace kinectfusion {

// Selects the registered storage combination of a pipeline. Pipeline::create
// maps the values to compile-time representations. RaycastBackend lives in
// raycasting.hpp.
enum class VoxelStore : std::uint8_t { kFloat, kQuantized, kBf16 };
enum class ColorStore : std::uint8_t { kFloat, kNone };
enum class StorageLayout : std::uint8_t { kDense, kSparse };

// Everything needed to compose one reconstruction pipeline
struct PipelineConfig {
  std::string name;
  MemorySpace space{MemorySpace::kHost};
  TsdfRuleVariant tsdf_rule{AngleWeightedTsdf{}};
  TsdfIntegrationOptions integration{};
  RaycastOptions raycast{};
  VolumeGeometry volume{};
  VoxelStore voxel{VoxelStore::kFloat};
  ColorStore color{ColorStore::kFloat};
  RaycastBackend raycast_backend{RaycastBackend::kMarch};
  StorageLayout storage{StorageLayout::kDense};
  std::size_t sparse_block_capacity{0};  // 0 = block_count / 4
};

using TrackingSurfacesVariant =
    std::variant<HostTrackingSurfaces, DeviceTrackingSurfaces>;

class Pipeline {
 public:
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;
  Pipeline(Pipeline&&) = delete;
  Pipeline& operator=(Pipeline&&) = delete;
  virtual ~Pipeline() = default;

  // Fuses one tracked depth frame into the pipeline's volume
  virtual void integrate(const DepthFrame& frame) = 0;

  // `shared_upload` is the set-wide device upload slot for this frame: the
  // first device pipeline publishes it, later ones reuse it, host pipelines
  // ignore it.
  virtual void integrate(const DepthFrame& frame,
                         const DeviceDepthFrame*& shared_upload) {
    static_cast<void>(shared_upload);
    integrate(frame);
  }

  [[nodiscard]] virtual SurfaceMaps raycast(const RaycastCamera& camera) = 0;

  // Views stay valid until the next tracking_surfaces or raycast call.
  [[nodiscard]] virtual TrackingSurfacesVariant tracking_surfaces(
      const RaycastCamera& camera, const LiveViewsVariant& live) = 0;

  [[nodiscard]] virtual std::size_t observed_voxel_count() const = 0;

  // If on GPU - copied, if not - viewed directly.
  [[nodiscard]] virtual ConstHostVolumeView host_view(
      std::optional<HostVolume>& staging) const = 0;

  // Meshes the volume on the host. Sparse pipelines sweep only allocated
  // blocks, so large volumes mesh without a dense staging copy.
  [[nodiscard]] virtual MarchingCubes::Mesh extract_mesh(
      float min_weight) const = 0;

  [[nodiscard]] const std::string& name() const { return name_; }

  struct Creation {
    std::unique_ptr<Pipeline> pipeline;
    // Where the pipeline actually runs; below the requested space exactly
    // when fallback_reason is set.
    MemorySpace space{MemorySpace::kHost};
    std::string fallback_reason;
  };

  // Warn-and-fallback factory: an unavailable memory space reports a reason
  // instead of throwing; misconfiguration throws std::invalid_argument.
  [[nodiscard]] static Creation create(const PipelineConfig& config);

  // Maps the configured storage enums to the registered compile-time types:
  // pick.operator()<GeomVoxel, Color>() runs for the selected combination.
  template <typename Pick>
  [[nodiscard]] static auto visit_storage(const PipelineConfig& config,
                                          const Pick& pick) {
    const auto with_color = [&]<typename Color>() {
      switch (config.voxel) {
        case VoxelStore::kQuantized:
          return pick.template operator()<QuantizedVoxel, Color>();
        case VoxelStore::kBf16:
          return pick.template operator()<Bf16Voxel, Color>();
        case VoxelStore::kFloat:
        default:
          return pick.template operator()<Voxel, Color>();
      }
    };
    if (config.color == ColorStore::kNone) {
      return with_color.template operator()<NoColorFacet>();
    }
    return with_color.template operator()<FloatColorFacet>();
  }

  // Throws std::invalid_argument on unsupported storage combinations.
  static void require_valid_storage(const PipelineConfig& config);

 protected:
  explicit Pipeline(std::string name);

 private:
  // Both defined in pipeline.cpp for CPU-only builds and in the CUDA
  // backend TU when it is compiled in.
  [[nodiscard]] static bool device_available();
  [[nodiscard]] static std::unique_ptr<Pipeline> create_device(
      const PipelineConfig& config);

  std::string name_;
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_PIPELINE_HPP */
