#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_PIPELINE_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_PIPELINE_HPP

#include <cstddef>
#include <kinectfusion/raycasting.hpp>
#include <kinectfusion/tsdf_integration.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <memory>
#include <optional>
#include <string>

namespace kinectfusion {

// Everything needed to compose one reconstruction pipeline
struct PipelineConfig {
  std::string name;
  MemorySpace space{MemorySpace::kHost};
  TsdfRuleVariant tsdf_rule{AngleWeightedTsdf{}};
  TsdfIntegrationOptions integration{};
  RaycastOptions raycast{};
  VolumeGeometry volume{};
};

class Pipeline {
 public:
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;
  Pipeline(Pipeline&&) = delete;
  Pipeline& operator=(Pipeline&&) = delete;
  virtual ~Pipeline() = default;

  // Fuses one tracked depth frame into the pipeline's volume
  virtual void integrate(const DepthFrame& frame) = 0;

  [[nodiscard]] virtual SurfaceMaps raycast(const RaycastCamera& camera) = 0;

  [[nodiscard]] virtual std::size_t observed_voxel_count() const = 0;

  // If on GPU - copied, if not - viewed directly.
  [[nodiscard]] virtual ConstHostVolumeView host_view(
      std::optional<HostVolume>& staging) const = 0;

  [[nodiscard]] const std::string& name() const { return name_; }

  struct Creation {
    std::unique_ptr<Pipeline> pipeline;
    // Non-empty when the requested space was unavailable and the pipeline
    // fell back to host execution.
    std::string fallback_reason;
  };

  // Warn-and-fallback factory: an unavailable memory space reports a reason
  // instead of throwing; misconfiguration throws std::invalid_argument.
  [[nodiscard]] static Creation create(const PipelineConfig& config);

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
