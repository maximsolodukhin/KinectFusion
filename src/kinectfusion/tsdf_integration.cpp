#include <cstddef>
#include <kinectfusion/grid.hpp>
#include <kinectfusion/tsdf_integration.hpp>
#include <kinectfusion/validation.hpp>
#include <kinectfusion/volume.hpp>
#include <variant>

namespace kinectfusion {
namespace {

// The CUDA port lifts this sweep into the per-rule kernel launch.
class IntegrationSweep {
 public:
  IntegrationSweep(const HostVolumeView& volume,
                   const IntegrationContext<MemorySpace::kHost>& context)
      : volume_(volume), context_(&context) {}

  template <TsdfUpdateRule<MemorySpace::kHost> Rule>
  void run(const Rule& rule) const {
    for (const auto [x, y, z] : GridIndices{volume_.resolution()}) {
      rule.update(volume_, *context_, x, y, z);
    }
  }

 private:
  HostVolumeView volume_;
  const IntegrationContext<MemorySpace::kHost>* context_;
};

}  // namespace

TsdfIntegrator::TsdfIntegrator(TsdfRuleVariant rule,
                               TsdfIntegrationOptions options)
    : options_(validated(options)), rule_(rule) {}

void TsdfIntegrator::integrate(const HostVolumeView& volume,
                               const DepthFrame& frame) const {
  validate_frame(frame);
  const IntegrationContext<MemorySpace::kHost> context{frame.view(), options_};
  const IntegrationSweep sweep{volume, context};
  std::visit([&](const auto& rule) { sweep.run(rule); }, rule_);
}

TsdfIntegrationOptions TsdfIntegrator::validated(
    TsdfIntegrationOptions options) {
  require(options.depth_scale > 0.0F, "Depth scale must be positive");
  require(options.observation_weight > 0.0F,
          "Observation weight must be positive");
  require(options.max_weight > 0.0F, "Max weight must be positive");
  require(options.min_depth >= 0.0F && options.max_depth > options.min_depth,
          "Depth range is invalid");
  require(options.truncation_distance_scale >= 0.0F,
          "Truncation distance scale must be non-negative");
  return options;
}

void TsdfIntegrator::validate_frame(const DepthFrame& frame) {
  require(frame.depth != nullptr, "Depth frame requires a depth image");
  require(frame.depth->width() > 0U && frame.depth->height() > 0U,
          "Depth frame depth image must be non-empty");
  require(frame.intrinsics.fx > 0.0F && frame.intrinsics.fy > 0.0F,
          "Depth frame intrinsics must have positive focal lengths");
  require(frame.world_to_camera.allFinite(),
          "Depth frame world_to_camera must be finite");
}

}  // namespace kinectfusion
