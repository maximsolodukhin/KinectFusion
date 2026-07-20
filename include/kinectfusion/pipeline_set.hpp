#ifndef KINECTFUSION_INCLUDE_KINECTFUSION_PIPELINE_SET_HPP
#define KINECTFUSION_INCLUDE_KINECTFUSION_PIPELINE_SET_HPP

#include <cstddef>
#include <kinectfusion/comparison.hpp>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/pipeline.hpp>
#include <kinectfusion/raycasting.hpp>
#include <kinectfusion/tsdf_integration.hpp>
#include <kinectfusion/volume.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace kinectfusion {

struct PipelineSetConfig {
  std::vector<PipelineConfig> pipelines;
  std::string reference;          // empty selects the first pipeline
  int compare_every_n_frames{1};  // <= 0 disables comparison
};

struct PipelineComparison {
  std::string name;
  VolumeComparison volume;
  std::optional<SurfaceMapsComparison> surface;  // set by compare(outputs)
};

struct PipelineOutput {
  std::string name;
  SurfaceMaps maps;
};

class PipelineSet {
 public:
  struct Member {
    std::unique_ptr<Pipeline> pipeline;
    std::string fallback_reason;
    // The space the pipeline actually runs in, after any fallback.
    MemorySpace space{MemorySpace::kHost};
  };

  // Throws std::invalid_argument on an empty set, duplicate names, an
  // unknown reference, or mismatched volume geometries.
  [[nodiscard]] static PipelineSet create(const PipelineSetConfig& config);

  // On nullptr the set uploads the frame once for all device members.
  void integrate(const DepthFrame& frame,
                 const DeviceDepthFrame* shared_upload = nullptr);

  [[nodiscard]] SurfaceMaps raycast_reference(const RaycastCamera& camera);

  void track(const RaycastCamera& camera, const PyramidLevel& live,
             TrackingSurfaceConsumer& consumer);

  // kDevice only when every member runs on the device
  // Decides where shared input processing runs.
  [[nodiscard]] MemorySpace common_space() const {
    for (const Member& member : members_) {
      if (member.space != MemorySpace::kDevice) {
        return MemorySpace::kHost;
      }
    }
    return MemorySpace::kDevice;
  }

  // Ordered like members().
  [[nodiscard]] std::vector<PipelineOutput> raycast_all(
      const RaycastCamera& camera);

  [[nodiscard]] std::vector<PipelineComparison> compare();

  // `outputs` must come from raycast_all.
  [[nodiscard]] std::vector<PipelineComparison> compare(
      const std::vector<PipelineOutput>& outputs);

  [[nodiscard]] bool should_compare(int frame_index) const {
    return members_.size() > 1 && compare_every_n_frames_ > 0 &&
           frame_index % compare_every_n_frames_ == 0;
  }

  [[nodiscard]] std::size_t size() const { return members_.size(); }
  [[nodiscard]] const std::vector<Member>& members() const { return members_; }
  [[nodiscard]] const Pipeline& reference() const {
    return *members_.at(reference_index_).pipeline;
  }

 private:
  PipelineSet(std::vector<Member> members, std::size_t reference_index,
              int compare_every_n_frames,
              std::unique_ptr<DepthUploader> uploader);

  [[nodiscard]] std::vector<PipelineComparison> compare_members(
      const std::vector<PipelineOutput>* outputs);

  std::vector<Member> members_;
  std::size_t reference_index_{};
  int compare_every_n_frames_{};
  // Null unless a member runs on the device.
  std::unique_ptr<DepthUploader> uploader_;
};

}  // namespace kinectfusion

#endif /* KINECTFUSION_INCLUDE_KINECTFUSION_PIPELINE_SET_HPP */
