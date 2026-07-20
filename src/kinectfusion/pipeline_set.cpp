#include <algorithm>
#include <cstddef>
#include <format>
#include <kinectfusion/comparison.hpp>
#include <kinectfusion/depth_processing.hpp>
#include <kinectfusion/pipeline.hpp>
#include <kinectfusion/pipeline_set.hpp>
#include <kinectfusion/raycasting.hpp>
#include <kinectfusion/tsdf_integration.hpp>
#include <kinectfusion/validation.hpp>
#include <kinectfusion/vector.hpp>
#include <kinectfusion/volume.hpp>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace kinectfusion {

PipelineSet::PipelineSet(std::vector<Member> members,
                         std::size_t reference_index,
                         int compare_every_n_frames,
                         std::unique_ptr<DepthUploader> uploader)
    : members_(std::move(members)),
      reference_index_(reference_index),
      compare_every_n_frames_(compare_every_n_frames),
      uploader_(std::move(uploader)) {}

PipelineSet PipelineSet::create(const PipelineSetConfig& config) {
  require(!config.pipelines.empty(),
          "Pipeline set requires at least one pipeline");

  std::unordered_set<std::string_view> names;
  for (const PipelineConfig& pipeline : config.pipelines) {
    require(names.insert(pipeline.name).second,
            "Pipeline set requires unique pipeline names");
    require(pipeline.volume == config.pipelines.front().volume,
            "Pipeline set requires matching volume geometries");
  }

  std::size_t reference_index = 0;
  if (!config.reference.empty()) {
    while (reference_index < config.pipelines.size() &&
           config.pipelines.at(reference_index).name != config.reference) {
      ++reference_index;
    }
    require(reference_index < config.pipelines.size(),
            "Pipeline set reference must name a configured pipeline");
  }

  std::vector<Member> members;
  members.reserve(config.pipelines.size());
  for (const PipelineConfig& pipeline : config.pipelines) {
    Pipeline::Creation creation = Pipeline::create(pipeline);

    // A multi-member set exists to measure its members against each other, so
    // a member that quietly ran somewhere other than requested would report a
    // difference it never measured. Only a lone pipeline may fall back.
    if (creation.space != pipeline.space && config.pipelines.size() > 1) {
      throw std::invalid_argument(std::format(
          "Pipeline '{}' cannot run as configured: {}. A set compares its "
          "members against each other, so it will not substitute one.",
          pipeline.name, creation.fallback_reason));
    }
    members.emplace_back(std::move(creation.pipeline),
                         std::move(creation.fallback_reason), creation.space);
  }

  const bool any_device =
      std::ranges::any_of(members, [](const Member& member) {
        return member.space == MemorySpace::kDevice;
      });
  return {std::move(members), reference_index, config.compare_every_n_frames,
          any_device ? DepthUploader::create() : nullptr};
}

void PipelineSet::integrate(const DepthFrame& frame,
                            const DeviceDepthFrame* shared_upload) {
  if (shared_upload == nullptr && uploader_ != nullptr) {
    shared_upload = &uploader_->upload(frame);
  }
  for (const Member& member : members_) {
    member.pipeline->integrate(frame, shared_upload);
  }
}

void PipelineSet::integrate_member(std::size_t index, const DepthFrame& frame,
                                   const DeviceDepthFrame* shared_upload) {
  if (shared_upload == nullptr && uploader_ != nullptr) {
    shared_upload = &uploader_->upload(frame);
  }
  members_.at(index).pipeline->integrate(frame, shared_upload);
}

void PipelineSet::track_member(std::size_t index, const RaycastCamera& camera,
                               const PyramidLevel& live,
                               TrackingSurfaceConsumer& consumer) {
  members_.at(index).pipeline->track(camera, live, consumer);
}

SurfaceMaps PipelineSet::raycast_member(std::size_t index,
                                        const RaycastCamera& camera) {
  return members_.at(index).pipeline->raycast(camera);
}

SurfaceMaps PipelineSet::raycast_reference(const RaycastCamera& camera) {
  return members_.at(reference_index_).pipeline->raycast(camera);
}

void PipelineSet::track(const RaycastCamera& camera, const PyramidLevel& live,
                        TrackingSurfaceConsumer& consumer) {
  members_.at(reference_index_).pipeline->track(camera, live, consumer);
}

std::vector<PipelineOutput> PipelineSet::raycast_all(
    const RaycastCamera& camera) {
  std::vector<PipelineOutput> outputs;
  outputs.reserve(members_.size());

  for (const Member& member : members_) {
    auto* pipeline = member.pipeline.get();
    outputs.emplace_back(pipeline->name(), pipeline->raycast(camera));
  }

  return outputs;
}

std::vector<PipelineComparison> PipelineSet::compare() {
  return compare_members(nullptr);
}

std::vector<PipelineComparison> PipelineSet::compare(
    const std::vector<PipelineOutput>& outputs) {
  require(outputs.size() == members_.size(),
          "Pipeline comparison requires one raycast output per pipeline");
  return compare_members(&outputs);
}

std::vector<PipelineComparison> PipelineSet::compare_members(
    const std::vector<PipelineOutput>* outputs) {
  auto* reference_pipeline = members_.at(reference_index_).pipeline.get();

  // Staging volumes are locals so device copies are freed after the compare.
  std::optional<HostVolume> reference_staging;
  std::optional<HostVolume> member_staging;
  const ConstHostVolumeView reference_view =
      reference_pipeline->host_view(reference_staging);

  std::vector<PipelineComparison> comparisons;
  comparisons.reserve(members_.size() - 1);

  for (std::size_t index = 0; index < members_.size(); ++index) {
    if (index == reference_index_) {
      continue;
    }

    const Member& member = members_.at(index);
    auto pipeline_view = member.pipeline->host_view(member_staging);
    auto pipeline_name = member.pipeline->name();
    auto volume_comparison = Comparator::compare(pipeline_view, reference_view);

    PipelineComparison comparison{.name = pipeline_name,
                                  .volume = volume_comparison,
                                  .surface = std::nullopt};
    if (outputs != nullptr) {
      const auto& compare_maps = outputs->at(index).maps;
      const auto& reference_maps = outputs->at(reference_index_).maps;

      comparison.surface = Comparator::compare(compare_maps, reference_maps);
    }
    comparisons.push_back(std::move(comparison));
  }
  return comparisons;
}

}  // namespace kinectfusion
