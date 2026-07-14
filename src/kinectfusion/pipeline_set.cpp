#include <cstddef>
#include <kinectfusion/comparison.hpp>
#include <kinectfusion/pipeline.hpp>
#include <kinectfusion/pipeline_set.hpp>
#include <kinectfusion/validation.hpp>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace kinectfusion {

PipelineSet::PipelineSet(std::vector<Member> members,
                         std::size_t reference_index,
                         int compare_every_n_frames)
    : members_(std::move(members)),
      reference_index_(reference_index),
      compare_every_n_frames_(compare_every_n_frames) {}

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

    members.emplace_back(std::move(creation.pipeline),
                         std::move(creation.fallback_reason));
  }
  return {std::move(members), reference_index, config.compare_every_n_frames};
}

void PipelineSet::integrate(const DepthFrame& frame) {
  for (const Member& member : members_) {
    member.pipeline->integrate(frame);
  }
}

SurfaceMaps PipelineSet::raycast_reference(const RaycastCamera& camera) {
  return members_.at(reference_index_).pipeline->raycast(camera);
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

  const ConstHostVolumeView reference_view =
      reference_pipeline->host_view(reference_staging_);

  std::vector<PipelineComparison> comparisons;
  comparisons.reserve(members_.size() - 1);

  for (std::size_t index = 0; index < members_.size(); ++index) {
    if (index == reference_index_) {
      continue;
    }

    const Member& member = members_.at(index);
    auto pipeline_view = member.pipeline->host_view(member_staging_);
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
