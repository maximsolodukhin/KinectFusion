#ifndef KINECTFUSION_SRC_APP_PIPELINE_CONFIG_HPP
#define KINECTFUSION_SRC_APP_PIPELINE_CONFIG_HPP

#include <filesystem>
#include <kinectfusion/pipeline.hpp>
#include <kinectfusion/pipeline_set.hpp>
#include <kinectfusion/tsdf_integration.hpp>
#include <kinectfusion/vector.hpp>
#include <string_view>

namespace app {

// Name <-> value mappings shared by the CLI and the TOML pipeline config.
// Throws std::invalid_argument on unknown names.
[[nodiscard]] kinectfusion::TsdfRuleVariant tsdf_rule_from_name(
    std::string_view name);
[[nodiscard]] std::string_view tsdf_rule_name(
    const kinectfusion::TsdfRuleVariant& rule);
[[nodiscard]] kinectfusion::MemorySpace memory_space_from_name(
    std::string_view name);
[[nodiscard]] std::string_view memory_space_name(
    kinectfusion::MemorySpace space);

// File format: README "Ablation pipelines". Throws std::invalid_argument on
// malformed input or unknown keys.
[[nodiscard]] kinectfusion::PipelineSetConfig parse_pipeline_set(
    const std::filesystem::path& path,
    const kinectfusion::PipelineConfig& defaults, int default_compare_every);

}  // namespace app

#endif /* KINECTFUSION_SRC_APP_PIPELINE_CONFIG_HPP */
