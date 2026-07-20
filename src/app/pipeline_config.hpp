#ifndef KINECTFUSION_SRC_APP_PIPELINE_CONFIG_HPP
#define KINECTFUSION_SRC_APP_PIPELINE_CONFIG_HPP

#include <filesystem>
#include <kinectfusion/pipeline.hpp>
#include <kinectfusion/pipeline_set.hpp>
#include <kinectfusion/tsdf_integration.hpp>
#include <kinectfusion/vector.hpp>
#include <map>
#include <string>
#include <string_view>

namespace app {

// The single name registry for memory spaces, shared by the TOML parser and
// the CLI transformer.
[[nodiscard]] const std::map<std::string, kinectfusion::MemorySpace,
                             std::less<>>&
memory_space_names();

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
[[nodiscard]] kinectfusion::VoxelStore voxel_store_from_name(
    std::string_view name);
[[nodiscard]] kinectfusion::ColorStore color_store_from_name(
    std::string_view name);
[[nodiscard]] kinectfusion::RaycastBackend raycast_backend_from_name(
    std::string_view name);
[[nodiscard]] kinectfusion::IntegrationMode integration_mode_from_name(
    std::string_view name);
[[nodiscard]] kinectfusion::StorageLayout storage_layout_from_name(
    std::string_view name);

// File format: README "Ablation pipelines". Throws std::invalid_argument on
// malformed input or unknown keys.
[[nodiscard]] kinectfusion::PipelineSetConfig parse_pipeline_set(
    const std::filesystem::path& path,
    const kinectfusion::PipelineConfig& defaults, int default_compare_every);

}  // namespace app

#endif /* KINECTFUSION_SRC_APP_PIPELINE_CONFIG_HPP */
