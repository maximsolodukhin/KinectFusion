#include "pipeline_config.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <kinectfusion/pipeline.hpp>
#include <kinectfusion/pipeline_set.hpp>
#include <kinectfusion/tsdf_integration.hpp>
#include <kinectfusion/validation.hpp>
#include <kinectfusion/vector.hpp>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
// Provides the whole toml:: namespace; include-cleaner cannot map it.
#include <toml++/toml.hpp>  // NOLINT(misc-include-cleaner)
#include <type_traits>
#include <variant>

// toml++ is a single header; include-cleaner cannot map toml:: symbols to it.
// NOLINTBEGIN(misc-include-cleaner)
namespace app {
namespace {

using kinectfusion::require;

[[noreturn]] void fail(const std::string& message) {
  throw std::invalid_argument(message);
}

[[nodiscard]] std::string_view key_of(const toml::key& key) {
  return key.str();
}

template <typename T>
[[nodiscard]] T value_of(const toml::node& node, std::string_view key) {
  const auto value = node.value<T>();
  if (!value) {
    fail(std::string{"Pipeline config key '"} + std::string{key} +
         "' has the wrong type");
  }
  return *value;
}

[[nodiscard]] float float_of(const toml::node& node, std::string_view key) {
  return static_cast<float>(value_of<double>(node, key));
}

// One [[pipeline]] table: `defaults` plus explicit overrides. Geometry is
// deliberately not overridable (see header).
[[nodiscard]] kinectfusion::PipelineConfig parse_pipeline(
    const toml::table& table, const kinectfusion::PipelineConfig& defaults) {
  kinectfusion::PipelineConfig config = defaults;
  config.name.clear();
  for (const auto& [key, node] : table) {
    const std::string_view name = key_of(key);
    if (name == "name") {
      config.name = value_of<std::string>(node, name);
    } else if (name == "space") {
      config.space = memory_space_from_name(value_of<std::string>(node, name));
    } else if (name == "tsdf-variant") {
      config.tsdf_rule = tsdf_rule_from_name(value_of<std::string>(node, name));
    } else if (name == "projective-distance") {
      config.integration.projective_distance = value_of<bool>(node, name);
    } else if (name == "distance-scaled-truncation") {
      config.integration.distance_scaled_truncation =
          value_of<bool>(node, name);
    } else if (name == "truncation-distance-scale") {
      config.integration.truncation_distance_scale = float_of(node, name);
    } else if (name == "observation-weight") {
      config.integration.observation_weight = float_of(node, name);
    } else if (name == "max-weight") {
      config.integration.max_weight = float_of(node, name);
    } else {
      fail(std::string{"Unknown pipeline config key '"} + std::string{name} +
           "'");
    }
  }
  require(!config.name.empty(), "Every [[pipeline]] requires a name");
  // Names label CSV columns and output subdirectories.
  require(!config.name.contains(','), "Pipeline names must not contain commas");
  return config;
}

}  // namespace

kinectfusion::TsdfRuleVariant tsdf_rule_from_name(std::string_view name) {
  if (name == "classic") {
    return kinectfusion::ClassicTsdf{};
  }
  if (name == "angle-weighted") {
    return kinectfusion::AngleWeightedTsdf{};
  }
  fail(std::string{"Unknown TSDF variant '"} + std::string{name} +
       "' (expected 'classic' or 'angle-weighted')");
}

std::string_view tsdf_rule_name(const kinectfusion::TsdfRuleVariant& rule) {
  return std::visit(
      [](const auto& alternative) -> std::string_view {
        using Rule = std::decay_t<decltype(alternative)>;
        if constexpr (std::is_same_v<Rule, kinectfusion::ClassicTsdf>) {
          return "classic";
        } else {
          return "angle-weighted";
        }
      },
      rule);
}

const std::map<std::string, kinectfusion::MemorySpace, std::less<>>&
memory_space_names() {
  static const std::map<std::string, kinectfusion::MemorySpace, std::less<>>
      kNames{{"cpu", kinectfusion::MemorySpace::kHost},
             {"cuda", kinectfusion::MemorySpace::kDevice}};
  return kNames;
}

kinectfusion::MemorySpace memory_space_from_name(std::string_view name) {
  const auto& names = memory_space_names();
  const auto found = names.find(name);
  if (found != names.end()) {
    return found->second;
  }
  fail(std::string{"Unknown memory space '"} + std::string{name} +
       "' (expected 'cpu' or 'cuda')");
}

std::string_view memory_space_name(kinectfusion::MemorySpace space) {
  for (const auto& [name, value] : memory_space_names()) {
    if (value == space) {
      return name;
    }
  }
  fail("Memory space has no registered name");
}

kinectfusion::PipelineSetConfig parse_pipeline_set(
    const std::filesystem::path& path,
    const kinectfusion::PipelineConfig& defaults, int default_compare_every) {
  toml::table table;
  try {
    table = toml::parse_file(path.string());
  } catch (const toml::parse_error& error) {
    fail(std::string{"Failed to parse pipeline config "} + path.string() +
         ": " + std::string{error.description()});
  }

  kinectfusion::PipelineSetConfig config{
      .pipelines = {},
      .reference = {},
      .compare_every_n_frames = default_compare_every};
  for (const auto& [key, node] : table) {
    const std::string_view name = key_of(key);
    if (name == "reference") {
      config.reference = value_of<std::string>(node, name);
    } else if (name == "compare-every-n-frames") {
      config.compare_every_n_frames =
          static_cast<int>(value_of<std::int64_t>(node, name));
    } else if (name == "pipeline") {
      const toml::array* pipelines = node.as_array();
      if (pipelines == nullptr) {
        fail("'pipeline' must be an array of tables ([[pipeline]])");
      }
      for (const toml::node& entry : *pipelines) {
        const toml::table* pipeline = entry.as_table();
        if (pipeline == nullptr) {
          fail("'pipeline' must be an array of tables ([[pipeline]])");
        }
        config.pipelines.push_back(parse_pipeline(*pipeline, defaults));
      }
    } else {
      fail(std::string{"Unknown pipeline config key '"} + std::string{name} +
           "'");
    }
  }
  require(!config.pipelines.empty(),
          "Pipeline config requires at least one [[pipeline]]");
  return config;
}

}  // namespace app
// NOLINTEND(misc-include-cleaner)
