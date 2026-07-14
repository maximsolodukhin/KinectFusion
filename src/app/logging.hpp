#ifndef KINECTFUSION_SRC_APP_LOGGING_HPP
#define KINECTFUSION_SRC_APP_LOGGING_HPP

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstddef>
#include <format>
#include <iterator>
#include <kinectfusion/comparison.hpp>
#include <kinectfusion/icp_optimizer.hpp>
#include <kinectfusion/pipeline_set.hpp>
#include <string_view>

#ifdef KINECTFUSION_ENABLE_LOGGING
#include <utility>
#endif

// Formatters so ICP tracking results log as single arguments instead of
// being enumerated field-by-field at every call site
template <>
struct std::formatter<kinectfusion::IcpFailure>
    : std::formatter<std::string_view> {
  auto format(kinectfusion::IcpFailure failure,
              std::format_context& ctx) const {
    return std::formatter<std::string_view>::format(name(failure), ctx);
  }

 private:
  [[nodiscard]] static constexpr std::string_view name(
      kinectfusion::IcpFailure failure) {
    switch (failure) {
      case kinectfusion::IcpFailure::kInvalidInput:
        return "kInvalidInput";
      case kinectfusion::IcpFailure::kTooFewCorrespondences:
        return "kTooFewCorrespondences";
      case kinectfusion::IcpFailure::kUnconstrainedSystem:
        return "kUnconstrainedSystem";
      case kinectfusion::IcpFailure::kSolveFailed:
        return "kSolveFailed";
      case kinectfusion::IcpFailure::kUpdateTooLarge:
        return "kUpdateTooLarge";
    }
    return "unknown";
  }
};

template <>
struct std::formatter<kinectfusion::IcpDiagnostics>
    : std::formatter<std::string_view> {
  static auto format(const kinectfusion::IcpDiagnostics& diagnostics,
                     std::format_context& ctx) {
    return std::format_to(
        ctx.out(),
        "correspondences={} mean_distance={} min_eigenvalue={} condition={} "
        "update_translation={} update_rotation={}",
        diagnostics.correspondences, diagnostics.mean_point_distance,
        diagnostics.min_system_eigenvalue, diagnostics.condition_number,
        diagnostics.update_translation, diagnostics.update_rotation);
  }
};

namespace app {

// {:csv} selects bare comma-separated values instead of labeled fields.
struct CsvSpecFormatter {
  bool csv{false};

  constexpr auto parse(std::format_parse_context& ctx) {
    const std::string_view remainder{ctx.begin(), ctx.end()};
    const auto length = std::min(remainder.find('}'), remainder.size());
    const std::string_view spec = remainder.substr(0, length);
    if (!spec.empty()) {
      if (spec != "csv") {
        throw std::format_error("expected ':csv'");
      }
      csv = true;
    }
    return std::next(ctx.begin(), static_cast<std::ptrdiff_t>(length));
  }
};

}  // namespace app

template <>
struct std::formatter<kinectfusion::VolumeComparison>
    : app::CsvSpecFormatter {
  auto format(const kinectfusion::VolumeComparison& comparison,
              std::format_context& ctx) const {
    return std::vformat_to(
        ctx.out(),
        csv ? "{},{},{},{},{},{}"
            : "compared_voxels={} only_pipeline={} only_reference={} "
              "max_distance_delta={} mean_distance_delta={} "
              "max_weight_delta={}",
        std::make_format_args(
            comparison.compared_voxels, comparison.only_primary,
            comparison.only_reference, comparison.max_distance_delta,
            comparison.mean_distance_delta, comparison.max_weight_delta));
  }
};

template <>
struct std::formatter<kinectfusion::SurfaceMapsComparison>
    : app::CsvSpecFormatter {
  auto format(const kinectfusion::SurfaceMapsComparison& comparison,
              std::format_context& ctx) const {
    return std::vformat_to(
        ctx.out(),
        csv ? "{},{},{},{},{},{},{}"
            : "compared_pixels={} only_pipeline={} only_reference={} "
              "max_point_distance={} mean_point_distance={} "
              "max_normal_angle={} mean_normal_angle={}",
        std::make_format_args(
            comparison.compared_pixels, comparison.only_primary,
            comparison.only_reference, comparison.max_point_distance,
            comparison.mean_point_distance, comparison.max_normal_angle,
            comparison.mean_normal_angle));
  }
};

template <>
struct std::formatter<kinectfusion::PipelineComparison>
    : app::CsvSpecFormatter {
  auto format(const kinectfusion::PipelineComparison& comparison,
              std::format_context& ctx) const {
    if (csv) {
      auto out = std::format_to(ctx.out(), "{},{:csv},", comparison.name,
                                comparison.volume);
      if (comparison.surface) {
        return std::format_to(out, "{:csv}", *comparison.surface);
      }
      return std::format_to(out, ",,,,,,");
    }
    auto out = std::format_to(ctx.out(), "pipeline '{}' vs reference: "
                                         "volume[{}]",
                              comparison.name, comparison.volume);
    if (comparison.surface) {
      return std::format_to(out, " surface[{}]", *comparison.surface);
    }
    return out;
  }
};

namespace app {

// Messages are rendered with std::format
template <typename... Args>
void log_info([[maybe_unused]] std::format_string<Args...> fmt,
              [[maybe_unused]] Args&&... args) {
#ifdef KINECTFUSION_ENABLE_LOGGING
  spdlog::info("{}", std::format(fmt, std::forward<Args>(args)...));
#endif
}

template <typename... Args>
void log_warn([[maybe_unused]] std::format_string<Args...> fmt,
              [[maybe_unused]] Args&&... args) {
#ifdef KINECTFUSION_ENABLE_LOGGING
  spdlog::warn("{}", std::format(fmt, std::forward<Args>(args)...));
#endif
}

}  // namespace app

#endif /* KINECTFUSION_SRC_APP_LOGGING_HPP */
