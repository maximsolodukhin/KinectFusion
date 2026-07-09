#ifndef KINECTFUSION_SRC_APP_LOGGING_HPP
#define KINECTFUSION_SRC_APP_LOGGING_HPP

#include <spdlog/spdlog.h>

#include <format>
#include <kinectfusion/icp_optimizer.hpp>
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
