#ifndef KINECTFUSION_SRC_APP_LOGGING_HPP
#define KINECTFUSION_SRC_APP_LOGGING_HPP

#include <spdlog/spdlog.h>

namespace app {
template <typename... Args>
void log_info([[maybe_unused]] spdlog::format_string_t<Args...> fmt,
              [[maybe_unused]] Args&&... args) {
#ifdef KINECTFUSION_ENABLE_LOGGING
  spdlog::info(fmt, std::forward<Args>(args)...);
#endif
}

template <typename... Args>
void log_warn([[maybe_unused]] spdlog::format_string_t<Args...> fmt,
              [[maybe_unused]] Args&&... args) {
#ifdef KINECTFUSION_ENABLE_LOGGING
  spdlog::warn(fmt, std::forward<Args>(args)...);
#endif
}

}  // namespace app

#endif /* KINECTFUSION_SRC_APP_LOGGING_HPP */
