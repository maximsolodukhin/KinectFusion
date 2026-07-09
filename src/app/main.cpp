#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <utility>

#include "app_options.hpp"
#include "logging.hpp"
#include "reconstruction.hpp"

namespace {

void configure_logging() {
#ifdef KINECTFUSION_ENABLE_LOGGING
  auto logger = spdlog::stdout_color_mt("kinectfusion");
  logger->set_pattern("[%H:%M:%S] [%^%l%$] %v");
  spdlog::set_default_logger(logger);
  spdlog::set_level(spdlog::level::info);
  spdlog::flush_on(spdlog::level::info);
#else
  auto logger = spdlog::stderr_color_mt("kinectfusion");
  logger->set_pattern("[%^%l%$] %v");
  spdlog::set_default_logger(logger);
  spdlog::set_level(spdlog::level::err);
  spdlog::flush_on(spdlog::level::err);
#endif
}

}  // namespace

int main(int argc, const char** argv) {
  try {
    configure_logging();

    app::AppOptions app_options;

    CLI::App app{"CPU KinectFusion reconstruction"};
    app::configure_cli(app, app_options);

    CLI11_PARSE(app, argc, argv);

    app::validate_options(app_options);

    app::log_info(
        "Starting KinectFusion: dataset={} max_frames={} volume={}^3 "
        "voxel_size={} truncation_distance={}",
        app_options.dataset_dir.string(), app_options.max_frames,
        app_options.volume_resolution, app_options.voxel_size,
        app_options.truncation_distance);

    app::Reconstruction reconstruction{std::move(app_options)};
    return reconstruction.run();
  } catch (const std::exception& e) {
    try {
      spdlog::error("Unhandled exception in main: {}", e.what());
    } catch (...) {
      std::cerr << "Unhandled exception in main: " << e.what() << '\n';
      return EXIT_FAILURE;
    }
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
