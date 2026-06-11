#include <cstdlib>
#include <exception>
#include <string>

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

#include <kinectfusion/sample_library.hpp>

int main(int argc, const char **argv)
{
  try {
    CLI::App app{"KinectFusion"};
    app.set_version_flag("--version", std::string{KINECTFUSION_VERSION});

    CLI11_PARSE(app, argc, argv);

    spdlog::info("Hello, world! factorial(5) = {}", factorial(5));
  } catch (const std::exception &e) {
    spdlog::error("Unhandled exception in main: {}", e.what());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
