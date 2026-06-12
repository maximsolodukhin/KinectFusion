#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <kinectfusion/sample_library.hpp>
#include <string>

int main(int argc, const char** argv) {
  try {
    CLI::App app{"KinectFusion"};
    app.set_version_flag("--version", std::string{KINECTFUSION_VERSION});

    CLI11_PARSE(app, argc, argv);

    spdlog::info("Hello, world! factorial(4) = {}", factorial(4));
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
