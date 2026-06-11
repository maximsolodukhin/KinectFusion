add_executable(intro main.cpp)

target_link_libraries(
  intro
  PRIVATE kinectfusion::kinectfusion_options
          kinectfusion::kinectfusion_warnings
          kinectfusion::sample_library
          CLI11::CLI11
          spdlog::spdlog)

target_compile_definitions(intro PRIVATE KINECTFUSION_VERSION="${PROJECT_VERSION}")

target_compile_features(intro PRIVATE cxx_std_${CMAKE_CXX_STANDARD})

kinectfusion_configure_linker(intro)
