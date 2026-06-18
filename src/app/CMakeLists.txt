add_executable(kinectfusion_app main.cpp app_options.cpp frame_output.cpp)

set_target_properties(kinectfusion_app PROPERTIES OUTPUT_NAME kinectfusion)

target_link_libraries(
  kinectfusion_app
  PRIVATE kinectfusion::kinectfusion_options
          kinectfusion::kinectfusion_warnings
          kinectfusion::kinectfusion
          CLI11::CLI11
          spdlog::spdlog)

target_compile_definitions(kinectfusion_app PRIVATE KINECTFUSION_VERSION="${PROJECT_VERSION}")

target_compile_features(kinectfusion_app PRIVATE cxx_std_${CMAKE_CXX_STANDARD})

kinectfusion_configure_linker(kinectfusion_app)
