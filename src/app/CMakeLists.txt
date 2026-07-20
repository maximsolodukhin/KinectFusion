set(KINECTFUSION_ENABLE_LOGGING_DEFAULT ON)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(KINECTFUSION_ENABLE_LOGGING_DEFAULT OFF)
endif()

option(KINECTFUSION_ENABLE_LOGGING "Enable KinectFusion progress logging" ${KINECTFUSION_ENABLE_LOGGING_DEFAULT})

# CLI/TOML -> PipelineConfig mapping as a library so tests can pin the
# plumbing (it has silently regressed before).
add_library(kinectfusion_app_config STATIC app_options.cpp pipeline_config.cpp)
target_link_libraries(
  kinectfusion_app_config
  PUBLIC kinectfusion::kinectfusion CLI11::CLI11 tomlplusplus::tomlplusplus
  PRIVATE kinectfusion::kinectfusion_options kinectfusion::kinectfusion_warnings)
target_include_directories(kinectfusion_app_config PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(kinectfusion_app_config PUBLIC cxx_std_${CMAKE_CXX_STANDARD})
target_compile_definitions(kinectfusion_app_config PRIVATE KINECTFUSION_VERSION="${PROJECT_VERSION}")

add_executable(kinectfusion_app main.cpp frame_output.cpp reconstruction.cpp)

set_target_properties(kinectfusion_app PROPERTIES OUTPUT_NAME kinectfusion)

target_link_libraries(
  kinectfusion_app
  PRIVATE kinectfusion::kinectfusion_options
          kinectfusion::kinectfusion_warnings
          kinectfusion::kinectfusion
          kinectfusion_app_config
          CLI11::CLI11
          spdlog::spdlog
          tomlplusplus::tomlplusplus)

target_compile_definitions(kinectfusion_app PRIVATE KINECTFUSION_VERSION="${PROJECT_VERSION}")

if(KINECTFUSION_ENABLE_LOGGING)
  target_compile_definitions(kinectfusion_app PRIVATE KINECTFUSION_ENABLE_LOGGING)
endif()

target_compile_features(kinectfusion_app PRIVATE cxx_std_${CMAKE_CXX_STANDARD})

kinectfusion_configure_linker(kinectfusion_app)
