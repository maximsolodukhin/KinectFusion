include(GenerateExportHeader)

add_library(kinectfusion)

add_library(kinectfusion::kinectfusion ALIAS kinectfusion)

target_link_libraries(
  kinectfusion
  PUBLIC Eigen3::Eigen
         Ceres::ceres
  PRIVATE kinectfusion_options
          kinectfusion_warnings)

target_include_directories(kinectfusion ${WARNING_GUARD} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                                                $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

target_include_directories(kinectfusion PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(kinectfusion PUBLIC cxx_std_${CMAKE_CXX_STANDARD})

kinectfusion_configure_linker(kinectfusion)

set_target_properties(
  kinectfusion
  PROPERTIES VERSION ${PROJECT_VERSION}
             CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN YES)

generate_export_header(kinectfusion EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/kinectfusion/kinectfusion_export.hpp)

if(NOT BUILD_SHARED_LIBS)
  target_compile_definitions(kinectfusion PUBLIC KINECTFUSION_STATIC_DEFINE)
endif()

# each component adds its sources via target_sources()
target_sources(
  kinectfusion
  PRIVATE comparison.cpp
          depth_processing.cpp
          icp_optimizer.cpp
          pipeline.cpp
          pipeline_set.cpp
          raycasting.cpp
          tsdf_integration.cpp
          virtual_sensor.cpp
          volume.cpp)

if(KINECTFUSION_HAS_CUDA)
  target_sources(kinectfusion PRIVATE cuda/depth_processing.cu cuda/device_pipeline.cu
                                      cuda/device_volume.cu cuda/icp_correspondence.cu
                                      cuda/raycasting.cu cuda/tsdf_integration.cu)
  target_compile_definitions(kinectfusion PRIVATE KINECTFUSION_HAS_CUDA)
  set_target_properties(kinectfusion PROPERTIES CUDA_STANDARD 20)
  target_compile_options(kinectfusion
                         PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  target_link_libraries(kinectfusion PUBLIC CUDA::cudart_static)
endif()

add_subdirectory(image_proc)
