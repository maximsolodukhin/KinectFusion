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

add_library(sample_library sample_library.cpp)

add_library(kinectfusion::sample_library ALIAS sample_library)

target_link_libraries(sample_library PRIVATE kinectfusion_options kinectfusion_warnings)

target_include_directories(sample_library ${WARNING_GUARD} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                                                      $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

target_compile_features(sample_library PUBLIC cxx_std_${CMAKE_CXX_STANDARD})

kinectfusion_configure_linker(sample_library)

set_target_properties(
  sample_library
  PROPERTIES VERSION ${PROJECT_VERSION}
             CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN YES)

generate_export_header(sample_library EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/kinectfusion/sample_library_export.hpp)

if(NOT BUILD_SHARED_LIBS)
  target_compile_definitions(sample_library PUBLIC SAMPLE_LIBRARY_STATIC_DEFINE)
endif()

# each component adds its sources via target_sources()
target_sources(kinectfusion PRIVATE depth_processing.cpp icp_optimizer.cpp virtual_sensor.cpp volume.cpp)
add_subdirectory(image_proc)
