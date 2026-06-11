include(GenerateExportHeader)

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
