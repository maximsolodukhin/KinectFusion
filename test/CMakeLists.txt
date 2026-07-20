cmake_minimum_required(VERSION 3.15...3.23)

project(CmakeConfigPackageTests
  VERSION ${PROJECT_VERSION}
  LANGUAGES CXX)

# ---- Test as standalone project the exported config package ----

if(PROJECT_IS_TOP_LEVEL OR TEST_INSTALLED_VERSION)
  enable_testing()

  find_package(kinectfusion CONFIG REQUIRED) # for kinectfusion_app, project_options, ...

  if(NOT TARGET kinectfusion_options)
    message(FATAL_ERROR "Required config package not found!")
    return() # be strictly paranoid for Template Janitor github action! CK
  endif()
endif()

# ---- Dependencies ----

if(NOT TARGET Catch2::Catch2WithMain)
  find_package(Catch2 3 CONFIG REQUIRED)
elseif(DEFINED Catch2_SOURCE_DIR)
  # Catch2 came from CPM; its extras/ is not on the module path
  list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
endif()
include(Catch)
# Provide a simple smoke test to make sure that the CLI works and can display a --help message
add_test(NAME cli.has_help COMMAND $<TARGET_FILE:kinectfusion_app> --help)
# CMake-script regression tests for cmake/Sanitizers.cmake (issue #141).
add_test(
  NAME cmake.sanitizers.apple_excludes_leak
  COMMAND ${CMAKE_COMMAND} -P
          ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_sanitizers_apple_excludes_leak.cmake)
add_test(
  NAME cmake.sanitizers.non_apple_includes_leak
  COMMAND ${CMAKE_COMMAND} -P
          ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_sanitizers_non_apple_includes_leak.cmake)

# Provide a test to verify that the version being reported from the application
# matches the version given to CMake. This will be important once you package
# your program. Real world shows that this is the kind of simple mistake that is easy
# to make, but also easy to test for.
add_test(NAME cli.version_matches COMMAND $<TARGET_FILE:kinectfusion_app> --version)
set_tests_properties(cli.version_matches PROPERTIES PASS_REGULAR_EXPRESSION "${kinectfusion_VERSION}")

add_executable(tests tests.cpp type_assertions.cpp app_config_tests.cpp)
target_link_libraries(
  tests
  PRIVATE kinectfusion::kinectfusion_warnings
          kinectfusion::kinectfusion_options
          kinectfusion::kinectfusion
          kinectfusion_app_config
          Catch2::Catch2WithMain)

kinectfusion_configure_linker(tests)

# Compile-only canary keeping the kernel-reachable headers C++20-compatible
# (nvcc's ceiling at 13.1 driver on lxhalle). 
# Cannot link kinectfusion targets: they force cxx_std_23.
add_library(cxx20_shared_headers OBJECT cxx20_shared_headers.cpp)
target_include_directories(cxx20_shared_headers PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(cxx20_shared_headers PRIVATE Eigen3::Eigen
                                                   kinectfusion::kinectfusion_warnings)
set_target_properties(cxx20_shared_headers PROPERTIES CXX_STANDARD 20
                                                      CXX_STANDARD_REQUIRED ON
                                                      CXX_EXTENSIONS OFF)
add_dependencies(tests cxx20_shared_headers)

if(KINECTFUSION_HAS_CUDA)
  add_library(cuda_type_assertions OBJECT cuda_type_assertions.cu)
  target_link_libraries(
    cuda_type_assertions
    PRIVATE kinectfusion::kinectfusion_warnings
            kinectfusion::kinectfusion_options
            kinectfusion::kinectfusion)
  # no c++ 23 on cuda 13.1 on lxhalle
  set_target_properties(cuda_type_assertions PROPERTIES CUDA_STANDARD 20)
  target_compile_options(cuda_type_assertions
                         PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  add_dependencies(tests cuda_type_assertions)
endif()

if(WIN32 AND BUILD_SHARED_LIBS)
  add_custom_command(
    TARGET tests
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:tests> $<TARGET_FILE_DIR:tests>
    COMMAND_EXPAND_LISTS)
endif()

# automatically discover tests that are defined in catch based test files you can modify the unittests. Set TEST_PREFIX
# to whatever you want, or use different for different binaries
catch_discover_tests(
  tests
  TEST_PREFIX
  "unittests."
  REPORTER
  XML
  OUTPUT_DIR
  .
  OUTPUT_PREFIX
  "unittests."
  OUTPUT_SUFFIX
  .xml)
