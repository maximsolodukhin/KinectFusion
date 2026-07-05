cmake_minimum_required(VERSION 3.29)

# Mostly taken from https://github.com/cpp-best-practices/cmake_template

# Disable modules support, this is broken with clang-tidy at the moment
cmake_policy(SET CMP0155 OLD)

# This template attempts to be "fetch_content"-able
# so that it works well with tools like CPM or other
# manual dependency management

# Only set the cxx_standard if it is not set by someone else
if (NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 23)
endif()

# strongly encouraged to enable this globally to avoid conflicts between
# -Wpedantic being enabled and -std=c++20 and -std=gnu++20 for example
# when compiling with PCH enabled
set(CMAKE_CXX_EXTENSIONS OFF)

# Set the project name and language
project(
  kinectfusion
  VERSION 0.1.0
  LANGUAGES CXX C)

include(cmake/PreventInSourceBuilds.cmake)

include(cmake/ProjectOptions.cmake)
kinectfusion_setup_options()
kinectfusion_global_options()
include(Dependencies.cmake)
kinectfusion_setup_dependencies()
kinectfusion_local_options()

# don't know if this should be set globally from here or not...
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

if(KINECTFUSION_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
endif()

target_compile_features(kinectfusion_options INTERFACE cxx_std_${CMAKE_CXX_STANDARD})

add_library(kinectfusion::kinectfusion_options ALIAS kinectfusion_options)
add_library(kinectfusion::kinectfusion_warnings ALIAS kinectfusion_warnings)

#add_library(kinectfusion::kinectfusion_options INTERFACE IMPORTED)
#add_library(kinectfusion::kinectfusion_warnings INTERFACE IMPORTED)

# Adding the src:
add_subdirectory(src)

# Don't even look at tests if we're not top level
if(NOT PROJECT_IS_TOP_LEVEL)
  return()
endif()

# Adding the tests:
include(CTest)

if(BUILD_TESTING)
  message(AUTHOR_WARNING "Building Tests. Be sure to check out test/constexpr_tests.cpp for constexpr testing")
  add_subdirectory(test)
endif()


# If MSVC is being used, and ASAN is enabled, we need to set the debugger environment
# so that it behaves well with MSVC's debugger, and we can run the target from visual studio
if(MSVC)
  get_all_installable_targets(all_targets)
  message("all_targets=${all_targets}")
  set_target_properties(${all_targets} PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=$(VC_ExecutablePath_x64);%PATH%")
endif()

# set the startup project for the "play" button in MSVC
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT kinectfusion_app)

if(CMAKE_SKIP_INSTALL_RULES)
  return()
endif()

# Add other targets that you want installed here, by default we just package the one executable
# we know we want to ship
# kinectfusion_package_project(
#   TARGETS
#   kinectfusion_app
#   kinectfusion_options
#   kinectfusion_warnings
#   # FIXME: this does not work! CK
#   # PRIVATE_DEPENDENCIES_CONFIGURED project_options project_warnings
# )

