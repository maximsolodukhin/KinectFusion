# Edited https://github.com/cpp-best-practices/cmake_template/blob/main/ProjectOptions.cmake

include(CMakeDependentOption)
include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)

macro(kinectfusion_supports_sanitizers)
  # Emscripten doesn't support sanitizers
  if((CMAKE_CXX_COMPILER_ID MATCHES ".*Clang.*" OR CMAKE_CXX_COMPILER_ID MATCHES ".*GNU.*") AND NOT WIN32)
    message(STATUS "Sanity checking UndefinedBehaviorSanitizer, it should be supported on this platform")
    set(TEST_PROGRAM "int main() { return 0; }")

    # Check if UndefinedBehaviorSanitizer works at link time
    set(_old_required_flags "${CMAKE_REQUIRED_FLAGS}")
    set(_old_required_link_options "${CMAKE_REQUIRED_LINK_OPTIONS}")
    set(CMAKE_REQUIRED_FLAGS "-fsanitize=undefined")
    set(CMAKE_REQUIRED_LINK_OPTIONS "-fsanitize=undefined")
    check_cxx_source_compiles("${TEST_PROGRAM}" HAS_UBSAN_LINK_SUPPORT)
    set(CMAKE_REQUIRED_FLAGS "${_old_required_flags}")
    set(CMAKE_REQUIRED_LINK_OPTIONS "${_old_required_link_options}")
    unset(_old_required_flags)
    unset(_old_required_link_options)

    if(HAS_UBSAN_LINK_SUPPORT)
      message(STATUS "UndefinedBehaviorSanitizer is supported at both compile and link time.")
      set(SUPPORTS_UBSAN ON)
    else()
      message(WARNING "UndefinedBehaviorSanitizer is NOT supported at link time.")
      set(SUPPORTS_UBSAN OFF)
    endif()
  else()
    set(SUPPORTS_UBSAN OFF)
  endif()

  if((CMAKE_CXX_COMPILER_ID MATCHES ".*Clang.*" OR CMAKE_CXX_COMPILER_ID MATCHES ".*GNU.*") AND WIN32)
    set(SUPPORTS_ASAN OFF)
  else()
    if (NOT WIN32)
      message(STATUS "Sanity checking AddressSanitizer, it should be supported on this platform")
      set(TEST_PROGRAM "int main() { return 0; }")

      # Check if AddressSanitizer works at link time
      set(_old_required_flags "${CMAKE_REQUIRED_FLAGS}")
      set(_old_required_link_options "${CMAKE_REQUIRED_LINK_OPTIONS}")
      set(CMAKE_REQUIRED_FLAGS "-fsanitize=address")
      set(CMAKE_REQUIRED_LINK_OPTIONS "-fsanitize=address")
      check_cxx_source_compiles("${TEST_PROGRAM}" HAS_ASAN_LINK_SUPPORT)
      set(CMAKE_REQUIRED_FLAGS "${_old_required_flags}")
      set(CMAKE_REQUIRED_LINK_OPTIONS "${_old_required_link_options}")
      unset(_old_required_flags)
      unset(_old_required_link_options)

      if(HAS_ASAN_LINK_SUPPORT)
        message(STATUS "AddressSanitizer is supported at both compile and link time.")
        set(SUPPORTS_ASAN ON)
      else()
        message(WARNING "AddressSanitizer is NOT supported at link time.")
        set(SUPPORTS_ASAN OFF)
      endif()
    else()
      set(SUPPORTS_ASAN ON)
    endif()
  endif()
endmacro()


macro(kinectfusion_setup_options)
  option(KINECTFUSION_ENABLE_HARDENING "Enable hardening" ON)
  option(KINECTFUSION_ENABLE_COVERAGE "Enable coverage reporting" OFF)
  option(KINECTFUSION_ENABLE_CUDA "Build the CUDA backend" OFF)
  kinectfusion_supports_sanitizers()

  if(NOT PROJECT_IS_TOP_LEVEL OR KINECTFUSION_PACKAGING_MAINTAINER_MODE)
    option(KINECTFUSION_WARNINGS_AS_ERRORS "Treat Warnings As Errors" OFF)
    option(KINECTFUSION_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
    option(KINECTFUSION_ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
    option(KINECTFUSION_ENABLE_SANITIZER_UNDEFINED "Enable undefined sanitizer" OFF)
    option(KINECTFUSION_ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
    option(KINECTFUSION_ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
    option(KINECTFUSION_ENABLE_UNITY_BUILD "Enable unity builds" OFF)
    option(KINECTFUSION_ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
    option(KINECTFUSION_ENABLE_CPPCHECK "Enable cpp-check analysis" OFF)
  else()
    option(KINECTFUSION_WARNINGS_AS_ERRORS "Treat Warnings As Errors" ON)
    option(KINECTFUSION_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" ${SUPPORTS_ASAN})
    option(KINECTFUSION_ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
    option(KINECTFUSION_ENABLE_SANITIZER_UNDEFINED "Enable undefined sanitizer" ${SUPPORTS_UBSAN})
    option(KINECTFUSION_ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
    option(KINECTFUSION_ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
    option(KINECTFUSION_ENABLE_UNITY_BUILD "Enable unity builds" OFF)
    option(KINECTFUSION_ENABLE_CLANG_TIDY "Enable clang-tidy" ON)
    option(KINECTFUSION_ENABLE_CPPCHECK "Enable cpp-check analysis" ON)
  endif()

  if(NOT PROJECT_IS_TOP_LEVEL)
    mark_as_advanced(
      KINECTFUSION_WARNINGS_AS_ERRORS
      KINECTFUSION_ENABLE_SANITIZER_ADDRESS
      KINECTFUSION_ENABLE_SANITIZER_LEAK
      KINECTFUSION_ENABLE_SANITIZER_UNDEFINED
      KINECTFUSION_ENABLE_SANITIZER_THREAD
      KINECTFUSION_ENABLE_SANITIZER_MEMORY
      KINECTFUSION_ENABLE_UNITY_BUILD
      KINECTFUSION_ENABLE_CLANG_TIDY
      KINECTFUSION_ENABLE_CPPCHECK
      KINECTFUSION_ENABLE_COVERAGE
      KINECTFUSION_ENABLE_PCH
      KINECTFUSION_ENABLE_CACHE)
  endif()

  # No fuzzer in this project
  set(DEFAULT_FUZZER OFF)
endmacro()

macro(kinectfusion_global_options)
  kinectfusion_supports_sanitizers()

  if(KINECTFUSION_ENABLE_HARDENING AND KINECTFUSION_ENABLE_GLOBAL_HARDENING)
    include(cmake/Hardening.cmake)
    if(NOT SUPPORTS_UBSAN
       OR KINECTFUSION_ENABLE_SANITIZER_UNDEFINED
       OR KINECTFUSION_ENABLE_SANITIZER_ADDRESS
       OR KINECTFUSION_ENABLE_SANITIZER_THREAD
       OR KINECTFUSION_ENABLE_SANITIZER_LEAK)
      set(ENABLE_UBSAN_MINIMAL_RUNTIME FALSE)
    else()
      set(ENABLE_UBSAN_MINIMAL_RUNTIME TRUE)
    endif()
    message("${KINECTFUSION_ENABLE_HARDENING} ${ENABLE_UBSAN_MINIMAL_RUNTIME} ${KINECTFUSION_ENABLE_SANITIZER_UNDEFINED}")
    kinectfusion_enable_hardening(kinectfusion_options ON ${ENABLE_UBSAN_MINIMAL_RUNTIME})
  endif()
endmacro()

macro(kinectfusion_local_options)
  if(PROJECT_IS_TOP_LEVEL)
    include(cmake/StandardProjectSettings.cmake)
  endif()

  add_library(kinectfusion_warnings INTERFACE)
  add_library(kinectfusion_options INTERFACE)

  include(cmake/CompilerWarnings.cmake)
  kinectfusion_set_project_warnings(
    kinectfusion_warnings
    ${KINECTFUSION_WARNINGS_AS_ERRORS}
    ""
    ""
    ""
    "")

  include(cmake/Linker.cmake)
  # Must configure each target with linker options, we're avoiding setting it globally for now

  include(cmake/Sanitizers.cmake)
  kinectfusion_enable_sanitizers(
    kinectfusion_options
    ${KINECTFUSION_ENABLE_SANITIZER_ADDRESS}
    ${KINECTFUSION_ENABLE_SANITIZER_LEAK}
    ${KINECTFUSION_ENABLE_SANITIZER_UNDEFINED}
    ${KINECTFUSION_ENABLE_SANITIZER_THREAD}
    ${KINECTFUSION_ENABLE_SANITIZER_MEMORY})

  set_target_properties(kinectfusion_options PROPERTIES UNITY_BUILD ${KINECTFUSION_ENABLE_UNITY_BUILD})


  include(cmake/StaticAnalyzers.cmake)
  if(KINECTFUSION_ENABLE_CLANG_TIDY)
    kinectfusion_enable_clang_tidy(kinectfusion_options ${KINECTFUSION_WARNINGS_AS_ERRORS})
  endif()

  if(KINECTFUSION_WARNINGS_AS_ERRORS)
    check_cxx_compiler_flag("-Wl,--fatal-warnings" LINKER_FATAL_WARNINGS)
    if(LINKER_FATAL_WARNINGS)
      # This is not working consistently, so disabling for now
      # target_link_options(KINECTFUSION_options INTERFACE -Wl,--fatal-warnings)
    endif()
  endif()

  if(KINECTFUSION_ENABLE_HARDENING AND NOT KINECTFUSION_ENABLE_GLOBAL_HARDENING)
    include(cmake/Hardening.cmake)
    if(NOT SUPPORTS_UBSAN
       OR KINECTFUSION_ENABLE_SANITIZER_UNDEFINED
       OR KINECTFUSION_ENABLE_SANITIZER_ADDRESS
       OR KINECTFUSION_ENABLE_SANITIZER_THREAD
       OR KINECTFUSION_ENABLE_SANITIZER_LEAK)
      set(ENABLE_UBSAN_MINIMAL_RUNTIME FALSE)
    else()
      set(ENABLE_UBSAN_MINIMAL_RUNTIME TRUE)
    endif()
    kinectfusion_enable_hardening(kinectfusion_options OFF ${ENABLE_UBSAN_MINIMAL_RUNTIME})
  endif()

endmacro()
