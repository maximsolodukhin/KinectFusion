include(cmake/CPM.cmake)


function(kinectfusion_setup_ceres)
  # override option() with nop if set, eigen uses cmake 2.8.5 -_-
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

  # skip broken installs
  set(CMAKE_SKIP_INSTALL_RULES YES)

  # abseil in ceres includes CTest and pins BUILD_TESTING
  # overriding out option. well done abseil.
  set(_build_testing "$CACHE{BUILD_TESTING}")

  if(NOT TARGET Eigen3::Eigen)
    cpmaddpackage(
      NAME
      Eigen3
      GITLAB_REPOSITORY
      libeigen/eigen
      GIT_TAG
      5b3089dd7b11cfc08560b1638f09d6849bf67e5e
      SYSTEM
      YES
      OPTIONS
      "EIGEN_BUILD_DOC OFF"
      "EIGEN_BUILD_TESTING OFF"
      "EIGEN_BUILD_BLAS OFF"
      "EIGEN_BUILD_LAPACK OFF"
      "EIGEN_BUILD_DEMOS OFF"
      "EIGEN_BUILD_PKGCONFIG OFF"
      "BUILD_TESTING OFF")
  endif()

  if(NOT TARGET Ceres::ceres)
    cpmaddpackage(
      NAME
      ceres
      GITHUB_REPOSITORY
      "ceres-solver/ceres-solver"
      GIT_TAG
      0ba987acaf9e8674070f116ed624edf017d2b630
      SYSTEM
      YES
      OPTIONS
      "MINIGLOG ON"
      "GFLAGS OFF"
      "PROVIDE_UNINSTALL_TARGET OFF"
      "BUILD_TESTING OFF"
      "BUILD_EXAMPLES OFF"
      "BUILD_BENCHMARKS OFF"
      "SUITESPARSE OFF"
      # Ceres must not enable_language(CUDA) behind our back: its compiler
      # detection grabs whatever nvcc it finds (system CUDA 12 vs the real
      # toolkit) and we don't use its CUDA solvers anyway.
      "USE_CUDA OFF")
  endif()
  unset(CMAKE_SKIP_INSTALL_RULES)

  if(_build_testing STREQUAL "")
    unset(BUILD_TESTING CACHE)
  else()
    set(BUILD_TESTING "${_build_testing}" CACHE BOOL "" FORCE)
  endif()
endfunction()

# Done as a function so that updates to variables like
# CMAKE_CXX_FLAGS don't propagate out to other
# targets
function(kinectfusion_setup_dependencies)

  # For each dependency, see if it's
  # already been provided to us by a parent project
  if(NOT TARGET fmtlib::fmtlib)
    cpmaddpackage(
      NAME
      fmt
      GITHUB_REPOSITORY
      "fmtlib/fmt"
      GIT_TAG
      "12.1.0"
      SYSTEM
      YES)
  endif()

  if(NOT TARGET spdlog::spdlog)
    cpmaddpackage(
      NAME
      spdlog
      VERSION
      1.17.0
      GITHUB_REPOSITORY
      "gabime/spdlog"
      SYSTEM
      YES
      OPTIONS
      "SPDLOG_FMT_EXTERNAL ON")
  endif()

  if(NOT TARGET Catch2::Catch2WithMain)
    cpmaddpackage(
      NAME
      Catch2
      VERSION
      3.12.0
      GITHUB_REPOSITORY
      "catchorg/Catch2"
      SYSTEM
      YES)
  endif()

  if(NOT TARGET CLI11::CLI11)
    cpmaddpackage(
      NAME
      CLI11
      VERSION
      2.6.1
      GITHUB_REPOSITORY
      "CLIUtils/CLI11"
      SYSTEM
      YES)
  endif()

  if(NOT TARGET tomlplusplus::tomlplusplus)
    cpmaddpackage(
      NAME
      tomlplusplus
      VERSION
      3.4.0
      GITHUB_REPOSITORY
      "marzer/tomlplusplus"
      SYSTEM
      YES)
  endif()

  if(NOT TARGET spng_static)
    cpmaddpackage(
      NAME
      libspng
      VERSION
      0.7.4
      GITHUB_REPOSITORY
      "randy408/libspng"
      SYSTEM
      YES
      OPTIONS
      "SPNG_SHARED OFF")
  endif()

  kinectfusion_setup_ceres()

  # if(NOT TARGET ftxui::screen)
  #   cpmaddpackage(
  #     NAME
  #     FTXUI
  #     VERSION
  #     6.1.9
  #     GITHUB_REPOSITORY
  #     "ArthurSonzogni/FTXUI"
  #     SYSTEM
  #     YES)
  # endif()
endfunction()
