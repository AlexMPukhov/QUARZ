cmake_minimum_required(VERSION 3.18)
project(QUARZ LANGUAGES C CXX)

# Backward compatibility: options were called QSRZ_* before the code was renamed QUARZ.
foreach(_opt FETCH_KOKKOS ENABLE_MPI ENABLE_OPENPMD BUILD_TESTS)
  if(DEFINED QSRZ_${_opt} AND NOT DEFINED QUARZ_${_opt})
    message(WARNING "QSRZ_${_opt} is deprecated, use QUARZ_${_opt}")
    set(QUARZ_${_opt} ${QSRZ_${_opt}} CACHE BOOL "" FORCE)
  endif()
endforeach()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# std::filesystem: GCC 8 has a different (experimental) ABI for it in libstdc++fs.  A GCC 8
# build that later loads a newer libstdc++.so (e.g. from a gcc module in LD_LIBRARY_PATH)
# crashes in std::filesystem::path.  CMake takes `c++` from the PATH, which on many
# clusters is the old system compiler even when a newer gcc module is loaded.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
  message(FATAL_ERROR "QUARZ needs GCC >= 9 (found ${CMAKE_CXX_COMPILER_VERSION}: ${CMAKE_CXX_COMPILER}).\n"
    "If a newer gcc module is loaded, configure a fresh build directory with\n"
    "  CXX=$(which g++) CC=$(which gcc) cmake -B build ...")
endif()

# Kokkos: either an installed Kokkos (-DKokkos_ROOT=...) or fetched from GitHub.
# The backend (OpenMP, CUDA, HIP, SYCL, Serial) is chosen when Kokkos is built,
# e.g. -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON  for an A100.
option(QUARZ_FETCH_KOKKOS "Download and build Kokkos if not found" ON)
find_package(Kokkos 4.0 QUIET)
if(NOT Kokkos_FOUND)
  if(QUARZ_FETCH_KOKKOS)
    message(STATUS "Kokkos not found - fetching Kokkos 4.4.01")
    include(FetchContent)
    FetchContent_Declare(kokkos
      GIT_REPOSITORY https://github.com/kokkos/kokkos.git
      GIT_TAG 4.4.01)
    FetchContent_MakeAvailable(kokkos)
  else()
    message(FATAL_ERROR "Kokkos not found; set Kokkos_ROOT or QUARZ_FETCH_KOKKOS=ON")
  endif()
endif()

add_library(quarz_core
  src/Config.cpp
  src/RadialGrid.cpp
  src/Tridiag.cpp
  src/FieldSolver.cpp
  src/Profiles.cpp
  src/Parser.cpp
  src/PlasmaSpecies.cpp
  src/Beam.cpp
  src/Simulation.cpp
  src/SimulationIO.cpp
  src/Parallel.cpp
  src/OpenPMDWriter.cpp
  src/Laser.cpp
  src/Ionization.cpp)
target_include_directories(quarz_core PUBLIC src)
target_link_libraries(quarz_core PUBLIC Kokkos::kokkos)

# MPI: domain decomposition along xi with time pipelining (optional)
option(QUARZ_ENABLE_MPI "Build with MPI (decomposition along xi)" ON)
if(QUARZ_ENABLE_MPI)
  find_package(MPI COMPONENTS C CXX)
  if(MPI_CXX_FOUND)
    target_link_libraries(quarz_core PUBLIC MPI::MPI_CXX)
    target_compile_definitions(quarz_core PUBLIC QUARZ_USE_MPI)
    message(STATUS "QUARZ: MPI enabled (${MPI_CXX_LIBRARY_VERSION_STRING})")
  else()
    message(WARNING "QUARZ: MPI not found - building the serial version (set QUARZ_ENABLE_MPI=OFF to silence)")
  endif()
endif()

# openPMD output (optional): https://github.com/openPMD/openPMD-api, e.g. -DopenPMD_ROOT=/path/to/install
option(QUARZ_ENABLE_OPENPMD "Build with openPMD output (needs openPMD-api)" ON)
if(QUARZ_ENABLE_OPENPMD)
  find_package(openPMD 0.15 CONFIG QUIET)
  if(openPMD_FOUND)
    target_link_libraries(quarz_core PUBLIC openPMD::openPMD)
    target_compile_definitions(quarz_core PUBLIC QUARZ_USE_OPENPMD)
    message(STATUS "QUARZ: openPMD output enabled (openPMD-api ${openPMD_VERSION}, MPI: ${openPMD_HAVE_MPI})")
    # rectilinear meshes (docs/openpmd_rectilinear.md): coordinate data sets referenced by the mesh
    # attribute gridCoordinates, if this openPMD-api supports them (Iteration::gridCoordinates)
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_LIBRARIES openPMD::openPMD)
    set(CMAKE_REQUIRED_FLAGS "-std=c++17")
    check_cxx_source_compiles("
      #include <openPMD/openPMD.hpp>
      int main() {
        openPMD::Iteration* it = nullptr;
        if (it) { it->gridCoordinates[\"r\"]; (void)it->gridCoordinatesPath(\"r\"); }
        openPMD::Mesh* m = nullptr;
        if (m) m->setGridCoordinates({\"\", \"\"});
        return 0;
      }" QUARZ_OPENPMD_HAS_GRID_COORDINATES)
    unset(CMAKE_REQUIRED_LIBRARIES)
    unset(CMAKE_REQUIRED_FLAGS)
    if(QUARZ_OPENPMD_HAS_GRID_COORDINATES)
      target_compile_definitions(quarz_core PUBLIC QUARZ_OPENPMD_GRID_COORDINATES)
      message(STATUS "QUARZ: openPMD-api writes coordinate data sets: native grid in the openPMD 2.0 form")
    else()
      message(STATUS "QUARZ: openPMD-api without coordinate data sets: native grid in the interim attribute form")
    endif()
    if(MPI_CXX_FOUND AND NOT openPMD_HAVE_MPI)
      message(WARNING "QUARZ: openPMD-api has no MPI support: openPMD output only for single-rank runs")
    endif()
  else()
    message(STATUS "QUARZ: openPMD-api not found - native output only (set openPMD_ROOT to enable openPMD)")
  endif()
endif()

add_executable(quarz src/main.cpp)
target_link_libraries(quarz PRIVATE quarz_core)

option(QUARZ_BUILD_TESTS "Build unit tests" ON)
if(QUARZ_BUILD_TESTS)
  enable_testing()
  add_executable(test_solvers tests/test_solvers.cpp)
  target_link_libraries(test_solvers PRIVATE quarz_core)
  add_test(NAME solvers COMMAND test_solvers)
  add_executable(test_pushers tests/test_pushers.cpp)
  target_link_libraries(test_pushers PRIVATE quarz_core)
  add_test(NAME pushers COMMAND test_pushers)
  add_executable(test_ab tests/test_ab.cpp)
  target_link_libraries(test_ab PRIVATE quarz_core)
  add_test(NAME adams_bashforth COMMAND test_ab)
  add_executable(test_filter tests/test_filter.cpp)
  target_link_libraries(test_filter PRIVATE quarz_core)
  add_test(NAME smoothing_filter COMMAND test_filter)
  add_executable(test_parser tests/test_parser.cpp)
  target_link_libraries(test_parser PRIVATE quarz_core)
  add_test(NAME parser COMMAND test_parser)
  add_executable(test_profiles tests/test_profiles.cpp)
  target_link_libraries(test_profiles PRIVATE quarz_core)
  add_test(NAME profiles COMMAND test_profiles)
endif()
