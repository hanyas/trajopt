cmake_minimum_required(VERSION 3.14)
project(core)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/")

set(ARMADILLO_LIBRARY "$ENV{HOME}/phd/libs/armadillo/")
include_directories(${ARMADILLO_LIBRARY}/include)

set(PYBIND_DIRECTORY "$ENV{HOME}/phd/libs/pybind11/build/mock_install/share/cmake/")
find_package(pybind11 CONFIG PATHS ${PYBIND_DIRECTORY})

pybind11_add_module(core src/util.cpp)

if (USE_OPENBLAS)
    set(OPENBLAS_LIBRARY "$ENV{HOME}/phd/libs/OpenBLAS/")
    target_link_libraries(core PRIVATE ${OPENBLAS_LIBRARY}/libopenblas.a pthread gfortran)
else ()
    target_link_libraries(core PRIVATE ${ARMADILLO_LIBRARY}/libarmadillo.so)
endif()