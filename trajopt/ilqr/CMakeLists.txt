cmake_minimum_required(VERSION 3.14)
project(core)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/")

set(ARMADILLO_LIBRARY "$ENV{HOME}/phd/libs/armadillo/")
include_directories(${ARMADILLO_LIBRARY}/include)

find_package(pybind11)
pybind11_add_module(core src/util.cpp)

set(OPENBLAS_LIBRARY "$ENV{HOME}/phd/libs/OpenBLAS/")
target_link_libraries(core PRIVATE ${OPENBLAS_LIBRARY}/libopenblas.a pthread gfortran)