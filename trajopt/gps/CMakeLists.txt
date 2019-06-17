cmake_minimum_required(VERSION 2.8.12)
project(core)

find_package(pybind11)

set(ARMADILLO_INCLUDES "$ENV{HOME}/phd/libs/armadillo/include")
set(OPENBLAS_LIBRARY "$ENV{HOME}/phd/libs/OpenBLAS/")

include_directories(${ARMADILLO_INCLUDES})

pybind11_add_module(core src/util.cpp)

target_link_libraries(core PRIVATE ${OPENBLAS_LIBRARY}/libopenblas.a pthread gfortran)
