cmake_minimum_required(VERSION 2.8.12)
project(core)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/")

set(PYBIND_DIRECTORY "$ENV{HOME}/phd/libs/pybind11/")
set(ARMADILLO_LIBRARY "$ENV{HOME}/phd/libs/armadillo/")
set(OPENBLAS_LIBRARY "$ENV{HOME}/phd/libs/OpenBLAS/")

find_package(pybind11 PATHS ${PYBIND_DIRECTORY})

include_directories(${ARMADILLO_LIBRARY}/include)

pybind11_add_module(core src/util.cpp)

if (USE_OPENBLAS)
    target_link_libraries(core PRIVATE ${OPENBLAS_LIBRARY}/libopenblas.a pthread gfortran)
else ()
    target_link_libraries(core PRIVATE ${ARMADILLO_LIBRARY}/libarmadillo.so)
endif()