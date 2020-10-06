# Trajectory Optimization 

A toolbox for trajectory optimization of dynamical systems

## Installation

This toolbox uses pybind11 to compile C++ code and use it in python.
The following setup has been test only while using Conda envs.

- Install Pybind11:
   ```shell
   conda install pybind11
   ```
  
- Compile OpenBLAS:
    * Prequisits: libpthread, libgfortran, gcc-fortran
    * https://github.com/xianyi/OpenBLAS.git
    * Change USE_THREAD to (de-)activate multi-threading
   ```shell
   USE_THREAD=0 C=gcc FC=gfortran NO_AFFINITY=1 NO_SHARED=1 COMMON_OPT=" -O2 -march=native "  make
   ```
   * Edit local CMakeLists.txt to reflect the path of OpenBLAS

- Configure Armadillo:
    * http://arma.sourceforge.net/download.html
    * Only configure, do not make
    ```shell
    ./configure
    ```
    * Edit local CMakeLists.txt to reflect the path of Armadillo

- Install Python Package:
   ```shell
   pip install -e .
   ```
