- Prequisits: CMake.

- Compile pybind11:
    * https://github.com/pybind/pybind11
    * https://pybind11.readthedocs.io/en/master/basics.html
    ```shell
    mkdir build
    cd build
    cmake ..
    make check -j 4
    ```
    * Edit local CMakeLists.txt to reflect the path to pybind11

Using Armadillo
===
- Compile Armadillo:
    * http://arma.sourceforge.net/download.html
    * Compile Armadillo with our OpenBLAS
    ```shell
    ./configure
    ccmake .
    # set OpenBLAS path to our previously compiled library
    # configure, generate and exit
    make -j 4
    ```
    * Edit local CMakeLists.txt to reflect the path of Armadillo

Using OpenBLAS
===
- Prequisits: libpthread, libgfortran, gcc-fortran

- Compile OpenBLAS:
    * Make sure you have the fortran compiler
    * https://github.com/xianyi/OpenBLAS.git
   ```shell
   USE_THREAD=1, CC=gcc, FC=gfortran, NO_AFFINITY=1 NO_SHARED=1 COMMON_OPT=" -O2 -march=native "  make
   ```     

- Configure Armadillo, no need to make:
    * http://arma.sourceforge.net/download.html
    ```shell
    ./configure
    ```
    * Edit local CMakeLists.txt to reflect the path of Armadillo
