- Cmake, libpthread, libgfortran are needed

- Install pybind11:
    https://github.com/pybind/pybind11

- Download Armadillo:
    http://arma.sourceforge.net/download.html

- Clone OpenBLAS:
    https://github.com/xianyi/OpenBLAS.git
    
- Extract Armadillo and OpenBLAS to ~/libs

- Make OpenBLAS using:
     USE_THREAD=1 NO_AFFINITY=1 NO_SHARED=1 COMMON_OPT=" -O2 -march=native "  make
     
 - Configure Armadillo, no need to make:
    ./configure
    
- Edit CMakeLists.txt to reflect the paths of Armadillo and OpenBLAS

