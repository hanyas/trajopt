import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


USE_OPENBLAS = os.environ.get('USE_OPENBLAS', False)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        cmake_args = ['-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DUSE_OPENBLAS=' + str(USE_OPENBLAS)]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(os.path.join(ext.sourcedir, self.build_temp)):
            os.makedirs(os.path.join(ext.sourcedir, self.build_temp))
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=os.path.join(ext.sourcedir, self.build_temp), env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=os.path.join(ext.sourcedir, self.build_temp))


setup(
    name='trajopt',
    version='0.0.1',
    author='Hany Abdulsamad',
    author_email='hany@robot-learning.de',
    description='A toolbox for trajectory optimization',
    long_description='',
    ext_modules=[CMakeExtension('gps', './trajopt/gps/'),
                 CMakeExtension('ilqr', './trajopt/ilqr/')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
