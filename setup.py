import os
import sys
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        cmake_args = ['-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

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
    install_requires=['numpy', 'scipy', 'matplotlib', 'scikit-learn',
                      'autograd', 'gym', 'pathos'],
    ext_modules=[CMakeExtension('gps', './trajopt/gps/'),
                 CMakeExtension('ilqr', './trajopt/ilqr/'),
                 CMakeExtension('bspilqr', './trajopt/bspilqr/')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
