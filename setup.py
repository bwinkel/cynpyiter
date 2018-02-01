#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
import numpy
import platform
import os


VERSION = '0.1.0'


COMP_ARGS = {
    'extra_compile_args': ['-fopenmp', '-O3'],
    'extra_link_args': ['-fopenmp'],
    'include_dirs': [numpy.get_include()],
    }

# need to handle compilation on Windows and MAC-OS machines:
if platform.system().lower() == 'windows':
    COMP_ARGS['extra_compile_args'] = ['/openmp']
    del COMP_ARGS['extra_link_args']

if 'darwin' in platform.system().lower():
    COMP_ARGS['extra_compile_args'].append('-mmacosx-version-min=10.7')
    os.environ["CC"] = "g++-6"
    os.environ["CPP"] = "cpp-6"
    os.environ["CXX"] = "g++-6"
    os.environ["LD"] = "gcc-6"

# Cython extensions:
ITER_EXT = Extension(
    'cynpyiter.iterfunc',
    ['cynpyiter/iterfunc.pyx'],
    **COMP_ARGS
    )

setup(
    name='cynpyiter',
    version=VERSION,
    author='Benjamin Winkel',
    author_email='bwinkel@mpifr.de',
    description=(
        'cynpyiter is an example package that demonstrates how to use'
        'the numpy npyiter C-API protocol with Cython and OpenMP'
        ),
    install_requires=[
        'setuptools',
        'numpy>=1.8',
        ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    packages=['cynpyiter'],
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        ITER_EXT,
        ],
    url='https://github.com/bwinkel/cynpyiter/',
    download_url='https://github.com/bwinkel/cynpyiter/tarball/{}'.format(VERSION),
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Cython',
        ],
)
