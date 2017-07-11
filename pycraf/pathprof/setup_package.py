#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Allow cythonizing of our pyx files and provide custom compiler options.
'''

import os
from setuptools.extension import Extension
import platform
# Note: importing numpy from here won't work, see:
# http://docs.astropy.org/en/stable/development/ccython.html#using-numpy-c-headers
# import numpy as np
# 'include_dirs': [np.get_include()], --> 'include_dirs': 'numpy'

PYXDIR = os.path.relpath(os.path.dirname(__file__))


def get_extensions():

    comp_args = {
        'extra_compile_args': ['-fopenmp', '-O3'],
        'extra_link_args': ['-fopenmp'],
        'libraries': ['m'],
        'include_dirs': ['numpy'],
        }

    if platform.system().lower() == 'windows':
        comp_args = {
            'extra_compile_args': ['/openmp'],
            'include_dirs': ['numpy'],
            }
    elif 'darwin' in platform.system().lower():
        os.environ['CC'] = 'gcc-6'
        os.environ['LD'] = 'gcc-6'
        comp_args = {
            'extra_compile_args': [
                '-fopenmp', '-O3', '-mmacosx-version-min=10.7'
                ],
            'extra_link_args': ['-fopenmp'],
            'libraries': ['m'],
            'include_dirs': ['numpy'],
            }

    ext_module_pathprof_cyprop = Extension(
        name='pycraf.pathprof.cyprop',
        # ['pycraf/pathprof/cyprop.pyx'],
        sources=[os.path.join(PYXDIR, 'cyprop.pyx')],
        **comp_args
        )

    ext_module_pathprof_geodesics = Extension(
        name='pycraf.pathprof.cygeodesics',
        # ['pycraf/pathprof/geodesics.pyx'],
        sources=[os.path.join(PYXDIR, 'cygeodesics.pyx')],
        **comp_args
        )

    print('get_extensions', ext_module_pathprof_cyprop)
    return [ext_module_pathprof_cyprop, ext_module_pathprof_geodesics]
