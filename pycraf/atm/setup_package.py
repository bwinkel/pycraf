#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Allow cythonizing of our pyx files and provide custom compiler options.
'''

import os
from setuptools.extension import Extension
# from extension_helpers import add_openmp_flags_if_available
import platform
import numpy as np
# Note: importing numpy from here won't work, see:
# http://docs.astropy.org/en/stable/development/ccython.html#using-numpy-c-headers
# import numpy as np
# 'include_dirs': [np.get_include()], --> 'include_dirs': 'numpy'

PYXDIR = os.path.relpath(os.path.dirname(__file__))


def get_extensions():

    comp_args = {
        'extra_compile_args': ['-O3'],
        'libraries': ['m'],
        # 'include_dirs': ['numpy'],
        'include_dirs': [np.get_include()],
        }

    if platform.system().lower() == 'windows':

        comp_args = {
            # 'include_dirs': ['numpy'],
            'include_dirs': [np.get_include()],
            }

    elif 'darwin' in platform.system().lower():

        extra_compile_args = ['-O3', '-mmacosx-version-min=10.7']
        comp_args['extra_compile_args'] = extra_compile_args

    ext_module_pathprof_atm_helper = Extension(
        name='pycraf.atm.atm_helper',
        sources=[os.path.join(PYXDIR, 'atm_helper.pyx')],
        **comp_args
        )

    return [ext_module_pathprof_atm_helper]
