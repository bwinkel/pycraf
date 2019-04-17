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
        'extra_compile_args': ['-O3'],
        'include_dirs': ['numpy'],
        }

    if platform.system().lower() == 'windows':

        comp_args = {
            'include_dirs': ['numpy'],
            }

    elif 'darwin' in platform.system().lower():

        comp_args['extra_compile_args'].append('-mmacosx-version-min=10.7')

    ext_module_pathprof_atm_helper = Extension(
        name='pycraf.atm.atm_helper',
        sources=[os.path.join(PYXDIR, 'atm_helper.pyx')],
        **comp_args
        )

    return [ext_module_pathprof_atm_helper]
