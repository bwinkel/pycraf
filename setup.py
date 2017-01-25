#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
import numpy as np


ext_module_pathprof_cyprop = Extension(
    "pycraf.pathprof.cyprop",
    ["src/pathprof/cyprop.pyx"],
    extra_compile_args=['-fopenmp', '-O3'],
    extra_link_args=['-fopenmp'],
    libraries=["m"],
    include_dirs=[np.get_include()],
)


ext_module_pathprof_geodesics = Extension(
    "pycraf.pathprof.geodesics",
    ["src/pathprof/geodesics.pyx"],
    extra_compile_args=['-fopenmp', '-O3'],
    extra_link_args=['-fopenmp'],
    libraries=["m"],
    include_dirs=[np.get_include()],
)


setup(
    name="pycraf",
    version="0.22",
    description="pycraf",
    author="Benjamin Winkel",
    author_email="bwinkel@mpifr.de",
    url="http://www.astro.uni-bonn.de/~bwinkel",
    packages=[
        'pycraf',
        'pycraf.antenna',
        'pycraf.atm',
        'pycraf.conversions',
        'pycraf.geospatial',
        'pycraf.helpers',
        'pycraf.pathprof',
        'pycraf.protection',
        ],
    install_requires=[
        'setuptools',
        'numpy>=1.8',
        # 'scipy>=0.15',
        'astropy>=1.1',
        'pyproj>=1.9',
    ],
    package_dir={
        'pycraf': 'src',
        'pycraf.antenna': 'src/antenna',
        'pycraf.atm': 'src/atm',
        'pycraf.conversions': 'src/conversions',
        'pycraf.geospatial': 'src/geospatial',
        'pycraf.helpers': 'src/helpers',
        'pycraf.pathprof': 'src/pathprof',
        'pycraf.protection': 'src/protection',
        },
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        ext_module_pathprof_cyprop,
        ext_module_pathprof_geodesics,
        ],
    package_data={
        'pycraf': [
            'atm/data/R-REC-P.676-10-201309_table1.csv',
            'atm/data/R-REC-P.676-10-201309_table2.csv',
            'protection/data/ra_769_table1_limits_continuum.csv',
            'protection/data/ra_769_table2_limits_spectroscopy.csv',
            'pathprof/data/refract_map.npz',
            ]
        },
    long_description='''pycraf ... the CRAF library.
    contains useful functions for the daily life'''
)
