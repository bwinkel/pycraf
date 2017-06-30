#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
import os
import numpy as np


ext_module_pathprof_cyprop = Extension(
    "pycraf.pathprof.cyprop",
    ["pycraf/pathprof/cyprop.pyx"],
    extra_compile_args=['-fopenmp', '-O3'],
    extra_link_args=['-fopenmp'],
    libraries=["m"],
    include_dirs=[np.get_include()],
    )


ext_module_pathprof_geodesics = Extension(
    "pycraf.pathprof.geodesics",
    ["pycraf/pathprof/geodesics.pyx"],
    extra_compile_args=['-fopenmp', '-O3'],
    extra_link_args=['-fopenmp'],
    libraries=["m"],
    include_dirs=[np.get_include()],
    )

setup(
    name="pycraf",
    version="0.23",
    description="pycraf",
    author="Benjamin Winkel",
    author_email="bwinkel@mpifr.de",
    url="https://github.com/bwinkel/pycraf",
    download_url="https://github.com/bwinkel/pycraf/archive/0.23.tar.gz",
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
        'pycraf': 'pycraf',
        'pycraf.antenna': 'pycraf/antenna',
        'pycraf.atm': 'pycraf/atm',
        'pycraf.conversions': 'pycraf/conversions',
        'pycraf.geospatial': 'pycraf/geospatial',
        'pycraf.helpers': 'pycraf/helpers',
        'pycraf.pathprof': 'pycraf/pathprof',
        'pycraf.protection': 'pycraf/protection',
        },
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        ext_module_pathprof_cyprop,
        ext_module_pathprof_geodesics,
        ],
    package_data={
        'pycraf': [
            'itudata/README.md',
            'itudata/LICENSE.ITU',
            'itudata/p.676-10/R-REC-P.676-10-201309_table1.csv',
            'itudata/p.676-10/R-REC-P.676-10-201309_table2.csv',
            'itudata/ra.769-2/ra_769_table1_limits_continuum.csv',
            'itudata/ra.769-2/ra_769_table2_limits_spectroscopy.csv',
            'itudata/p.452-16/refract_map.npz',
            'itudata/p.452-16/make_npy.py',
            'itudata/p.452-16/R-REC-P.452-16-201507/DN50.TXT',
            'itudata/p.452-16/R-REC-P.452-16-201507/LAT.TXT',
            'itudata/p.452-16/R-REC-P.452-16-201507/LON.TXT',
            'itudata/p.452-16/R-REC-P.452-16-201507/N050.TXT',
            'itudata/p.452-16/R-REC-P.452-16-201507/ReadMe.doc',
            ]
        },
    long_description='''pycraf ... the CRAF library.
        Contains useful functions for the daily life of a spectrum manager
        in RAS.
        ''',
    keywords=['pycraf', 'radio', 'compatibility study'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Cython',
        'Topic :: Scientific/Engineering',
        ],
    )
