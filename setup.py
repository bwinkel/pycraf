#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup


setup(
    name="pycraf",
    version="0.2",
    description="pycraf",
    author="Benjamin Winkel",
    author_email="bwinkel@mpifr.de",
    url="http://www.astro.uni-bonn.de/~bwinkel",
    packages=[
        'pycraf',
        'pycraf.atm',
        'pycraf.conversions',
        'pycraf.geospatial',
        'pycraf.protection'
        ],
    install_requires=[
        'setuptools',
        'numpy>=1.8',
        # 'scipy>=0.15',
        'astropy>=1.1',
    ],
    package_dir={
        'pycraf': 'src',
        'pycraf.atm': 'src/atm',
        'pycraf.conversions': 'src/conversions',
        'pycraf.geospatial': 'src/geospatial',
        'pycraf.protection': 'src/protection',
        },
    package_data={
        'pycraf': [
            'atm/data/R-REC-P.676-10-201309_table1.csv',
            'atm/data/R-REC-P.676-10-201309_table2.csv',
            'protection/data/ra_769_table1_limits_continuum.csv',
            'protection/data/ra_769_table2_limits_spectroscopy.csv',
            ]
        },
    long_description='''pycraf ... the CRAF library.
    contains useful functions for the daily life'''
)
