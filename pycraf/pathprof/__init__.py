#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Contains functions to construct height profiles and terrain maps from `SRTM
<https://www2.jpl.nasa.gov/srtm/>`_ data, calculate shortest path on the
Geoid - so-called geodesics -, and perform path propagation/attenuation
estimation.
'''

from .cyprop import *
from .geodesics import *
from .gis import *
from .heightprofile import *
from .helper import *
from .propagation import *
from .imt import *
from .srtm import *

_clutter_table = '''
+-------+-------------------+------+------+
| Value | Alias             | |ha| | |dk| |
+=======+===================+======+======+
| -1    | UNKNOWN           | 0    | 0    |
+-------+-------------------+------+------+
| 0     | SPARSE            | 4    | 100  |
+-------+-------------------+------+------+
| 1     | VILLAGE           | 5    | 70   |
+-------+-------------------+------+------+
| 2     | DECIDIOUS_TREES   | 15   | 50   |
+-------+-------------------+------+------+
| 3     | CONIFEROUS_TREES  | 20   | 50   |
+-------+-------------------+------+------+
| 4     | TROPICAL_FOREST   | 20   | 30   |
+-------+-------------------+------+------+
| 5     | SUBURBAN          | 9    | 25   |
+-------+-------------------+------+------+
| 6     | DENSE_SUBURBAN    | 12   | 20   |
+-------+-------------------+------+------+
| 7     | URBAN             | 20   | 20   |
+-------+-------------------+------+------+
| 8     | DENSE_URBAN       | 25   | 20   |
+-------+-------------------+------+------+
| 9     | HIGH_URBAN        | 35   | 20   |
+-------+-------------------+------+------+
| 10    | INDUSTRIAL_ZONE   | 20   | 50   |
+-------+-------------------+------+------+

.. |ha| replace:: :math:`h_\mathrm{a}~[\mathrm{m}]`
.. |dk| replace:: :math:`d_\mathrm{k}~[\mathrm{m}]`
'''

# enum docstring can't be set within pyx-file. Need to monkey-patch it here.
CLUTTER.__doc__ = '''
Clutter types are defined according to `ITU-R Recommendation P.452-16
<https://www.itu.int/rec/R-REC-P.452-16-201507-I/en>`_.

''' + _clutter_table

# __doc__ += _clutter_table
