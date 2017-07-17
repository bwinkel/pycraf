#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This subpackage contains a couple of functions to convert coordinates
between typical GIS frames, such as GPS (WGS84), UTM, and ETRS89. The
underlying transformations are based on the `pyproj
<https://pypi.python.org/pypi/pyproj>`_ package, which itself is a wrapper
around the `proj.4 <http://proj4.org/>`_ software.
'''


from .geospatial import *
