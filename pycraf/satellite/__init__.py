#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This sub-package provides a helper class to get position of a satellite
in horizontal system (i.e, azimuth/elevation) from a two-line element
string (TLE).

All the complicated calculations are done with the help of the
`sgp4 Python package <https://pypi.python.org/pypi/sgp4/>`_.
'''

from .satellite import *
