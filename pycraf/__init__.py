#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Top-level functionality:
'''

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# Enforce Python version check during package import.
# This is the same check as the one at the top of setup.py
import sys

__minimum_python_version__ = "3.5"

class UnsupportedPythonError(Exception):
    pass

if sys.version_info < tuple((int(val) for val in __minimum_python_version__.split('.'))):
    raise UnsupportedPythonError("pycraf does not support Python < {}".format(__minimum_python_version__))


if not _ASTROPY_SETUP_:
    # For egg_info test builds to pass, put package imports here.

    # from .example_mod import *

    import astropy

    if astropy.__version__ >= '4':
        astropy.physical_constants.set('astropyconst20')
        astropy.astronomical_constants.set('astropyconst20')

    from . import antenna
    from . import atm
    from . import conversions
    from . import geometry
    from . import geospatial
    from . import mc
    from . import pathprof
    from . import protection
    from . import satellite
    from . import utils
