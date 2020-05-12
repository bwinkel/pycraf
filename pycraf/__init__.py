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

    # pycraf should try to avoid working with the astropy.constants module
    # as this introduced versioning in v4+; this can lead to serious
    # import problems all over the place, which are extremely hard to
    # debug; the code below solves *some* problems, but not all

    # import astropy

    # if astropy.__version__ >= '4':
    #     try:
    #         astropy.physical_constants.set('astropyconst20')
    #         astropy.astronomical_constants.set('astropyconst20')
    #     except RuntimeError as e:
    #         # import ipdb
    #         # ipdb.set_trace()
    #         if 'astropy.units is already imported' in e.args:
    #             e.args = (
    #                 'Please note that pycraf uses the astropy.constants '
    #                 'from Astropy v2 for backwards compatibility. '
    #                 'Starting from Astropy v4, a "ScienceState" is used '
    #                 'to allow versioning of physical constants. For '
    #                 'technical reasons, it is necessary to import the '
    #                 'astropy.units sub-package *after* pycraf.'
    #                 '(see https://github.com/bwinkel/pycraf/issues/24)',
    #                 )

    #             raise e

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
