#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Top-level functionality:
'''

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

if not _ASTROPY_SETUP_:
    # For egg_info test builds to pass, put package imports here.

    # from .example_mod import *

    from . import antenna
    from . import atm
    from . import conversions
    from . import geospatial
    from . import utils
    from . import pathprof
    from . import protection
    from . import satellite
