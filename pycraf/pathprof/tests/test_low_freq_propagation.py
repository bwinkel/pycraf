#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

import os
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from astropy.tests.helper import assert_quantity_allclose
from astropy import units as apu
from astropy.units import Quantity
from ... import conversions as cnv
from ... import pathprof
from ...utils import check_astro_quantities
# from astropy.utils.data import get_pkg_data_filename
from astropy.utils.misc import NumpyRNGContext


def test_gw_prop():
    # Known magnetic field strength, distance, frequency, and ground type
#     Hm_dBu = -20 * u.dB(u.uA / u.m)
#     d = 10 * u.m
#     freq = 13 * u.MHz
#     key_ground_term = "Land"
#     E_limit = -54.866202 * u.dB(u.uV / u.m)
    
#     args_list = [
#         (Hm_dBu, d, freq, key_ground_term, E_limit),
#         (120 * u.dB(u.uA / u.m), 1000 * u.m, 15 * u.MHz, "Land", 40 * u.dB(u.uV / u.m)),
#     ]
    
    
    args_list = [
        (None, None, u.dB(u.uA / u.m)),
        (1.e-30, None, apu.m),
        (0.009, 30.01, apu.MHz),
        (None, None, str),
        (None, None, u.dB(u.uV / u.m)),
        ]
    check_astro_quantities(gw_prop.gw_prop, args_list)
    
    Hm_dBu = -20 * u.dB(u.uA / u.m)
    d = 10 * u.m
    freq = 13 * u.MHz
    key_ground_term = "Land"
    E_limit = -54.866202 * u.dB(u.uV / u.m)
    
    
    
    
    
    
    
    