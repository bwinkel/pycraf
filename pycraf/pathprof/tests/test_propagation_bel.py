#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from astropy.tests.helper import assert_quantity_allclose
from astropy import units as apu
from astropy.units import Quantity
from ... import conversions as cnv
from ... import pathprof
from ...utils import check_astro_quantities
from astropy.utils.misc import NumpyRNGContext


TOL_KWARGS = {'atol': 1.e-4, 'rtol': 1.e-4}


def test_building_entry_loss():

    # the below is not working as building_type has no unit
    # args_list = [
    #     (0.08, 100, apu.GHz),
    #     (-90, 90, apu.deg),
    #     (0, 1, cnv.dimless),
    #     (None, None, None),
    #     ]

    # check_astro_quantities(pathprof.building_entry_loss, args_list)

    freq = np.logspace(-1, 2, 4) * apu.GHz
    theta = 0 * apu.deg
    prob = [0.02, 0.5, 0.98] * cnv.dimless
    btype = pathprof.BuildingType.TRADITIONAL

    L_bel = pathprof.building_entry_loss(
        freq[np.newaxis], theta, prob[:, np.newaxis], btype
        )

    assert_quantity_allclose(
        L_bel,
        [
            [1.81238859, 2.21279183, 2.95193539, 3.99581293],
            [14.22372482, 14.31281335, 17.67349232, 23.96455418],
            [28.48546096, 32.53034413, 41.15318352, 51.85705288],
            ] * cnv.dB,
        )

    freq = np.logspace(-1, 2, 4) * apu.GHz
    theta = 10 * apu.deg
    prob = [0.02, 0.5, 0.98] * cnv.dimless
    btype = pathprof.BuildingType.TRADITIONAL

    L_bel = pathprof.building_entry_loss(
        freq[np.newaxis], theta, prob[:, np.newaxis], btype
        )

    assert_quantity_allclose(
        L_bel,
        [
            [2.26864060, 2.52305727, 3.25085556, 4.40754278],
            [15.12794026, 15.86048478, 19.66034869, 26.0673998],
            [29.67755212, 34.58541981, 43.2714092 , 53.9770191],
            ] * cnv.dB,
        )


    freq = np.logspace(-1, 2, 4) * apu.GHz
    theta = 0 * apu.deg
    prob = [0.02, 0.5, 0.98] * cnv.dimless
    btype = pathprof.BuildingType.THERM_EFF
    L_bel = pathprof.building_entry_loss(
        freq[np.newaxis], theta, prob[:, np.newaxis], btype
        )

    assert_quantity_allclose(
        L_bel,
        [
            [19.99505932, 9.38355493, 10.38780121, 15.17937640],
            [40.18854264, 31.01140106, 34.21212557, 56.11169074],
            [60.72502565, 56.45793080, 69.21254924, 99.45338111],
            ] * cnv.dB,
        )
