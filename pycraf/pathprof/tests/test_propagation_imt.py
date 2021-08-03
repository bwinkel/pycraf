#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

import os
import pytest
# from functools import partial
import numpy as np
# from zipfile import ZipFile
from numpy.testing import assert_equal, assert_allclose
from astropy.tests.helper import assert_quantity_allclose, remote_data
from astropy import units as apu
from astropy.units import Quantity
from ... import conversions as cnv
from ... import pathprof
from ...utils import check_astro_quantities
# from astropy.utils.data import get_pkg_data_filename
from astropy.utils.misc import NumpyRNGContext
# import json
# from itertools import product
# import importlib


TOL_KWARGS = {'atol': 1.e-4, 'rtol': 1.e-4}


def test_clutter_imt():

    args_list = [
        (2, 67, apu.GHz),
        (0.25, None, apu.km),
        (0, 100, apu.percent),
        ]

    check_astro_quantities(pathprof.clutter_imt, args_list)

    freq = [2.5, 10, 20] * apu.GHz
    dist = np.logspace(2.5, 4, 4)[:, np.newaxis] * apu.m

    assert_quantity_allclose(
        pathprof.clutter_imt(freq, dist, 2 * apu.percent, num_end_points=1),
        [
            [9.70031407, 11.67179752, 12.59465169],
            [14.90452998, 20.26362512, 22.56060172],
            [14.99471557, 20.77252115, 23.65890113],
            [14.99509074, 20.77485862, 23.66473226]
            ] * cnv.dB,
        )

    assert_quantity_allclose(
        pathprof.clutter_imt(freq, dist, 50 * apu.percent, num_end_points=2),
        [
            [44.05089174, 47.99385864, 49.83956699],
            [54.45932356, 65.17751385, 69.77146704],
            [54.63969475, 66.19530591, 71.96806586],
            [54.64044508, 66.19998085, 71.97972813]
            ] * cnv.dB,
        )

    assert_quantity_allclose(
        pathprof.clutter_imt(freq, dist, 90 * apu.percent, num_end_points=1),
        [
            [29.71581878, 31.68730223, 32.61015641],
            [34.92003469, 40.27912984, 42.57610643],
            [35.01022029, 40.78802587, 43.67440584],
            [35.01059545, 40.79036334, 43.68023698]
            ] * cnv.dB,
        )


def test_imt_rural_macro_losses():

    _func = pathprof.imt_rural_macro_losses
    args_list = [
        (0.5, 30, apu.GHz),
        (0, 100000, apu.m),
        (10, 150, apu.m),
        (1, 10, apu.m),
        (5, 50, apu.m),
        (5, 50, apu.m),
        ]

    check_astro_quantities(_func, args_list)

    freq = [1, 5] * apu.GHz
    dist = np.array([5, 20, 1000, 20000])[:, np.newaxis] * apu.m

    PL_los, PL_nlos, los_prob = _func(freq, dist)
    print(PL_los)
    print(PL_nlos)
    print(los_prob)
    assert_quantity_allclose(
        PL_los,
        [
            [      np.nan,       np.nan],
            [ 64.38071083,  78.36011092],
            [ 94.57828524, 108.55768533],
            [      np.nan,       np.nan],
            ] * cnv.dB,
        )
    assert_quantity_allclose(
        PL_nlos,
        [
            [      np.nan,       np.nan],
            [ 65.10847815,  79.08787824],
            [119.54294519, 133.52234528],
            [      np.nan,       np.nan],
            ] * cnv.dB,
        )
    assert_quantity_allclose(
        los_prob,
        [
            [1.00000000e+00, 1.00000000e+00],
            [9.90049834e-01, 9.90049834e-01],
            [3.71576691e-01, 3.71576691e-01],
            [2.08186856e-09, 2.08186856e-09],
            ] * cnv.dimless,
        )

    PL_los, PL_nlos, los_prob = _func(freq, dist, h_bs=20 * apu.m)
    print(PL_los)
    print(PL_nlos)
    assert_quantity_allclose(
        PL_los,
        [
            [      np.nan,       np.nan],
            [ 61.17035763,  75.14975772],
            [ 98.00008131, 108.55367443],
            [      np.nan,       np.nan],
            ] * cnv.dB,
        )
    assert_quantity_allclose(
        PL_nlos,
        [
            [      np.nan,       np.nan],
            [ 64.01112576,  77.99052585],
            [125.64356998, 139.62297006],
            [      np.nan,       np.nan],
            ] * cnv.dB,
        )


def test_imt_urban_macro_losses():

    _func = pathprof.imt_urban_macro_losses
    args_list = [
        (0.5, 30, apu.GHz),
        (0, 100000, apu.m),
        (10, 150, apu.m),
        (1, 13, apu.m),
        ]

    check_astro_quantities(_func, args_list)

    freq = [1, 5] * apu.GHz
    dist = np.array([5, 20, 1000, 20000])[:, np.newaxis] * apu.m

    PL_los, PL_nlos, los_prob = _func(freq, dist)
    print(PL_los)
    print(PL_nlos)
    print(los_prob)
    assert_quantity_allclose(
        PL_los,
        [
            [      np.nan,       np.nan],
            [ 60.76626079,  74.74566088],
            [108.24721393, 109.7252045 ],
            [      np.nan,       np.nan],
            ] * cnv.dB,
        )
    assert_quantity_allclose(
        PL_nlos,
        [
            [      np.nan,       np.nan],
            [ 71.74479418,  85.72419426],
            [130.78468516, 144.76408525],
            [      np.nan,       np.nan],
            ] * cnv.dB,
        )
    assert_quantity_allclose(
        los_prob,
        [
            [1.00000000e+00, 1.00000000e+00],
            [9.72799557e-01, 9.72799557e-01],
            [1.80001255e-02, 1.80001255e-02],
            [9.00000000e-04, 9.00000000e-04],
            ] * cnv.dimless,
        )

    PL_los, PL_nlos, los_prob = _func(freq, dist, h_bs=20 * apu.m)
    print(PL_los)
    print(PL_nlos)
    assert_quantity_allclose(
        PL_los,
        [
            [      np.nan,       np.nan],
            [ 59.57605227,  73.55545236],
            [110.07255004, 111.54965644],
            [      np.nan,       np.nan],
            ] * cnv.dB,
        )
    assert_quantity_allclose(
        PL_nlos,
        [
            [      np.nan,       np.nan],
            [ 69.63055103,  83.60995112],
            [130.78290388, 144.76230396],
            [      np.nan,       np.nan],
            ] * cnv.dB,
        )


def test_imt_urban_micro_losses():

    _func = pathprof.imt_urban_micro_losses
    args_list = [
        (0.5, 100, apu.GHz),
        (0, 100000, apu.m),
        (10, 150, apu.m),
        (1, 22.5, apu.m),
        ]

    check_astro_quantities(_func, args_list)

    freq = [1, 5] * apu.GHz
    dist = np.array([5, 20, 1000, 20000])[:, np.newaxis] * apu.m

    PL_los, PL_nlos, los_prob = _func(freq, dist)
    print(PL_los)
    print(PL_nlos)
    assert_quantity_allclose(
        PL_los,
        [
            [      np.nan,       np.nan],
            [ 60.47880565,  74.45820574],
            [118.53377126, 119.31141301],
            [      np.nan,       np.nan],
            ] * cnv.dB,
        )
    assert_quantity_allclose(
        PL_nlos,
        [
            [      np.nan,       np.nan],
            [ 69.59913521,  84.4871963 ],
            [128.3005538 , 143.18861489],
            [      np.nan,       np.nan],
            ] * cnv.dB,
        )
