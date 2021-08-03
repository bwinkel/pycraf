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

    PL_los, PL_nlos = _func(freq, dist)
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

    PL_los, PL_nlos = _func(freq, dist, h_bs=20 * apu.m)
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
        (1, 22.5, apu.m),
        (0, 20, apu.m),
        ]

    check_astro_quantities(_func, args_list)

    freq = [1, 5] * apu.GHz
    dist = np.array([5, 20, 1000, 20000])[:, np.newaxis] * apu.m

    PL_los, PL_nlos = _func(freq, dist)
    print(PL_los)
    print(PL_nlos)
    assert_quantity_allclose(
        PL_los,
        [
            [      np.nan,       np.nan],
            [ 62.89579811,  76.87519819],
            [123.32557394, 137.30497402],
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

    PL_los, PL_nlos = _func(freq, dist, h_bs=20 * apu.m)
    print(PL_los)
    print(PL_nlos)
    assert_quantity_allclose(
        PL_los,
        [
            [      np.nan,       np.nan],
            [ 62.60191301,  76.5813131 ],
            [125.19388113, 139.17328121],
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

    PL_los, PL_nlos = _func(freq, dist)
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
