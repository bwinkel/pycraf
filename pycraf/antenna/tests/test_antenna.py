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
from ... import conversions as cnv
from ...utils import check_astro_quantities
from ...antenna import ras, imt, fixedlink
# from astropy.utils.misc import NumpyRNGContext


def test_ras_pattern():

    args_list = [
        (-180, 180, apu.deg),
        (0.1, 1000., apu.m),
        (0.001, 2, apu.m),
        (0, None, cnv.dimless),
        ]
    check_astro_quantities(ras.ras_pattern, args_list)

    phi = np.logspace(0, 2, 10) * apu.deg
    diam = np.linspace(10, 100, 10) * apu.m

    assert_quantity_allclose(
        ras.ras_pattern(phi, diam, 0.21 * apu.m),
        [
            37.82967732, 23.44444444, 17.88888889, 12.33333333,
            6.77777778, 0.66666667, -6., -12., -12., -7.
            ] * cnv.dB
        )

    assert_quantity_allclose(
        ras.ras_pattern(phi, diam, 0.21 * apu.m, eta_a=20 * apu.percent),
        [
            30.83997728, 23.44444444, 17.88888889, 12.33333333,
            6.77777778, 0.66666667, -6., -12., -12., -7.
            ] * cnv.dB
        )


def test_ras_pattern_broadcast():

    phi = np.logspace(0, 2, 5) * apu.deg
    diam = np.linspace(10, 100, 2) * apu.m

    assert_quantity_allclose(
        ras.ras_pattern(phi, diam[:, np.newaxis], 0.21 * apu.m),
        [
            [37.82967732, 16.5, 4., -11., -7.],
            [29., 16.5, 4., -11., -7.]
            ] * cnv.dB
        )

    assert_quantity_allclose(
        ras.ras_pattern(phi[:, np.newaxis], diam, 0.21 * apu.m),
        [
            [37.82967732, 29.],
            [16.5, 16.5],
            [4., 4.],
            [-11., -11.],
            [-7., -7.]
            ] * cnv.dB
        )


def test_ras_pattern_bessel():

    phi = np.linspace(0.1, 2, 10) * apu.deg
    wavlen = [0.5, 0.21, 0.01] * apu.m

    assert_quantity_allclose(
        ras.ras_pattern(
            phi, 100 * apu.m, wavlen[:, np.newaxis], do_bessel=True
            ),
        [
            [54.62345427, 36.15257448, np.nan, 33.54936039,
             np.nan, 27.43022925, 25.60843495, 24.04885413,
             22.68541584, 21.47425011],
            [54.58317856, 41.34826701, 29.00208721, np.nan,
             np.nan, 27.43022925, 25.60843495, 24.04885413,
             22.68541584, 21.47425011],
            [np.nan, 29.99803136, 24.97163212, np.nan,
             31.89615114, 27.43022925, 25.60843495, 24.04885413,
             22.68541584, 21.47425011]
            ] * cnv.dB
        )


def test_imt2020_single_element_pattern():

    args_list = [
        (-180, 180, apu.deg),
        (-90, 90, apu.deg),
        (None, None, cnv.dB),
        (0, None, cnv.dimless),
        (0, None, cnv.dimless),
        (0, None, apu.deg),
        (0, None, apu.deg),
        ]
    check_astro_quantities(imt.imt2020_single_element_pattern, args_list)

    azims = np.arange(-180, 180, 0.5)[::200] * apu.deg
    elevs = np.arange(-90, 90, 0.5)[::100] * apu.deg

    # BS (outdoor) according to IMT.PARAMETER table 10 (multipage!)
    G_Emax = 5 * cnv.dB
    A_m, SLA_nu = 30. * cnv.dimless, 30. * cnv.dimless
    azim_3db, elev_3db = 65. * apu.deg, 65. * apu.deg

    gains_single = imt.imt2020_single_element_pattern(
        azims[:, np.newaxis], elevs[np.newaxis],
        G_Emax,
        A_m, SLA_nu,
        azim_3db, elev_3db
        )

    assert_quantity_allclose(
        gains_single,
        [
            [-25., -25., -25., -25.],
            [-25., -17.72189349, -13.46153846, -23.40236686],
            [-19.14201183, -0.68047337, 3.57988166, -6.36094675],
            [-25., -25., -25., -25.]
            ] * cnv.dB
        )


def test_imt2020_composite_pattern():

    args_list = [
        (-180, 180, apu.deg),
        (-90, 90, apu.deg),
        (-180, 180, apu.deg),
        (-90, 90, apu.deg),
        (None, None, cnv.dB),
        (0, None, cnv.dimless),
        (0, None, cnv.dimless),
        (0, None, apu.deg),
        (0, None, apu.deg),
        (0, None, cnv.dimless),
        (0, None, cnv.dimless),
        ]

    # not working:
    # check_astro_quantities(imt.imt2020_composite_pattern, args_list)

    azims = np.arange(-180, 180, 0.5)[::200] * apu.deg
    elevs = np.arange(-90, 90, 0.5)[::100] * apu.deg

    # BS (outdoor) according to IMT.PARAMETER table 10 (multipage!)
    G_Emax = 5 * cnv.dB
    A_m, SLA_nu = 30. * cnv.dimless, 30. * cnv.dimless
    azim_3db, elev_3db = 65. * apu.deg, 65. * apu.deg

    d_H, d_V = 0.5 * cnv.dimless, 0.5 * cnv.dimless
    N_H, N_V = 8, 8

    azim_i, elev_j = 0 * apu.deg, 0 * apu.deg

    gains_array = imt.imt2020_composite_pattern(
        azims[:, np.newaxis], elevs[np.newaxis],
        azim_i, elev_j,
        G_Emax,
        A_m, SLA_nu,
        azim_3db, elev_3db,
        d_H, d_V,
        N_H, N_V,
        )

    assert_quantity_allclose(
        gains_array,
        [
            [-3.3119858e+02, -2.3773064e+01, -1.5343370e+01, -2.4861577e+01],
            [-3.4144141e+02, -5.9032086e+01, -3.0494818e+01, -5.8627716e+01],
            [-3.4980321e+02, -2.6020485e+01, 7.2968710e-02, -1.4303173e+01],
            [-3.2730426e+02, -4.1613638e+01, -3.3511436e+01, -4.1442556e+01]
            ] * cnv.dB,
        atol=1.e-2 * cnv.dB, rtol=1.e-4,
        )

    azim_i, elev_j = 30 * apu.deg, 15 * apu.deg

    gains_array = imt.imt2020_composite_pattern(
        azims[:, np.newaxis], elevs[np.newaxis],
        azim_i, elev_j,
        G_Emax,
        A_m, SLA_nu,
        azim_3db, elev_3db,
        d_H, d_V,
        N_H, N_V,
        )

    assert_quantity_allclose(
        gains_array,
        [
            [-71.65367508, -48.39750099, -51.72422600, -53.09899139],
            [-71.65367508, -34.74055308, -42.48623672, -23.25435181],
            [-65.79568691, -37.13879018, -14.86999894, -23.49184427],
            [-71.65367508, -38.27782536, -41.34740448, -44.02411652]
            ] * cnv.dB
        )


def test_fl_pattern():

    args_list = [
        (-180, 180, apu.deg),
        (0.1, 1000., apu.m),
        (0.001, 2, apu.m),
        (None, None, cnv.dBi),
        ]
    check_astro_quantities(fixedlink.fl_pattern, args_list)

    phi = np.logspace(0, 2, 10) * apu.deg
    diam = np.linspace(1, 10, 10) * apu.m

    assert_quantity_allclose(
        fixedlink.fl_pattern(phi, diam, 0.21 * apu.m, 20 * cnv.dBi),
        [
            19.94331066, 19.36903415, 19.3235294, 22.53492637,
            16.01027068, 9.66290267, 3.43787921, -2.69759581,
            -26.32023215, -26.77780705
            ] * cnv.dB
        )


def test_fl_pattern_broadcast():

    args_list = [
        (-180, 180, apu.deg),
        (0.1, 1000., apu.m),
        (0.001, 2, apu.m),
        (None, None, cnv.dBi),
        ]
    check_astro_quantities(fixedlink.fl_pattern, args_list)

    phi = np.logspace(0, 2, 10) * apu.deg
    diam = np.linspace(1, 10, 2) * apu.m

    # apparently, if phi == 1, there is a problem!
    assert_quantity_allclose(
        fixedlink.fl_pattern(
            phi[:, np.newaxis], diam, 0.21 * apu.m, 20 * cnv.dBi
            ),
        [
            [19.94331066, 0.],
            [19.84225854, 29.66663739],
            [19.56107501, 24.11108184],
            [18.77866514, 18.55552628],
            [22.99997073, 12.99997073],
            [17.44441517, 7.44441517],
            [11.88885961, 1.88885961],
            [6.33330406, -3.66669594],
            [-16.77780705, -26.77780705],
            [-16.77780705, -26.77780705]
            ] * cnv.dB
        )


def test_fl_hpbw_from_size():

    args_list = [
        (0.1, 1000., apu.m),
        (0.001, 2, apu.m),
        ]
    check_astro_quantities(fixedlink.fl_hpbw_from_size, args_list)

    diam = np.linspace(1, 10, 10) * apu.m

    # apparently, if phi == 1, there is a problem!
    assert_quantity_allclose(
        fixedlink.fl_hpbw_from_size(diam, 0.21 * apu.m),
        [
            14.7, 7.35, 4.9, 3.675, 2.94, 2.45, 2.1, 1.8375, 1.63333333, 1.47
            ] * apu.deg
        )


def test_fl_G_max_from_size():

    args_list = [
        (0.1, 1000., apu.m),
        (0.001, 2, apu.m),
        ]
    check_astro_quantities(fixedlink.fl_G_max_from_size, args_list)

    diam = np.linspace(1, 10, 10) * apu.m

    # apparently, if phi == 1, there is a problem!
    assert_quantity_allclose(
        fixedlink.fl_G_max_from_size(diam, 0.21 * apu.m),
        [
            21.25561411, 27.27621402, 30.7980392, 33.29681393,
            35.23501419, 36.81863911, 38.15757491, 39.31741385,
            40.34046429, 41.25561411
            ] * cnv.dB
        )


def test_fl_G_max_from_hpbw():

    args_list = [
        (1.e-3, 90., apu.deg),
        ]
    check_astro_quantities(fixedlink.fl_G_max_from_hpbw, args_list)

    hpbw = np.linspace(1, 10, 10) * apu.deg

    # apparently, if phi == 1, there is a problem!
    assert_quantity_allclose(
        fixedlink.fl_G_max_from_hpbw(hpbw),
        [
            44.5, 38.47940009, 34.95757491, 32.45880017,
            30.52059991, 28.93697499, 27.5980392, 26.43820026,
            25.41514981, 24.5
            ] * cnv.dB
        )
