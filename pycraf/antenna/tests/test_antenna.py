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

    phi = np.linspace(0, 2, 11) * apu.deg
    wavlen = [0.5, 0.21, 0.01] * apu.m

    assert_quantity_allclose(
        ras.ras_pattern(
            phi, 100 * apu.m, wavlen[:, np.newaxis], do_bessel=True
            ),
        [
            [55.96359737, 50.07824589, np.nan, np.nan,
             33.70863163, 29., 27.02046885, 25.34679911,
             23.89700043, 22.61818737, 21.47425011],
            [63.49861156, np.nan, np.nan, 32.93738587,
             33.72159846, 29., 27.02046885, 25.34679911,
             23.89700043, 22.61818737, 21.47425011],
            [89.94299745, 44.3868825, np.nan, 36.43697464,
             np.nan, 29., 27.02046885, 25.34679911,
             23.89700043, 22.61818737, 21.47425011]
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

    azims = np.linspace(-170, 170, 7) * apu.deg
    elevs = np.linspace(-65, 65, 4) * apu.deg

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
            [-26.84474397, -26.78742409, -26.78742337, -26.84474599],
            [-38.93066406, -37.95333672, -37.95333672, -38.93066406],
            [-29.54158595, -27.4920591, -27.4920591, -29.54159263],
            [-7.59590143, 8.86832778, 8.86832778, -7.59590143],
            [-29.54159263, -27.4920591, -27.4920591, -29.54158595],
            [-38.93066406, -37.95333672, -37.95333672, -38.93066406],
            [-26.84474599, -26.78742337, -26.78742409, -26.84474397]
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
            [-38.41604328, -23.76623774, -37.31241322, -40.32924271],
            [-26.08025503, -22.76059723, -36.30677032, -27.99344826],
            [-19.27646115, -7.04822194, -20.59439408, -21.18965438],
            [-34.2128334, -9.48487504, -23.03104146, -36.12606049],
            [-33.92118076, -26.211614, -39.75806176, -35.83437732],
            [-41.83896255, -27.74344563, -41.28961372, -43.75215149],
            [-43.43812561, -26.77295995, -40.31912613, -45.35131454]
            ] * cnv.dB
        )

    # test rho
    rho = 50 * apu.percent

    gains_array = imt.imt2020_composite_pattern(
        azims[:, np.newaxis], elevs[np.newaxis],
        azim_i, elev_j,
        G_Emax,
        A_m, SLA_nu,
        azim_3db, elev_3db,
        d_H, d_V,
        N_H, N_V,
        rho,
        )

    assert_quantity_allclose(
        gains_array,
        [
            [-27.81689191, -24.33945388, -27.76250386, -27.88482141],
            [-25.50662625, -23.73752856, -27.70019412, -26.24376309],
            [-17.4177759, -6.17815893, -8.33298694, -17.95386317],
            [-10.00205684, 0.8616368, 0.66564671, -10.00499249],
            [-19.05914643, -8.42762696, -8.46233689, -19.08447626],
            [-27.92129159, -26.15860176, -27.9094286, -27.95279717],
            [-27.94851589, -25.79662728, -27.88453221, -27.97042871]
            ] * cnv.dB
        )


def test_imt2020_composite_pattern_broadcast():

    azims = np.linspace(-50, 50, 3) * apu.deg
    elevs = np.linspace(-65, 65, 3) * apu.deg

    azim_i, elev_j = [0, 10] * apu.deg, [0, 5] * apu.deg

    G_Emax = 5 * cnv.dB
    A_m, SLA_nu = 30. * cnv.dimless, 30. * cnv.dimless
    azim_3db, elev_3db = 65. * apu.deg, 65. * apu.deg

    d_H, d_V = 0.5 * cnv.dimless, 0.5 * cnv.dimless
    N_H, N_V = 8, 8

    gains_array = imt.imt2020_composite_pattern(
        azims[np.newaxis, :, np.newaxis, np.newaxis],
        elevs[:, np.newaxis, np.newaxis, np.newaxis],
        azim_i[np.newaxis, np.newaxis, np.newaxis],
        elev_j[np.newaxis, np.newaxis, :, np.newaxis],
        G_Emax,
        A_m, SLA_nu,
        azim_3db, elev_3db,
        d_H, d_V,
        N_H, N_V,
        )

    assert_quantity_allclose(
        gains_array,
        [
            [[[-28.44916445, -20.56675583],
              [-29.83977323, -22.01867205]],
             [[-7.59590143, -16.00107479],
              [-8.98651206, -17.3092823]],
             [[-28.44916445, -59.40173918],
              [-29.83977323, -58.8273068]]],
            [[[-15.46860319, -0.93379455],
              [-17.25319963, -2.69413352]],
             [[23.0618, 14.65663147],
              [21.2772007, 12.95442963]],
             [[-15.46860319, -5.31790286],
              [-17.25319963, -7.02605587]]],
            [[[-28.44916445, -20.5667544],
              [-49.57172209, -41.75039869]],
             [[-7.59590143, -16.00107002],
              [-28.7182579, -37.04088974]],
             [[-28.44916445, -59.40173918],
              [-49.57172209, -78.5662747]]]
            ] * cnv.dB,
        atol=1.e-4 * cnv.dB, rtol=1.e-6
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
