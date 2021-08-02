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
        (0, None, cnv.dB),
        (0, None, cnv.dB),
        (0, None, apu.deg),
        (0, None, apu.deg),
        ]
    check_astro_quantities(imt.imt2020_single_element_pattern, args_list)

    azims = np.arange(-180, 180, 0.5)[::200] * apu.deg
    elevs = np.arange(-90, 90, 0.5)[::100] * apu.deg

    # BS (outdoor) according to IMT.PARAMETER table 10 (multipage!)
    G_Emax = 5 * cnv.dB
    A_m, SLA_nu = 30. * cnv.dB, 30. * cnv.dB
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
        (0, None, cnv.dB),
        (0, None, cnv.dB),
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
    A_m, SLA_nu = 30. * cnv.dB, 30. * cnv.dB
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

            [-26.84474666, -26.78742420, -26.78742420, -26.84474666],
            [-38.93066459, -37.95333703, -37.95333703, -38.93066459],
            [-29.54158970, -27.49204873, -27.49204873, -29.54158970],
            [-7.59590121, 8.86832789, 8.86832789, -7.59590121],
            [-29.54158970, -27.49204873, -27.49204873, -29.54158970],
            [-38.93066459, -37.95333703, -37.95333703, -38.93066459],
            [-26.84474666, -26.78742420, -26.78742420, -26.84474666]
            ] * cnv.dB,
        atol=1.e-4 * cnv.dB,  # rtol=1.e-4,
        )

    azim_i, elev_j = -30 * apu.deg, -15 * apu.deg

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
            [-38.41604646, -23.76623966, -37.31241105, -40.32923892],
            [-26.08025568, -22.76059837, -36.30676977, -27.99344814],
            [-19.27646106, -7.04822685, -20.59439824, -21.18965352],
            [-34.21285545, -9.48487260, -23.0310440, -36.12604791],
            [-33.92118696, -26.21161972, -39.75779112, -35.83437942],
            [-41.83896228, -27.74344561, -41.2896170, -43.75215474],
            [-43.4381214 , -26.77295971, -40.31913110, -45.35131386]
            ] * cnv.dB,
        atol=1.e-4 * cnv.dB,  # rtol=1.e-4,
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
            [-27.81689242, -24.33945495, -27.76250384, -27.88482087],
            [-25.50662648, -23.73752951, -27.70019423, -26.24376283],
            [-17.41777597, -6.17816113, -8.33298701, -17.95386296],
            [-10.00205693, 0.86163716, 0.66564674, -10.00499216],
            [-19.05914636, -8.42762686, -8.46233715, -19.08447646],
            [-27.92129175, -26.15860178, -27.90942827, -27.95279696],
            [-27.94851549, -25.79662730, -27.88453268, -27.97042889]
            ] * cnv.dB,
        atol=1.e-4 * cnv.dB,  # rtol=1.e-4,
        )


def test_imt2020_composite_pattern_oob():

    azims = np.linspace(-170, 170, 7) * apu.deg
    elevs = np.linspace(-65, 65, 4) * apu.deg

    # BS (outdoor) according to IMT.PARAMETER table 10 (multipage!)
    G_Emax = 5 * cnv.dB
    A_m, SLA_nu = 30. * cnv.dB, 30. * cnv.dB
    azim_3db, elev_3db = 65. * apu.deg, 65. * apu.deg

    d_H, d_V = 0.46 * cnv.dimless, 0.46 * cnv.dimless
    N_H, N_V = 8, 8
    k = 8

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
            [-26.97256187, -25.73457693, -25.73457693, -26.97256187],
            [-38.72211304, -44.96933643, -44.96933643, -38.72211304],
            [-30.73151994, -24.88846358, -24.88846358, -30.73151994],
            [-7.92057169, 8.65348278, 8.65348278, -7.92057169],
            [-30.73151994, -24.88846358, -24.88846358, -30.73151994],
            [-38.72211304, -44.96933643, -44.96933643, -38.72211304],
            [-26.97256187, -25.73457693, -25.73457693, -26.97256187]
            ] * cnv.dB,
        atol=1.e-4 * cnv.dB,  # rtol=1.e-4,
        )

    azim_i, elev_j = -30 * apu.deg, -15 * apu.deg

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
            [-36.71479239, -25.76694493, -40.92360020, -40.03564568],
            [-25.51056417, -22.44697868, -37.60363395, -28.83141746],
            [-18.33945759, -12.28584711, -27.44250238, -21.66031089],
            [-23.76359404, 1.22003128, -13.93662399, -27.08444734],
            [-44.76444619, -8.44673641, -23.60339168, -48.08529948],
            [-45.76930152, -38.72325572, -53.87991099, -49.09015481],
            [-55.50873432, -26.28244428, -41.43909955, -58.82958762]
            ] * cnv.dB,
        atol=1.e-4 * cnv.dB,  # rtol=1.e-4,
        )


def test_imt2020_composite_pattern_broadcast():

    azims = np.linspace(-50, 50, 3) * apu.deg
    elevs = np.linspace(-65, 65, 3) * apu.deg

    azim_i, elev_j = [0, -10] * apu.deg, [0, -5] * apu.deg

    G_Emax = 5 * cnv.dB
    A_m, SLA_nu = 30. * cnv.dB, 30. * cnv.dB
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
            [[[-28.44916415, -20.56675462],
              [-29.83977594, -22.01867157]],
             [[-7.59590121, -16.00107180],
              [-8.98651300, -17.30928289]],
             [[-28.44916415, -59.40530382],
              [-29.83977594, -58.82576260]]],
            [[[-15.46860066, -0.93379524],
              [-17.25319896, -2.69413471]],
             [[23.06179974, 14.65662915],
              [21.27720145, 12.95443156]],
             [[-15.46860066, -5.31790368],
              [-17.25319896, -7.02605793]]],
            [[[-28.44916415, -20.56675462],
              [-49.57151146, -41.75040709]],
             [[-7.59590121, -16.00107180],
              [-28.71824852, -37.04101841]],
             [[-28.44916415, -59.40530382],
              [-49.57151146, -78.55749812]]]
            ] * cnv.dB,
        atol=1.e-4 * cnv.dB,  # rtol=1.e-4,
        )


def test_imt_advanced_sectoral_peak_sidelobe_pattern_400_to_6000_mhz():

    _gfunc = imt.imt_advanced_sectoral_peak_sidelobe_pattern_400_to_6000_mhz
    args_list = [
        (-180, 180, apu.deg),
        (-90, 90, apu.deg),
        (None, None, cnv.dB),
        (0, None, apu.deg),
        (0, None, apu.deg),
        (0, None, cnv.dimless),
        (0, None, cnv.dimless),
        (0, None, cnv.dimless),
        (-90, 90, apu.deg),
        (-90, 90, apu.deg),
        ]
    check_astro_quantities(_gfunc, args_list)

    azims = np.arange(-180, 180, 0.5)[50:-50:150] * apu.deg
    elevs = np.arange(-90, 90, 0.5)[30:-30:70] * apu.deg

    G0 = 18. * cnv.dB
    phi_3db = 65. * apu.deg
    # theta_3db can be inferred in the following way:
    theta_3db = 31000 / G0.to(cnv.dimless) / phi_3db.value * apu.deg
    k_p, k_h, k_v = (0.7, 0.7, 0.3) * cnv.dimless
    tilt_m, tilt_e = (0, 0) * apu.deg

    bs_gain = _gfunc(
        0 * apu.deg, 0 * apu.deg,
        G0, phi_3db, theta_3db,
        k_p, k_h, k_v,
        tilt_m=tilt_m, tilt_e=tilt_e,
        )

    assert_quantity_allclose(bs_gain, G0)

    bs_gains = _gfunc(
        azims[np.newaxis], elevs[:, np.newaxis],
        G0, phi_3db, theta_3db,
        k_p, k_h, k_v,
        tilt_m=tilt_m, tilt_e=tilt_e,
        )

    assert_quantity_allclose(
        bs_gains,
        [
            [-6.45692316, -3.90234884, -0.58659194, -3.29935692, -6.45692316],
            [-6.45692316, -3.04567901, 1.38200832, -2.24047532, -6.45692316],
            [-6.45692316, 1.87668674, 12.69344956, 3.84378529, -6.45692316],
            [-6.45692316, -2.6577429, 2.27347326, -1.76096923, -6.45692316],
            [-6.45692316, -3.70733057, -0.1384461, -3.05830576, -6.45692316]
            ] * cnv.dB
        )

    tilt_m, tilt_e = (40, 0) * apu.deg

    bs_gains = _gfunc(
        azims[np.newaxis], elevs[:, np.newaxis],
        G0, phi_3db, theta_3db,
        k_p, k_h, k_v,
        tilt_m=tilt_m, tilt_e=tilt_e,
        )

    assert_quantity_allclose(
        bs_gains,
        [
            [-0.16502266, 0.62075256, 1.81786011, 0.83655421, -0.17431214],
            [-5.82861701, -0.21702966, 17.955959, 0.77617614, -4.97296229],
            [-6.45692316, 3.77560389, 1.80402464, 0.28287166, -6.45692316],
            [-6.45692316, -4.88864499, -0.44864244, -4.34394808, -6.45692316],
            [-6.45692316, -6.45692316, -6.45692316, -6.45692316, -6.45692316]
            ] * cnv.dB
        )

    tilt_m, tilt_e = (0, 40) * apu.deg

    bs_gains = _gfunc(
        azims[np.newaxis], elevs[:, np.newaxis],
        G0, phi_3db, theta_3db,
        k_p, k_h, k_v,
        tilt_m=tilt_m, tilt_e=tilt_e,
        )

    assert_quantity_allclose(
        bs_gains,
        [
            [-6.45692316, -3.66473951, -0.04057318, -3.00566133, -6.45692316],
            [-6.45692316, 4.15502284, 17.92899408, 6.65990894, -6.45692316],
            [-6.45692316, -2.45771699, 2.7331265, -1.5137284, -6.45692316],
            [-6.45692316, -3.30718867, 0.78106736, -2.56371277, -6.45692316],
            [-6.45692316, -3.85975778, -0.48871902, -3.24671249, -6.45692316]
            ] * cnv.dB
        )




def test_imt_advanced_sectoral_avg_sidelobe_pattern_400_to_6000_mhz():

    _gfunc = imt.imt_advanced_sectoral_avg_sidelobe_pattern_400_to_6000_mhz
    args_list = [
        (-180, 180, apu.deg),
        (-90, 90, apu.deg),
        (None, None, cnv.dB),
        (0, None, apu.deg),
        (0, None, apu.deg),
        (0, None, cnv.dimless),
        (0, None, cnv.dimless),
        (0, None, cnv.dimless),
        (-90, 90, apu.deg),
        (-90, 90, apu.deg),
        ]
    check_astro_quantities(_gfunc, args_list)

    azims = np.arange(-180, 180, 0.5)[50:-50:150] * apu.deg
    elevs = np.arange(-90, 90, 0.5)[30:-30:70] * apu.deg

    G0 = 18. * cnv.dB
    phi_3db = 65. * apu.deg
    # theta_3db can be inferred in the following way:
    theta_3db = 31000 / G0.to(cnv.dimless) / phi_3db.value * apu.deg
    k_a, k_h, k_v = (0.7, 0.7, 0.3) * cnv.dimless
    tilt_m, tilt_e = (0, 0) * apu.deg

    bs_gain = _gfunc(
        0 * apu.deg, 0 * apu.deg,
        G0, phi_3db, theta_3db,
        k_a, k_h, k_v,
        tilt_m=tilt_m, tilt_e=tilt_e,
        )

    assert_quantity_allclose(bs_gain, G0)

    bs_gains = _gfunc(
        azims[np.newaxis], elevs[:, np.newaxis],
        G0, phi_3db, theta_3db,
        k_a, k_h, k_v,
        tilt_m=tilt_m, tilt_e=tilt_e,
        )
    assert_quantity_allclose(
        bs_gains,
        [
            [-9.45692316, -8.73264105, -7.99973768, -8.59935781, -9.45692316],
            [-9.45692316, -6.23545959, -2.9756504 , -5.64264212, -9.45692316],
            [-9.45692316,  1.55191039, 12.6917839 ,  3.57776872, -9.45692316],
            [-9.45692316, -5.11617053, -0.72374921, -4.3173802 , -9.45692316],
            [-9.45692316, -8.16416531, -6.85601954, -7.9262705 , -9.45692316],
            ] * cnv.dB
        )

    tilt_m, tilt_e = (40, 0) * apu.deg

    bs_gains = _gfunc(
        azims[np.newaxis], elevs[:, np.newaxis],
        G0, phi_3db, theta_3db,
        k_a, k_h, k_v,
        tilt_m=tilt_m, tilt_e=tilt_e,
        )

    assert_quantity_allclose(
        bs_gains,
        [
            [-6.65167057, -4.21643061, -1.90092642, -3.78461285, -6.46818041],
            [-9.04857002, -2.88936373, 17.95595854, -1.93430988, -8.57901892],
            [-9.45692316,  3.69272869, -1.87877821, -2.04640382, -9.45692316],
            [-9.45692316, -7.09347806, -7.4361134 , -6.90931547, -9.45692316],
            [-9.45692316, -9.45692316, -9.45692316, -9.45692316, -9.45692316],
            ] * cnv.dB
        )

    tilt_m, tilt_e = (0, 40) * apu.deg

    bs_gains = _gfunc(
        azims[np.newaxis], elevs[:, np.newaxis],
        G0, phi_3db, theta_3db,
        k_a, k_h, k_v,
        tilt_m=tilt_m, tilt_e=tilt_e,
        )

    assert_quantity_allclose(
        bs_gains,
        [
            [-9.45692316, -8.04001291, -6.60623694, -7.77927144, -9.45692316],
            [-9.45692316,  4.15502284, 17.92899408,  6.65990894, -9.45692316],
            [-9.45692316, -4.88763097, -0.26394973, -4.04678451, -9.45692316],
            [-9.45692316, -6.99775686, -4.5093187 , -6.54521822, -9.45692316],
            [-9.45692316, -8.60848865, -7.74995508, -8.45235874, -9.45692316],
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
    G_max = [20., 40.] * cnv.dBi

    # Note, if G_max is too small for the given diameter, wrong results
    # will be returned!
    gain = fixedlink.fl_pattern(
        phi[:, np.newaxis], diam, 0.21 * apu.m, G_max
        )
    assert_quantity_allclose(
        gain,
        [
            [19.94331066, 34.33106576],
            [19.84225854, 29.66663739],
            [19.56107501, 24.11108184],
            [18.77866514, 18.55552628],
            [16.60156321, 12.99997073],
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
