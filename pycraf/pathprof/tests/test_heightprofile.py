#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

import pytest
from functools import partial
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from astropy.tests.helper import assert_quantity_allclose, remote_data
from astropy import units as apu
from astropy.units import Quantity
from ... import pathprof
from ...utils import check_astro_quantities
from astropy.utils.misc import NumpyRNGContext


TOL_KWARGS = {'atol': 1.e-4, 'rtol': 1.e-4}


@remote_data(source='any')
@pytest.mark.usefixtures('srtm_handler')
def test_srtm_height_profile(srtm_temp_dir):

    lon_t, lat_t = 6.5 * apu.deg, 50.5 * apu.deg
    lon_r, lat_r = 6.52 * apu.deg, 50.52 * apu.deg

    # Geodesics are already tested, so only check heights
    (
        _, _, _, _, heights, _, _, _,
        ) = pathprof.srtm_height_profile(
        lon_t, lat_t, lon_r, lat_r, 100 * apu.m
        )

    assert_quantity_allclose(
        heights,
        np.array([
            500.30369658, 506.03255363, 510.84655991, 516.08284811,
            517.22129847, 515.50985828, 513.96561413, 509.34411013,
            506.24529689, 504.01725449, 499.69464749, 498.18230625,
            494.72718055, 492.61897135, 497.07525745, 500.37202712,
            502.26444265, 505.52286053, 511.76030441, 516.25117734,
            513.52680980, 513.64512396, 516.23836125, 515.85071130,
            513.37103590, 509.62782117, 504.46273853, 502.27201090,
            ]) * apu.m
        )

    lon_t, lat_t = 6.5 * apu.deg, 50.5 * apu.deg
    lon_r, lat_r = 6.502 * apu.deg, 50.502 * apu.deg

    (
        _, _, _, _, heights, _, _, _,
        ) = pathprof.srtm_height_profile(
        lon_t, lat_t, lon_r, lat_r, 10 * apu.m
        )

    assert_quantity_allclose(
        heights,
        np.array([
            498.00000000, 498.98388672, 499.93469238, 500.85241699,
            501.73706055, 502.58859253, 503.40704346, 504.19244385,
            504.94473267, 505.66394043, 506.35003662, 507.00308228,
            507.61468506, 508.17663574, 508.68896484, 509.15167236,
            509.56472778, 509.92816162, 510.24197388, 510.50613403,
            510.72067261, 510.88558960, 511.00558472, 511.56820679,
            512.16387939, 512.79266357, 513.45452881, 514.14953613,
            ]) * apu.m
        )


def test_srtm_height_profile_generic():

    lon_t, lat_t = 6.5 * apu.deg, 50.5 * apu.deg
    lon_r, lat_r = 6.52 * apu.deg, 50.52 * apu.deg

    # Geodesics are already tested, so only check heights
    (
        _, _, _, _, heights, _, _, _,
        ) = pathprof.srtm_height_profile(
        lon_t, lat_t, lon_r, lat_r, 100 * apu.m, generic_heights=True
        )

    assert_quantity_allclose(
        heights,
        np.array([
            0., 0., 0., 0.,
            0., 0., 0., 0.,
            0., 0., 0., 0.,
            0., 0., 0., 0.,
            0., 0., 0., 0.,
            0., 0., 0., 0.,
            0., 0., 0., 0.,
            ]) * apu.m
        )

    lon_t, lat_t = 6.5 * apu.deg, 50.5 * apu.deg
    lon_r, lat_r = 6.502 * apu.deg, 50.502 * apu.deg

    (
        _, _, _, _, heights, _, _, _,
        ) = pathprof.srtm_height_profile(
        lon_t, lat_t, lon_r, lat_r, 10 * apu.m, generic_heights=True
        )

    assert_quantity_allclose(
        heights,
        np.array([
            0., 0., 0., 0.,
            0., 0., 0., 0.,
            0., 0., 0., 0.,
            0., 0., 0., 0.,
            0., 0., 0., 0.,
            0., 0., 0., 0.,
            0., 0., 0., 0.,
            ]) * apu.m
        )


@remote_data(source='any')
@pytest.mark.usefixtures('srtm_handler')
def test_srtm_height_map():

    lon_t, lat_t = 6.5 * apu.deg, 50.5 * apu.deg
    ms_lon, ms_lat = 0.2 * apu.deg, 0.2 * apu.deg

    # Geodesics are already tested, so only check heights
    _, _, hmap = pathprof.srtm_height_map(lon_t, lat_t, ms_lon, ms_lat)

    assert_quantity_allclose(
        hmap[::70, ::70],
        np.array([
            [657.37585449, 611.11309814, 579.11462402, 471.04782104],
            [574.93627930, 600.39331055, 514.88537598, 533.50799561],
            [601.37585449, 525.39331055, 446.45861816, 482.01593018],
            [322.49667358, 467.42672729, 438.44268799, 335.00000000],
            ]) * apu.m
        )
