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
from astropy.tests.helper import assert_quantity_allclose
from astropy import units as apu
from astropy.units import Quantity
from ... import conversions as cnv
from ... import pathprof
from ...utils import check_astro_quantities
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.misc import NumpyRNGContext


TOL_KWARGS = {'atol': 1.e-4, 'rtol': 1.e-4}


def test_inverse_scalar():

    distance, bearing, bbearing = pathprof.cygeodesics.inverse_cython(
        0.1, 0.2, 0.11, 0.19
        )
    assert_allclose(distance, 89067.84192818255)
    assert_allclose(bearing, 2.3615536070091063)
    assert_allclose(bbearing, 2.3634913123259564)


def test_direct_scalar():

    lon2, lat2, bbearing = pathprof.cygeodesics.direct_cython(
        0.1, 0.2, 2.3615536070091063, 89067.84192818255
        )
    assert_allclose(lon2, 0.11)
    assert_allclose(lat2, 0.19)
    assert_allclose(bbearing, 2.3634913123259564)


def test_inverse_broadcast():

    lon1_rad = np.array([
        1.50127922e+02, 2.59316818e+02, 4.11749342e-02, 1.08839726e+02
        ])
    lat1_rad = np.array([
        26.41606035, 16.62094706, 33.52683805, 62.20093087
        ])
    lon2_rad = np.array([
        142.83629072, 193.97402424, 150.91002519, 246.67902014
        ])
    lat2_rad = np.array([
        36.80140495, 158.06113855, 4.92976678, 120.68415183
        ])

    distance, bearing, bbearing = pathprof.cygeodesics.inverse_cython(
        lon1_rad[np.newaxis], lat1_rad[np.newaxis],
        lon2_rad[:, np.newaxis], lat2_rad[:, np.newaxis]
        )
    assert_allclose(distance, np.array([
        [14534170.095, 19059672.900, 5737251.274, 9808826.355],
        [1959064.126, 7525083.635, 13485937.244, 17402290.615],
        [17017579.233, 15638886.761, 2068263.906, 6555134.330],
        [3234810.680, 2501682.415, 16226044.972, 12405292.273],
        ]))
    assert_allclose(bearing, np.array([
        [-2.36707497, 1.55935925, -2.23312062, 2.80342054],
        [-2.8887663, -0.36268785, -0.51154311, -0.43993657],
        [2.79661057, -2.79244412, 3.09245435, -2.9013161],
        [0.41726618, -0.0494581, 0.49412473, -0.10893228],
        ]), atol=1.e-6)
    assert_allclose(bbearing, np.array([
        [-2.8175359, 1.76754203, -0.70856419, 0.4437692],
        [-3.01353187, -2.74073623, -0.470134, -2.47532756],
        [2.68092885, -1.82156649, 3.02429343, -1.09552795],
        [2.69129243, -0.11462584, 1.17631058, -0.33829475],
        ]), atol=1.e-6)


def test_direct_broadcast():

    lon1_rad = np.array([
        1.50127922e+02, 2.59316818e+02, 4.11749342e-02, 1.08839726e+02
        ])
    lat1_rad = np.array([
        26.41606035, 16.62094706, 33.52683805, 62.20093087
        ])
    bearing = np.array([
        [-2.36707497, 1.55935925, -2.23312062, 2.80342054],
        [-2.8887663, -0.36268785, -0.51154311, -0.43993657],
        [2.79661057, -2.79244412, 3.09245435, -2.9013161],
        [0.41726618, -0.0494581, 0.49412473, -0.10893228],
        ])
    distance = np.array([
        [14534170.095, 19059672.900, 5737251.274, 9808826.355],
        [1959064.126, 7525083.635, 13485937.244, 17402290.615],
        [17017579.233, 15638886.761, 2068263.906, 6555134.330],
        [3234810.680, 2501682.415, 16226044.972, 12405292.273],
        ])

    lon2, lat2, bbearing = pathprof.cygeodesics.direct_cython(
        lon1_rad[np.newaxis], lat1_rad[np.newaxis],
        bearing, distance,
        )
    assert_allclose(lon2, np.array([
        [-1.67697103, -1.67697098, -1.67697134, -1.6769715],
        [-0.80471997, -0.80471992, -0.80472029, -0.80472043],
        [0.1135781, 0.11357818, 0.11357781, 0.11357767],
        [1.63479349, 1.63479352, 1.63479317, 1.63479301],
        ]), atol=1.e-6)
    assert_allclose(lat2, np.array([
        [-0.89770689, -0.89770689, -0.89770689, -0.89770689],
        [0.98150587, 0.98150587, 0.98150587, 0.98150587],
        [-1.35341853, -1.35341853, -1.35341853, -1.35341853],
        [1.30363099, 1.303631, 1.30363099, 1.303631],
        ]), atol=1.e-6)
    assert_allclose(bbearing, np.array([
        [-2.8175359, 1.76754203, -0.70856419, 0.4437692],
        [-3.01353187, -2.74073623, -0.470134, -2.47532756],
        [2.68092885, -1.82156649, 3.02429343, -1.09552795],
        [2.69129243, -0.11462584, 1.17631058, -0.33829475],
        ]), atol=1.e-6)


def test_inverse_vs_geographicslib():

    # testing against geographic-lib
    args_list = [
        (-np.pi, np.pi, apu.rad),
        (-np.pi / 2, np.pi / 2, apu.rad),
        (-np.pi, np.pi, apu.rad),
        (-np.pi / 2, np.pi / 2, apu.rad),
        ]

    check_astro_quantities(pathprof.geoid_inverse, args_list)

    with NumpyRNGContext(1):

        lon1 = np.random.uniform(0, 360, 50)
        lon2 = np.random.uniform(0, 360, 50)
        lat1 = np.random.uniform(-90, 90, 50)
        lat2 = np.random.uniform(-90, 90, 50)

    lon1 = (lon1 + 180) % 360 - 180
    lon2 = (lon2 + 180) % 360 - 180

    distance, bearing1, bearing2 = pathprof.geoid_inverse(
        lon1 * apu.deg, lat1 * apu.deg,
        lon2 * apu.deg, lat2 * apu.deg,
        )
    (
        distance_lowprec, bearing1_lowprec, bearing2_lowprec
        ) = pathprof.geoid_inverse(
        lon1 * apu.deg, lat1 * apu.deg,
        lon2 * apu.deg, lat2 * apu.deg,
        eps=1.e-8
        )

    def produce_geographicslib_results():

        from geographiclib.geodesic import Geodesic

        distance_gglib = np.empty_like(lon1)
        bearing1_gglib = np.empty_like(lon1)
        bearing2_gglib = np.empty_like(lon1)

        for idx, (_lon1, _lat1, _lon2, _lat2) in enumerate(zip(
                lon1, lat1, lon2, lat2
                )):

            aux = Geodesic.WGS84.Inverse(_lat1, _lon1, _lat2, _lon2)
            distance_gglib[idx] = aux['s12']
            bearing1_gglib[idx] = aux['azi1']
            bearing2_gglib[idx] = aux['azi2']

        # move manually to testcases, if desired
        np.savez(
            '/tmp/gglib_inverse.npz',
            distance=distance_gglib,
            bearing1=bearing1_gglib, bearing2=bearing2_gglib,
            )

    # produce_geographicslib_results()

    gglib_inverse_name = get_pkg_data_filename('geolib/gglib_inverse.npz')
    gglib = np.load(gglib_inverse_name)

    assert_quantity_allclose(
        distance.to(apu.m).value,
        gglib['distance'],
        # atol=1.e-10, rtol=1.e-4
        )

    assert_quantity_allclose(
        distance_lowprec.to(apu.m).value,
        gglib['distance'],
        atol=1.,
        )

    assert_quantity_allclose(
        bearing1.to(apu.deg).value,
        gglib['bearing1'],
        # atol=1.e-10, rtol=1.e-4
        )

    assert_quantity_allclose(
        bearing1_lowprec.to(apu.deg).value,
        gglib['bearing1'],
        atol=1.e-6,
        )

    assert_quantity_allclose(
        bearing2.to(apu.deg).value,
        gglib['bearing2'],
        # atol=1.e-10, rtol=1.e-4
        )

    assert_quantity_allclose(
        bearing2_lowprec.to(apu.deg).value,
        gglib['bearing2'],
        atol=1.e-6,
        )


def test_direct():

    # testing against geographic-lib
    args_list = [
        (-np.pi, np.pi, apu.rad),
        (-np.pi / 2, np.pi / 2, apu.rad),
        (-np.pi, np.pi, apu.rad),
        (0.1, None, apu.m),
        ]

    check_astro_quantities(pathprof.geoid_direct, args_list)

    with NumpyRNGContext(1):

        lon1 = np.random.uniform(0, 360, 50)
        lat1 = np.random.uniform(-90, 90, 50)
        bearing1 = np.random.uniform(-90, 90, 50)
        dist = np.random.uniform(1, 10.e6, 50)  # 10000 km max

    lon1 = (lon1 + 180) % 360 - 180

    lon2, lat2, bearing2 = pathprof.geoid_direct(
        lon1 * apu.deg, lat1 * apu.deg,
        bearing1 * apu.deg, dist * apu.m
        )
    (
        lon2_lowprec, lat2_lowprec, bearing2_lowprec
        ) = pathprof.geoid_direct(
        lon1 * apu.deg, lat1 * apu.deg,
        bearing1 * apu.deg, dist * apu.m,
        eps=1.e-8
        )

    def produce_geographicslib_results():

        from geographiclib.geodesic import Geodesic

        lon2_gglib = np.empty_like(lon1)
        lat2_gglib = np.empty_like(lon1)
        bearing2_gglib = np.empty_like(lon1)

        for idx, (_lon1, _lat1, _bearing1, _dist) in enumerate(zip(
                lon1, lat1, bearing1, dist
                )):

            line = Geodesic.WGS84.Line(_lat1, _lon1, _bearing1)
            pos = line.Position(_dist)
            lon2_gglib[idx] = pos['lon2']
            lat2_gglib[idx] = pos['lat2']
            bearing2_gglib[idx] = pos['azi2']

        # move manually to testcases, if desired
        np.savez(
            '/tmp/gglib_direct.npz',
            bearing2=bearing2_gglib,
            lon2=lon2_gglib, lat2=lat2_gglib,
            )

    # produce_geographicslib_results()
    gglib_direct_name = get_pkg_data_filename('geolib/gglib_direct.npz')
    gglib = np.load(gglib_direct_name)

    assert_quantity_allclose(
        lon2.to(apu.deg).value,
        gglib['lon2'],
        # atol=1.e-10, rtol=1.e-4
        )

    assert_quantity_allclose(
        lon2_lowprec.to(apu.deg).value,
        gglib['lon2'],
        atol=1.e-6,
        )

    assert_quantity_allclose(
        lat2.to(apu.deg).value,
        gglib['lat2'],
        # atol=1.e-10, rtol=1.e-4
        )

    assert_quantity_allclose(
        lat2_lowprec.to(apu.deg).value,
        gglib['lat2'],
        atol=1.e-6,
        )

    assert_quantity_allclose(
        bearing2.to(apu.deg).value,
        gglib['bearing2'],
        # atol=1.e-10, rtol=1.e-4
        )

    assert_quantity_allclose(
        bearing2_lowprec.to(apu.deg).value,
        gglib['bearing2'],
        atol=1.e-6,
        )


def test_geoid_area():

    # testing against geographic-lib
    args_list = [
        (-np.pi, np.pi, apu.rad),
        (-np.pi, np.pi, apu.rad),
        (-np.pi / 2, np.pi / 2, apu.rad),
        (-np.pi / 2, np.pi / 2, apu.rad),
        ]

    check_astro_quantities(pathprof.geoid_area, args_list)

    lon1 = np.array([-0.5, 6, 7, -180, 10])
    lon2 = np.array([0.5, 7, 6, 180, 40])
    lat1 = np.array([-0.5, 50, 50, -90, 30])
    lat2 = np.array([0.5, 51, 51, 90, 60])

    area = pathprof.geoid_area(
        lon1 * apu.deg, lon2 * apu.deg,
        lat1 * apu.deg, lat2 * apu.deg,
        )

    assert_quantity_allclose(
        area.to(apu.km ** 2).value,
        np.array([
            1.23918686e+04, 7.83488948e+03, -7.83488948e+03,
            5.09490053e+08, 7.75889348e+06
            ])
        )

    lons = np.linspace(0, 180, 6)
    lats = np.linspace(0, 90, 6)

    area = pathprof.geoid_area(
        lons[1:] * apu.deg, lons[:-1] * apu.deg,
        lats[1:, np.newaxis] * apu.deg, lats[:-1, np.newaxis] * apu.deg,
        )

    assert_quantity_allclose(
        area.to(apu.km ** 2).value,
        np.array([
            [7896050.19052842, 7896050.19052842, 7896050.19052842,
             7896050.19052842, 7896050.19052842],
            [7110472.64473981, 7110472.64473981, 7110472.64473981,
             7110472.64473981, 7110472.64473981],
            [5626711.01884766, 5626711.01884766, 5626711.01884766,
             5626711.01884766, 5626711.01884766],
            [3602222.87978228, 3602222.87978228, 3602222.87978228,
             3602222.87978228, 3602222.87978228],
            [1239045.92382274, 1239045.92382274, 1239045.92382274,
             1239045.92382274, 1239045.92382274]
            ])
        )
