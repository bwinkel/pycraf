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


class TestGeodesics:

    def setup(self):

        pass

    def teardown(self):

        pass

    def test_inverse(self):

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
            distance.to_value(apu.m),
            gglib['distance'],
            # atol=1.e-10, rtol=1.e-4
            )

        assert_quantity_allclose(
            distance_lowprec.to_value(apu.m),
            gglib['distance'],
            atol=1.,
            )

        assert_quantity_allclose(
            bearing1.to_value(apu.deg),
            gglib['bearing1'],
            # atol=1.e-10, rtol=1.e-4
            )

        assert_quantity_allclose(
            bearing1_lowprec.to_value(apu.deg),
            gglib['bearing1'],
            atol=1.e-6,
            )

        assert_quantity_allclose(
            bearing2.to_value(apu.deg),
            gglib['bearing2'],
            # atol=1.e-10, rtol=1.e-4
            )

        assert_quantity_allclose(
            bearing2_lowprec.to_value(apu.deg),
            gglib['bearing2'],
            atol=1.e-6,
            )

    def test_direct(self):

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
            lon2.to_value(apu.deg),
            gglib['lon2'],
            # atol=1.e-10, rtol=1.e-4
            )

        assert_quantity_allclose(
            lon2_lowprec.to_value(apu.deg),
            gglib['lon2'],
            atol=1.e-6,
            )

        assert_quantity_allclose(
            lat2.to_value(apu.deg),
            gglib['lat2'],
            # atol=1.e-10, rtol=1.e-4
            )

        assert_quantity_allclose(
            lat2_lowprec.to_value(apu.deg),
            gglib['lat2'],
            atol=1.e-6,
            )

        assert_quantity_allclose(
            bearing2.to_value(apu.deg),
            gglib['bearing2'],
            # atol=1.e-10, rtol=1.e-4
            )

        assert_quantity_allclose(
            bearing2_lowprec.to_value(apu.deg),
            gglib['bearing2'],
            atol=1.e-6,
            )

    def test_geoid_area(self):

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
            area.to_value(apu.km ** 2),
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
            area.to_value(apu.km ** 2),
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
