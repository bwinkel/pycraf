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
from pycraf import conversions as cnv
from pycraf import pathprof
from pycraf.helpers import check_astro_quantities
from astropy.utils.misc import NumpyRNGContext
from geographiclib.geodesic import Geodesic


TOL_KWARGS = {'atol': 1.e-4, 'rtol': 1.e-4}


class TestGeodesics:

    def setup(self):

        pass

    def teardown(self):

        pass

    def test_inverse(self):

        # testing against geographic-lib

        with NumpyRNGContext(1):

            lon1 = np.random.uniform(0, 360, 50)
            lon2 = np.random.uniform(0, 360, 50)
            lat1 = np.random.uniform(-90, 90, 50)
            lat2 = np.random.uniform(-90, 90, 50)

        distance = np.empty_like(lon1)
        bearing1 = np.empty_like(lon1)
        bearing2 = np.empty_like(lon1)
        distance_gglib = np.empty_like(lon1)
        bearing1_gglib = np.empty_like(lon1)
        bearing2_gglib = np.empty_like(lon1)
        distance_lowprec = np.empty_like(lon1)
        bearing1_lowprec = np.empty_like(lon1)
        bearing2_lowprec = np.empty_like(lon1)

        for idx, (_lon1, _lon2, _lat1, _lat2) in enumerate(zip(
                lon1, lat1, lon2, lat2
                )):

            (
                distance[idx], bearing1[idx], bearing2[idx]
                ) = pathprof.geodesics.inverse(
                _lon1, _lon2, _lat1, _lat2
                )
            (
                distance_lowprec[idx], bearing1_lowprec[idx],
                bearing2_lowprec[idx]
                ) = pathprof.geodesics.inverse(
                _lon1, _lon2, _lat1, _lat2, eps=1.e-8
                )

            aux = Geodesic.WGS84.Inverse(_lon2, _lon1, _lat2, _lat1)
            distance_gglib[idx] = aux['s12']
            bearing1_gglib[idx] = aux['azi1']
            bearing2_gglib[idx] = aux['azi2']

        assert_quantity_allclose(
            distance,
            distance_gglib,
            # atol=1.e-10, rtol=1.e-4
            )

        assert_quantity_allclose(
            distance_lowprec,
            distance_gglib,
            atol=1.,
            )

        assert_quantity_allclose(
            bearing1,
            bearing1_gglib,
            # atol=1.e-10, rtol=1.e-4
            )

        assert_quantity_allclose(
            bearing1_lowprec,
            bearing1_gglib,
            atol=1.e-6,
            )

        assert_quantity_allclose(
            bearing2,
            bearing2_gglib,
            # atol=1.e-10, rtol=1.e-4
            )

        assert_quantity_allclose(
            bearing2_lowprec,
            bearing2_gglib,
            atol=1.e-6,
            )

    def test_direct(self):

        # testing against geographic-lib

        with NumpyRNGContext(1):

            lon1 = np.random.uniform(0, 360, 50)
            lat1 = np.random.uniform(-90, 90, 50)
            bearing1 = np.random.uniform(-90, 90, 50)
            dist = np.random.uniform(1, 10.e6, 50)  # 10000 km max

        lon2 = np.empty_like(lon1)
        lat2 = np.empty_like(lon1)
        bearing2 = np.empty_like(lon1)
        lon2_gglib = np.empty_like(lon1)
        lat2_gglib = np.empty_like(lon1)
        bearing2_gglib = np.empty_like(lon1)
        lon2_lowprec = np.empty_like(lon1)
        lat2_lowprec = np.empty_like(lon1)
        bearing2_lowprec = np.empty_like(lon1)

        for idx, (_lon1, _lat1, _bearing1, _dist) in enumerate(zip(
                lon1, lat1, bearing1, dist
                )):

            (
                lon2[idx], lat2[idx], bearing2[idx]
                ) = pathprof.geodesics.direct(
                _lon1, _lat1, _bearing1, _dist
                )
            (
                lon2_lowprec[idx], lat2_lowprec[idx], bearing2_lowprec[idx]
                ) = pathprof.geodesics.direct(
                _lon1, _lat1, _bearing1, _dist, eps=1.e-8
                )

            line = Geodesic.WGS84.Line(_lat1, _lon1, _bearing1)
            pos = line.Position(_dist)
            lon2_gglib[idx] = pos['lon2']
            lat2_gglib[idx] = pos['lat2']
            bearing2_gglib[idx] = pos['azi2']

        assert_quantity_allclose(
            lon2,
            lon2_gglib,
            # atol=1.e-10, rtol=1.e-4
            )

        assert_quantity_allclose(
            lon2_lowprec,
            lon2_gglib,
            atol=1.e-6,
            )

        assert_quantity_allclose(
            lat2,
            lat2_gglib,
            # atol=1.e-10, rtol=1.e-4
            )

        assert_quantity_allclose(
            lat2_lowprec,
            lat2_gglib,
            atol=1.e-6,
            )

        assert_quantity_allclose(
            bearing2,
            bearing2_gglib,
            # atol=1.e-10, rtol=1.e-4
            )

        assert_quantity_allclose(
            bearing2_lowprec,
            bearing2_gglib,
            atol=1.e-6,
            )
