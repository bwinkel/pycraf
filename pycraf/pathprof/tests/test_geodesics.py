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
from astropy.utils.misc import NumpyRNGContext
from geographiclib.geodesic import Geodesic


TOL_KWARGS = {'atol': 1.e-4, 'rtol': 1.e-4}


class TestGeodesics:

    def setup(self):

        pass

    def teardown(self):

        pass

    def test_inverse_cython(self):

        # testing against geographic-lib

        with NumpyRNGContext(1):

            lon1 = np.random.uniform(0, 360, 50)
            lon2 = np.random.uniform(0, 360, 50)
            lat1 = np.random.uniform(-90, 90, 50)
            lat2 = np.random.uniform(-90, 90, 50)

        distance, bearing1, bearing2 = pathprof.geodesics.inverse_cython(
            np.radians(lon1), np.radians(lat1),
            np.radians(lon2), np.radians(lat2),
            )
        (
            distance_lowprec, bearing1_lowprec, bearing2_lowprec
            ) = pathprof.geodesics.inverse_cython(
            np.radians(lon1), np.radians(lat1),
            np.radians(lon2), np.radians(lat2),
            eps=1.e-8
            )

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
            np.degrees(bearing1),
            bearing1_gglib,
            # atol=1.e-10, rtol=1.e-4
            )

        assert_quantity_allclose(
            np.degrees(bearing1_lowprec),
            bearing1_gglib,
            atol=1.e-6,
            )

        assert_quantity_allclose(
            np.degrees(bearing2),
            bearing2_gglib,
            # atol=1.e-10, rtol=1.e-4
            )

        assert_quantity_allclose(
            np.degrees(bearing2_lowprec),
            bearing2_gglib,
            atol=1.e-6,
            )

    def test_direct_cython(self):

        # testing against geographic-lib

        with NumpyRNGContext(1):

            lon1 = np.random.uniform(0, 360, 50)
            lat1 = np.random.uniform(-90, 90, 50)
            bearing1 = np.random.uniform(-90, 90, 50)
            dist = np.random.uniform(1, 10.e6, 50)  # 10000 km max

        lon2, lat2, bearing2 = pathprof.cygeodesics.direct_cython(
            np.radians(lon1), np.radians(lat1),
            np.radians(bearing1), dist
            )
        (
            lon2_lowprec, lat2_lowprec, bearing2_lowprec
            ) = pathprof.cygeodesics.direct_cython(
            np.radians(lon1), np.radians(lat1),
            np.radians(bearing1), dist, eps=1.e-8
            )

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

        assert_quantity_allclose(
            np.degrees(lon2),
            lon2_gglib,
            # atol=1.e-10, rtol=1.e-4
            )

        assert_quantity_allclose(
            np.degrees(lon2_lowprec),
            lon2_gglib,
            atol=1.e-6,
            )

        assert_quantity_allclose(
            np.degrees(lat2),
            lat2_gglib,
            # atol=1.e-10, rtol=1.e-4
            )

        assert_quantity_allclose(
            np.degrees(lat2_lowprec),
            lat2_gglib,
            atol=1.e-6,
            )

        assert_quantity_allclose(
            np.degrees(bearing2),
            bearing2_gglib,
            # atol=1.e-10, rtol=1.e-4
            )

        assert_quantity_allclose(
            np.degrees(bearing2_lowprec),
            bearing2_gglib,
            atol=1.e-6,
            )
