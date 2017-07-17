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
from astropy.units import Quantity
from astropy.utils.misc import NumpyRNGContext
from ... import conversions as cnv
from ... import geospatial as gsp
import pyproj


class TestTransformations:

    def setup(self):

        # with NumpyRNGContext(1):
        #     lon = np.random.uniform(-180., -174., 10)
        #     lat = np.random.uniform(0., 84., 10)

        self.glon = np.array([
            -177.49786797, -175.67805304, -179.99931375, -178.18600456,
            -179.11946466, -179.44596843, -178.88243873, -177.92663564,
            -177.61939515, -176.7670996
            ]) * apu.deg
        self.glat = np.array([
            35.21233921, 57.55843803, 17.17398898, 73.76186466,
            2.30055783, 56.31927085, 35.05360340, 46.92994559,
            11.79250284, 16.64052508
            ]) * apu.deg

        self.ulon = np.array([
            454686.20628782, 579095.79975176, 180918.24020919,
            462984.30710277, 264291.51548385, 348727.96653084,
            328325.84860651, 429460.16728274, 432520.86377695,
            524837.50505706
            ]) * apu.m
        self.ulat = np.array([
            3896704.61049107, 6380321.53548225, 1901270.36631222,
            8185838.99227284, 254457.1973956, 6244302.06697137,
            3880607.58384058, 5197795.98216574, 1303683.71865443,
            1839803.27907012
            ]) * apu.m

        # with NumpyRNGContext(1):
        #     lon = np.random.uniform(0., 20., 10)
        #     lat = np.random.uniform(42., 62., 10)

        self.glon2 = np.array([
            8.34044009e+00, 1.44064899e+01, 2.28749635e-03,
            6.04665145e+00, 2.93511782e+00, 1.84677190e+00,
            3.72520423e+00, 6.91121454e+00, 7.93534948e+00,
            1.07763347e+01
            ]) * apu.deg
        self.glat2 = np.array([
            50.38389029, 55.70439001, 46.08904499, 59.56234873,
            42.54775186, 55.4093502, 50.34609605, 53.17379657,
            44.80773877, 45.96202978
            ]) * apu.deg
        self.elon = np.array([
            4202970.22261754, 4597990.61594984, 3549505.95735659,
            4097111.89062938, 3739808.93619669, 3805588.0880451,
            3874957.43154435, 4114517.95320501, 4157389.23251153,
            4381254.0325178
            ]) * apu.m
        self.elat = np.array([
            3031544.07799421, 3630702.75390756, 2604858.91209381,
            4057530.73354106, 2187517.43947754, 3618600.91341798,
            3045111.91418351, 3345007.83946865, 2412965.16210406,
            2539108.27478651
            ]) * apu.m

    def teardown(self):

        pass

    def test_create_proj(self):

        _create_proj = gsp.geospatial._create_proj

        with pytest.raises(AssertionError):
            _create_proj(None, None)

        with pytest.raises(TypeError):
            _create_proj('UTM', 'WGS84')

        with pytest.raises(TypeError):
            _create_proj('UTM', 'WGS84', '32')

        for args, kwargs in [
                (('UTM', 'WGS84'), {'zone': 2}),
                (('UTM', 'WGS84'), {'zone': 32}),
                (('UTM', 'WGS84'), {'zone': 32, 'south': False}),
                (('ETRS89', 'WGS84'), {}),
                ]:

            proj = _create_proj(*args, **kwargs)
            assert isinstance(proj, pyproj.Proj)

    def test_wgs84_to_utm(self):

        zones = [1, 32, 60]

        for zone in zones:

            glon = self.glon + (zone - 1) * 6. * apu.deg
            ulon, ulat = gsp.wgs84_to_utm(glon, self.glat, zone)

            assert_allclose(ulon, self.ulon)
            assert_allclose(ulat, self.ulat)

    def test_utm_to_wgs84(self):

        zones = [1, 32, 60]

        for zone in zones:

            glon, glat = gsp.utm_to_wgs84(self.ulon, self.ulat, zone)
            glon -= (zone - 1) * 6. * apu.deg

            assert_allclose(glon, self.glon)
            assert_allclose(glat, self.glat)

    def test_wgs84_to_etrs89(self):

        elon, elat = gsp.wgs84_to_etrs89(self.glon2, self.glat2)

        assert_allclose(elon, self.elon)
        assert_allclose(elat, self.elat)

    def test_etrs89_to_wgs84(self):

        glon, glat = gsp.etrs89_to_wgs84(self.elon, self.elat)

        assert_allclose(glon, self.glon2)
        assert_allclose(glat, self.glat2)
