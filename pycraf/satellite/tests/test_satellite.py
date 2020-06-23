#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from astropy.tests.helper import assert_quantity_allclose, remote_data
from astropy import units as apu
from astropy.units import Quantity
from astropy.coordinates import EarthLocation
from astropy import time
from ... import conversions as cnv
from ...utils import check_astro_quantities
from .. import satellite
# from astropy.utils.misc import NumpyRNGContext


TLE = '''ISS (ZARYA)
1 25544U 98067A   13165.59097222  .00004759  00000-0  88814-4 0    47
2 25544  51.6478 121.2152 0011003  68.5125 263.9959 15.50783143834295'''

TLE_ERR = '''ISS (ZARYA)
1 25544U 98067A   58005.59097222  .00004759  00000-0  88814-4 0    47
2 25544  51.6478 121.2152 0011003  68.5125 263.9959 15.50783143834295'''


class TestSatelliteObserver:

    def setup(self):

        self.location = EarthLocation(6.88375, 50.525, 366.)
        self.so = satellite.SatelliteObserver(self.location)

    def teardown(self):

        pass

    def test_azel_from_sat(self):

        mjd = 55123. + np.array([0., 0.123, 0.99, 50.3])
        obstime = time.Time(mjd, format='mjd')

        az, el, dist = self.so.azel_from_sat(TLE, obstime)

        assert_quantity_allclose(
            az,
            [-34.51703096, 28.39693587, -143.17010479, -6.97428002] * apu.deg,
            atol=1e-6 * apu.deg,
            )
        assert_quantity_allclose(
            el,
            [-30.79780592, -43.16852656, -21.46811918, -36.77340433] * apu.deg,
            atol=1e-6 * apu.deg,
            )
        assert_quantity_allclose(
            dist,
            [7363.93018303, 9388.29246437, 5675.03921204, 8361.28408463
             ] * apu.km,
            atol=1e-6 * apu.km,
            )

    def test_azel_from_sat_error(self):

        mjd = 55123. + np.array([0., 0.123, 0.99, 50.3])
        obstime = time.Time(mjd, format='mjd')

        with pytest.raises(ValueError):

            az, el, dist = self.so.azel_from_sat(TLE_ERR, obstime)
