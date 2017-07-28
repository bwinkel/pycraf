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


TOL_KWARGS = {'atol': 1.e-4, 'rtol': 1.e-4}


class TestHelper:

    def setup(self):

        pass

    def teardown(self):

        pass

    def test_annual_timepercent_from_worst_month(self):

        pfunc = pathprof.annual_timepercent_from_worst_month
        args_list = [
            (0, 100, apu.percent),
            (-90, 90, apu.deg),
            (0, 100, apu.percent),
            ]
        check_astro_quantities(pfunc, args_list)

        p_w = Quantity([0.01, 10., 20., 20., 50., 50.], apu.percent)
        phi = Quantity([30., 50., -50., 50, 40., 40.], apu.deg)
        omega = Quantity([0., 0., 10., 10., 0., 100.], apu.percent)

        p = Quantity(
            [
                1.407780e-03, 4.208322e+00, 9.141926e+00,
                9.141926e+00, 4.229364e+01, 1.889433e+01
                ], apu.percent
            )
        assert_quantity_allclose(
            pfunc(p_w, phi, omega),
            p,
            **TOL_KWARGS
            )

    def test_deltaN_N0_from_map(self):

        pfunc = pathprof.deltaN_N0_from_map
        args_list = [
            (-180, 360, apu.deg),
            (-90, 90, apu.deg),
            ]
        check_astro_quantities(pfunc, args_list)

        lon = Quantity([0.01, 10., 20., 20., 50., 359.], apu.deg)
        lat = Quantity([30., 50., -50., 50, 40., 40.], apu.deg)

        DN, N0 = pfunc(lon, lat)

        assert_quantity_allclose(
            DN,
            Quantity([
                33.82150749, 36.5930002, 44.36488936, 35.57166629,
                40.73833338, 44.74622175], cnv.dimless / apu.km),
            )

        assert_quantity_allclose(
            N0,
            Quantity([
                304.58966064, 323.69878472, 319.52155219, 322.11834378,
                326.14456177, 329.69577705], cnv.dimless),
            )

    def test_radiomet_data_for_pathcenter(self):

        pfunc = pathprof.radiomet_data_for_pathcenter
        args_list = [
            (-180, 360, apu.deg),
            (-90, 90, apu.deg),
            (0, None, apu.km),
            (0, None, apu.km),
            ]
        check_astro_quantities(pfunc, args_list)

        lon = Quantity([0.01, 10., 20., 20., 50., 359.], apu.deg)
        lat = Quantity([30., 50., -50., 50, 40., 40.], apu.deg)
        d_tm = Quantity([0.1, 1., 10., 10., 0., 100.], apu.km)
        d_lm = Quantity([0.1, 1., 10., 10., 0., 100.], apu.km)

        DN, beta_0, N0 = pfunc(lon, lat, d_tm, d_lm)

        assert_quantity_allclose(
            DN,
            Quantity([
                33.82150749, 36.5930002, 44.36488936, 35.57166629,
                40.73833338, 44.74622175], cnv.dimless / apu.km),
            )

        assert_quantity_allclose(
            beta_0,
            Quantity([
                16.57415849, 8.10024667, 6.27440928, 6.27440928,
                11.74897555, 2.60825487], apu.percent),
            )

        assert_quantity_allclose(
            N0,
            Quantity([
                304.58966064, 323.69878472, 319.52155219, 322.11834378,
                326.14456177, 329.69577705], cnv.dimless),
            )

    def test_eff_earth_radius_median(self):

        pfunc = pathprof.eff_earth_radius_median
        args_list = [
            (-180, 360, apu.deg),
            (-90, 90, apu.deg),
            ]
        check_astro_quantities(pfunc, args_list)

        lon = Quantity([0.01, 10., 20., 20., 50., 359.], apu.deg)
        lat = Quantity([30., 50., -50., 50, 40., 40.], apu.deg)

        a_e = pfunc(lon, lat)

        assert_quantity_allclose(
            a_e,
            Quantity([
                8120.30557961, 8307.21637166, 8880.41920755, 8237.34436165,
                8603.41184774, 8910.586491], apu.km),
            )

    def test_eff_earth_radius_factor_median(self):

        pfunc = pathprof.eff_earth_radius_factor_median
        args_list = [
            (-180, 360, apu.deg),
            (-90, 90, apu.deg),
            ]
        check_astro_quantities(pfunc, args_list)

        lon = Quantity([0.01, 10., 20., 20., 50., 359.], apu.deg)
        lat = Quantity([30., 50., -50., 50, 40., 40.], apu.deg)

        k50 = pfunc(lon, lat)

        assert_quantity_allclose(
            k50,
            Quantity([
                1.27457316, 1.3039109, 1.39388153, 1.29294371, 1.35040211,
                1.39861662], cnv.dimless),
            )

    def test_eff_earth_radius_factor_beta(self):

        assert pathprof.eff_earth_radius_factor_beta() == 3 * cnv.dimless

    def test_eff_earth_radius_beta(self):

        assert pathprof.eff_earth_radius_beta() == (
            3 * 6371. * apu.km
            )
