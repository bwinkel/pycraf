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
from pycraf import conversions as cnv
from pycraf import pathprof
from pycraf.helpers import check_astro_quantities
# from astropy.utils.misc import NumpyRNGContext


TOL_KWARGS = {'atol': 1.e-4, 'rtol': 1.e-4}


class TestHelper:

    def setup(self):

        pass

    def teardown(self):

        pass

    def test_anual_time_percentage_from_worst_month(self):

        pfunc = pathprof.anual_time_percentage_from_worst_month
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

    def test_radiomet_data_for_pathcenter(self):

        pfunc = pathprof.radiomet_data_for_pathcenter
        args_list = [
            (0, 360, apu.deg),
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

        print(DN)
        print(beta_0)
        print(N0)

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

    def test_median_effective_earth_radius(self):

        pfunc = pathprof.median_effective_earth_radius
        args_list = [
            (0, 360, apu.deg),
            (-90, 90, apu.deg),
            ]
        check_astro_quantities(pfunc, args_list)

        lon = Quantity([0.01, 10., 20., 20., 50., 359.], apu.deg)
        lat = Quantity([30., 50., -50., 50, 40., 40.], apu.deg)

        a_e = pfunc(lon, lat)

        print(a_e)

        assert_quantity_allclose(
            a_e,
            Quantity([
                8120.30557961, 8307.21637166, 8880.41920755, 8237.34436165,
                8603.41184774, 8910.586491], apu.km),
            )


class TestPropagation:

    def setup(self):

        pass

    def teardown(self):

        pass

    def test_free_space_loss_bfsg(self):

        pfunc = pathprof.free_space_loss_bfsg
        args_list = [
            (1.e-30, None, apu.km),
            (1.e-30, None, apu.GHz),
            (0, 100, apu.percent),
            (0, None, apu.K),
            (0, None, apu.hPa),
            ]
        kwargs_list = [
            ('d_lt', 0, None, apu.m),
            ('d_lr', 0, None, apu.m),
            ('time_percent', 0, 100, apu.percent),
            ]
        check_astro_quantities(pfunc, args_list, kwargs_list)

        # first, test without corrections
        dist = Quantity([1., 2., 5., 10., 50., 50.], apu.km)
        freq = Quantity([1., 1., 0.5, 10., 2., 2.], apu.GHz)
        omega = Quantity([0., 0., 0., 0., 0., 50.], apu.percent)
        tamb = Quantity([270., 280., 280., 300., 300., 300.], apu.K)
        pressure = Quantity([1000., 1010., 990., 990., 1000., 1000.], apu.hPa)

        Lbfsg_annex1 = Quantity(
            [
                92.50615024, 98.53204435, 100.47441164, 132.62854741,
                132.7989964, 132.80053965
                ], cnv.dB
            )
        assert_quantity_allclose(
            pfunc(dist, freq, omega, tamb, pressure, atm_method='annex1'),
            Lbfsg_annex1,
            )

        Lbfsg_annex2 = Quantity(
            [
                92.50622914, 98.5321929, 100.47467167, 132.6272019,
                132.8020698, 132.80412104
                ], cnv.dB
            )
        assert_quantity_allclose(
            pfunc(dist, freq, omega, tamb, pressure, atm_method='annex2'),
            Lbfsg_annex2,
            )

        # test broadcasting
        dist = Quantity([1., 10., 20.], apu.km)
        freq = Quantity([1., 2., 22.], apu.GHz)
        omega = Quantity(0., apu.percent)
        tamb = Quantity(300., apu.K)
        pressure = Quantity(1013., apu.hPa)
        Lbfsg_annex1 = Quantity(
            [
                [92.5048683, 98.52672276, 119.53615245],
                [112.54868299, 118.58182837, 141.22544192],
                [118.61796588, 124.66365675, 149.12303013],
                ], cnv.dB
            )
        assert_quantity_allclose(
            pfunc(dist[:, np.newaxis], freq, omega, tamb, pressure),
            Lbfsg_annex1,
            )

    def test_tropospheric_scatter_loss_bs(self):

        pfunc = pathprof.tropospheric_scatter_loss_bs
        args_list = [
            (1.e-30, None, apu.km),
            (1.e-30, None, apu.GHz),
            (0, None, apu.K),
            (0, None, apu.hPa),
            (1.e-30, None, cnv.dimless),
            (1.e-30, None, apu.km),
            (None, None, apu.mrad),
            (None, None, apu.mrad),
            (0, None, cnv.dBi),
            (0, None, cnv.dBi),
            (0, 100, apu.percent),
            ]
        check_astro_quantities(pfunc, args_list)

        dist = Quantity([1., 2., 5., 10., 50., 50.], apu.km)
        freq = Quantity([1., 1., 0.5, 10., 2., 2.], apu.GHz)
        tamb = Quantity([270., 280., 280., 300., 300., 300.], apu.K)
        pressure = Quantity([1000., 1010., 990., 990., 1000., 1000.], apu.hPa)

        lon = Quantity([50., 50., 51., 51., 52., 52.], apu.deg)
        lat = Quantity([6., 7., 6., 7., 6., 7.], apu.deg)
        d_tm = d_lm = dist
        delta_N, beta_0, N_0 = pathprof.radiomet_data_for_pathcenter(
            lon, lat, d_tm, d_lm
            )
        a_e = pathprof.median_effective_earth_radius(lon, lat)

        # 1 mrad ~= 0.06 deg
        theta_t = Quantity([4., 3., 2., 1., 1., 1.], apu.mrad)
        theta_r = Quantity([-5., -4., -3., 1., 1., 1.], apu.mrad)

        G_t = Quantity([0., 0., 20., 20., 0., 0.], cnv.dBi)
        G_r = Quantity([0., 0., 0., 10., 0., 0.], cnv.dBi)

        time_percent = 50. * apu.percent

        L_bs = Quantity([
            134.34727553, 140.62529652, 140.30808769, 180.93589722,
            180.63083783, 180.82137118
            ], cnv.dB)

        L_bs_f = pfunc(
            dist,
            freq,
            tamb,
            pressure,
            N_0,
            a_e,
            theta_t,
            theta_r,
            G_t,
            G_r,
            time_percent,
            )
        print(L_bs_f)

        assert_quantity_allclose(
            L_bs_f,
            L_bs,
            )

