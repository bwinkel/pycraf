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

        # TODO: add further test cases

        self.freq = 2. * apu.GHz
        self.lon_t, self.lat_t = 6 * apu.deg, 50 * apu.deg
        self.lon_r, self.lat_r = 6.1 * apu.deg, 50.25 * apu.deg
        self.hprof_step = 100 * apu.m

        self.omega = 0. * apu.percent
        self.temperature = (273.15 + 15.) * apu.K  # as in Excel sheet
        self.pressure = 1013. * apu.hPa
        self.time_percent = 50. * apu.percent

        # transhorizon numbers:
        self.h_tg_trans, self.h_rg_trans = 50 * apu.m, 90 * apu.m

        self.pathprop_trans = pathprof.path_properties_with_units(
            self.freq,
            self.lon_t, self.lat_t,
            self.lon_r, self.lat_r,
            self.h_tg_trans, self.h_rg_trans,
            self.hprof_step,
            )

        # LOS numbers:
        self.h_tg_los, self.h_rg_los = 200 * apu.m, 200 * apu.m

        self.pathprop_los = pathprof.path_properties_with_units(
            self.freq,
            self.lon_t, self.lat_t,
            self.lon_r, self.lat_r,
            self.h_tg_los, self.h_rg_los,
            self.hprof_step,
            )

    def teardown(self):

        pass

    def test_path_properties(self):

        pfunc = pathprof.path_properties_with_units
        args_list = [
            (1.e-30, None, apu.GHz),
            (0, 360, apu.deg),
            (-90, 90, apu.deg),
            (0, 360, apu.deg),
            (-90, 90, apu.deg),
            (0, None, apu.m),
            (0, None, apu.m),
            (0, None, apu.m),
            ]
        check_astro_quantities(pfunc, args_list)

        pathprop_trans = pfunc(
            self.freq,
            self.lon_t, self.lat_t,
            self.lon_r, self.lat_r,
            self.h_tg_trans, self.h_rg_trans,
            self.hprof_step,
            )

        pathprop_trans_true = pathprof.PathProps(
            lon_mid=Quantity(6.050021979477151, apu.deg),
            lat_mid=Quantity(50.125392608270644, apu.deg),
            delta_N=Quantity(39.276915092335486, 1 / apu.km),
            N0=Quantity(325.48649839940964, cnv.dimless),
            distance=Quantity(28.712600748321183, apu.km),
            bearing=Quantity(14.383138371761586, apu.deg),
            back_bearing=Quantity(-165.6168616282384, apu.deg),
            a_e=Quantity(8496.60880688387, apu.km),
            h0=Quantity(391.0, apu.m),
            hn=Quantity(514.5184936523438, apu.m),
            h_ts=Quantity(441.0, apu.m),
            h_rs=Quantity(604.5184936523438, apu.m),
            h_st=Quantity(391.0, apu.m),
            h_sr=Quantity(475.1045924277951, apu.m),
            h_std=Quantity(391.0, apu.m),
            h_srd=Quantity(468.3875514121701, apu.m),
            h_te=Quantity(50.0, apu.m),
            h_re=Quantity(129.41390122454862, apu.m),
            h_m=Quantity(96.53708718864101, apu.m),
            d_lt=Quantity(9.6, apu.km),
            d_lr=Quantity(19.112600748321185, apu.km),
            theta_t=Quantity(7.211744865969712, apu.mrad),
            theta_r=Quantity(-5.77400698654872, apu.mrad),
            theta=Quantity(4.8170391278271225, apu.mrad),
            nu_bull=Quantity(1.4065802319358092, cnv.dimless),
            path_type=1,
            )

        for t1, t2 in zip(pathprop_trans, pathprop_trans_true):
            assert_quantity_allclose(t1, t2)

        # LOS path
        pathprop_los = pfunc(
            self.freq,
            self.lon_t, self.lat_t,
            self.lon_r, self.lat_r,
            self.h_tg_los, self.h_rg_los,
            self.hprof_step,
            )

        pathprop_los_true = pathprof.PathProps(
            lon_mid=Quantity(6.050021979477151, apu.deg),
            lat_mid=Quantity(50.125392608270644, apu.deg),
            delta_N=Quantity(39.276915092335486, 1. / apu.km),
            N0=Quantity(325.48649839940964, cnv.dimless),
            distance=Quantity(28.712600748321183, apu.km),
            bearing=Quantity(14.383138371761586, apu.deg),
            back_bearing=Quantity(-165.6168616282384, apu.deg),
            a_e=Quantity(8496.60880688387, apu.km),
            h0=Quantity(391.0, apu.m),
            hn=Quantity(514.5184936523438, apu.m),
            h_ts=Quantity(591.0, apu.m),
            h_rs=Quantity(714.5184936523438, apu.m),
            h_st=Quantity(391.0, apu.m),
            h_sr=Quantity(475.1045924277951, apu.m),
            h_std=Quantity(391.0, apu.m),
            h_srd=Quantity(475.1045924277951, apu.m),
            h_te=Quantity(200.0, apu.m),
            h_re=Quantity(239.41390122454862, apu.m),
            h_m=Quantity(96.53708718864101, apu.m),
            d_lt=Quantity(9.6, apu.km),
            d_lr=Quantity(19.112600748321185, apu.km),
            theta_t=Quantity(2.6122349531799145, apu.mrad),
            theta_r=Quantity(-5.991470448965087, apu.mrad),
            theta=Quantity(6.575262095775969e-05, apu.mrad),
            nu_bull=Quantity(-31.84165949880204, cnv.dimless),
            path_type=0,
            )

        for t1, t2 in zip(pathprop_los, pathprop_los_true):
            assert_quantity_allclose(t1, t2)

    def test_free_space_loss_bfsg(self):

        pfunc = pathprof.free_space_loss_bfsg
        # args_list = [
        #     (None, None, None),
        #     (1.e-30, None, apu.GHz),
        #     (0, 100, apu.percent),
        #     (0, None, apu.K),
        #     (0, None, apu.hPa),
        #     (0, 100, apu.percent),
        #     ]
        # check_astro_quantities(pfunc, args_list)

        # trans == los:
        # Lbfsg_trans = pfunc(
        #     self.pathprop_trans, self.freq, self.omega,
        #     self.temperature, self.pressure, self.time_percent
        #     )
        # Lbfsg_trans_true = 127.879838 * cnv.dB

        # assert_quantity_allclose(
        #     Lbfsg_trans,
        #     Lbfsg_trans_true,
        #     )

        Lbfsg_los = pfunc(
            self.pathprop_los, self.freq, self.omega,
            self.temperature, self.pressure, self.time_percent
            )
        Lbfsg_los_true = 127.879838 * cnv.dB

        assert_quantity_allclose(
            Lbfsg_los,
            Lbfsg_los_true,
            )

        # # test broadcasting
        # dist = Quantity([1., 10., 20.], apu.km)
        # freq = Quantity([1., 2., 22.], apu.GHz)
        # omega = Quantity(0., apu.percent)
        # temperature = Quantity(300., apu.K)
        # pressure = Quantity(1013., apu.hPa)
        # Lbfsg_annex1 = Quantity(
        #     [
        #         [92.5048683, 98.52672276, 119.53615245],
        #         [112.54868299, 118.58182837, 141.22544192],
        #         [118.61796588, 124.66365675, 149.12303013],
        #         ], cnv.dB
        #     )
        # assert_quantity_allclose(
        #     pfunc(dist[:, np.newaxis], freq, omega, temperature, pressure),
        #     Lbfsg_annex1,
        #     )

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
        temperature = Quantity([270., 280., 280., 300., 300., 300.], apu.K)
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
            temperature,
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

    def test_ducting_loss_ba(self):

        pfunc = pathprof.ducting_loss_ba
        args_list = [
            (1.e-30, None, apu.km),
            (1.e-30, None, apu.GHz),
            (0, 100, apu.percent),
            (0, None, apu.K),
            (0, None, apu.hPa),
            (1.e-30, None, cnv.dimless),
            (1.e-30, None, apu.km),
            (0, None, apu.m),
            (0, None, apu.m),
            (0, None, apu.m),
            (0, None, apu.m),
            (0, None, apu.m),
            (0, None, apu.km),
            (0, None, apu.km),
            (0, None, apu.km),
            (0, None, apu.km),
            (0, None, apu.km),
            (None, None, apu.mrad),
            (None, None, apu.mrad),
            (0, None, cnv.dBi),
            (0, None, cnv.dBi),
            (0, 100, apu.percent),
            (0, 100, apu.percent),
            ]
        check_astro_quantities(pfunc, args_list)

        # numbers below fully consistent with Excel sheet
        dist = Quantity(22.2, apu.km)
        freq = Quantity(2., apu.GHz)
        omega = Quantity(0., apu.percent)
        temperature = Quantity(293., apu.K)
        pressure = Quantity(1013., apu.hPa)

        lon = Quantity(6., apu.deg)
        lat = Quantity(50., apu.deg)
        d_tm = d_lm = dist
        delta_N, beta_0, N_0 = pathprof.radiomet_data_for_pathcenter(
            lon, lat, d_tm, d_lm
            )
        a_e = pathprof.median_effective_earth_radius(lon, lat)

        # 1 mrad ~= 0.06 deg
        theta_t = Quantity(-9.5, apu.mrad)
        theta_r = Quantity(-9.5, apu.mrad)

        G_t = Quantity(0., cnv.dBi)
        G_r = Quantity(0., cnv.dBi)

        h_ts = Quantity(210., apu.m)
        h_rs = Quantity(210., apu.m)
        h_te = Quantity(200., apu.m)
        h_re = Quantity(200., apu.m)
        h_m = Quantity(29.8, apu.m)

        d_lt = Quantity(11.1, apu.km)
        d_lr = Quantity(11.1, apu.km)
        d_ct = Quantity(1000., apu.km)
        d_cr = Quantity(1000., apu.km)
        d_lm = Quantity(22.2, apu.km)

        time_percent = Quantity(50., apu.percent)

        L_ba = Quantity(199.59448682, cnv.dB)

        L_ba_f = pfunc(
            dist,
            freq,
            omega,
            temperature,
            pressure,
            N_0,
            a_e,
            h_ts,
            h_rs,
            h_te,
            h_re,
            h_m,
            d_lt,
            d_lr,
            d_ct,
            d_cr,
            d_lm,
            theta_t,
            theta_r,
            G_t,
            G_r,
            beta_0,
            time_percent,
            atm_method='annex1',
            )
        print(L_ba_f)

        assert_quantity_allclose(
            L_ba_f,
            L_ba,
            )

        # TODO: add further test cases (once path analysis is ready)

