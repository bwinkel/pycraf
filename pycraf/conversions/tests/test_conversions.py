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
from ... import conversions as cnv
from ...utils import check_astro_quantities
# from astropy.utils.misc import NumpyRNGContext


class TestConversions:

    def setup(self):

        pass

    def teardown(self):

        pass

    def test_db_scales(self):

        for o in [-10, -5, -1, 0, 1, 5, 10]:

            _db = 10 * o
            _val = 10 ** o

            assert_quantity_allclose(
                (_db * cnv.dB).to(cnv.dimless).value,
                _val
                )

            assert_quantity_allclose(
                (_db * cnv.dB_W).to(apu.W).value,
                _val
                )

            assert_quantity_allclose(
                (_db * cnv.dB_mW).to(apu.mW).value,
                _val
                )

            assert_quantity_allclose(
                (_db * cnv.dB_W_Hz).to(apu.W / apu.Hz).value,
                _val
                )

            assert_quantity_allclose(
                (_db * cnv.dB_W_m2).to(apu.W / apu.m ** 2).value,
                _val
                )

            assert_quantity_allclose(
                (_db * cnv.dB_W_m2_Hz).to(apu.W / apu.m ** 2 / apu.Hz).value,
                _val
                )

            assert_quantity_allclose(
                (_db * cnv.dB_Jy_Hz).to(apu.Jy * apu.Hz).value,
                _val
                )

            assert_quantity_allclose(
                (_db * cnv.dB_uV_m).to((apu.uV / apu.m) ** 2).value,
                _val
                )

    def test_constants(self):

        assert_quantity_allclose(
            cnv.R0, 376.7303135 * apu.Ohm
            )
        assert_quantity_allclose(
            cnv.Erx_unit, 5475.330657 * apu.uV / apu.m
            )

        # also check, if default unit is the most simple one:

        assert_equal(
            cnv.R0.unit, apu.Ohm
            )
        assert_equal(
            cnv.Erx_unit.unit, apu.uV / apu.m
            )

    def test_iso_eff_area(self):

        args_list = [
            (0, None, apu.Hz),
            ]
        check_astro_quantities(cnv.iso_eff_area, args_list)

        freq = 1.42 * apu.GHz
        A_iso = 0.0035469483 * apu.m ** 2

        assert_quantity_allclose(
            cnv.iso_eff_area(freq),
            A_iso
            )

    def test_eff_from_geom_area(self):

        args_list = [
            (0, None, apu.m ** 2),
            (0, 100, apu.percent),
            ]
        check_astro_quantities(cnv.eff_from_geom_area, args_list)

        Aeff = 50. * apu.m ** 2
        Ageom = 100. * apu.m ** 2

        assert_quantity_allclose(
            cnv.eff_from_geom_area(Ageom, 50 * apu.percent),
            Aeff
            )
        assert_quantity_allclose(
            cnv.eff_from_geom_area(Ageom, 0.5 * cnv.dimless),
            Aeff
            )

    def test_geom_from_eff_area(self):

        args_list = [
            (0, None, apu.m ** 2),
            (0, 100, apu.percent),
            ]
        check_astro_quantities(cnv.geom_from_eff_area, args_list)

        Aeff = 50. * apu.m ** 2
        Ageom = 100. * apu.m ** 2

        assert_quantity_allclose(
            cnv.geom_from_eff_area(Aeff, 50 * apu.percent),
            Ageom
            )
        assert_quantity_allclose(
            cnv.geom_from_eff_area(Aeff, 0.5 * cnv.dimless),
            Ageom
            )

    def test_eta_a_from_areas(self):

        args_list = [
            (0, None, apu.m ** 2),
            (0, None, apu.m ** 2),
            ]
        check_astro_quantities(cnv.eta_a_from_areas, args_list)

        Aeff = 50. * apu.m ** 2
        Ageom = 100. * apu.m ** 2

        assert_quantity_allclose(
            cnv.eta_a_from_areas(Ageom, Aeff),
            50 * apu.percent
            )

    def test_gamma_from_eff_area(self):

        args_list = [
            (0, None, apu.m ** 2),
            ]
        check_astro_quantities(cnv.gamma_from_eff_area, args_list)

        assert_quantity_allclose(
            cnv.gamma_from_eff_area(4000. * apu.m ** 2),
            1.4485946068301296 * apu.K / apu.Jy
            )

    def test_eff_area_from_gamma(self):

        args_list = [
            (0, None, apu.K / apu.Jy),
            ]
        check_astro_quantities(cnv.eff_area_from_gamma, args_list)

        assert_quantity_allclose(
            cnv.eff_area_from_gamma(1.4485946068301296 * apu.K / apu.Jy),
            4000. * apu.m ** 2
            )

    def test_gain_from_eff_area(self):

        args_list = [
            (0, None, apu.m ** 2),
            (0, None, apu.Hz),
            ]
        check_astro_quantities(cnv.gain_from_eff_area, args_list)

        assert_quantity_allclose(
            cnv.gain_from_eff_area(50. * apu.m ** 2, 1 * apu.GHz),
            38.4453846250226 * cnv.dBi
            )

        assert_quantity_allclose(
            cnv.gain_from_eff_area(50. * apu.m ** 2, 10 * apu.GHz),
            58.4453846250226 * cnv.dBi
            )

    def test_eff_area_from_gain(self):

        args_list = [
            (1.e-60, None, cnv.dimless),
            (0, None, apu.Hz),
            ]
        check_astro_quantities(cnv.eff_area_from_gain, args_list)

        assert_quantity_allclose(
            cnv.eff_area_from_gain(38.4453846250226 * cnv.dB, 1 * apu.GHz),
            50. * apu.m ** 2
            )
        assert_quantity_allclose(
            cnv.eff_area_from_gain(58.4453846250226 * cnv.dB, 10 * apu.GHz),
            50. * apu.m ** 2
            )

    def test_antfactor_from_gain(self):

        args_list = [
            (1.e-60, None, cnv.dimless),
            (0, None, apu.Hz),
            (0, None, apu.Ohm),
            ]
        check_astro_quantities(cnv.antfactor_from_gain, args_list)

        assert_quantity_allclose(
            cnv.antfactor_from_gain(38 * cnv.dB, 1 * apu.GHz, 10 * apu.Ohm),
            -0.39200487434260217 * cnv.dB_1_m
            )
        assert_quantity_allclose(
            cnv.antfactor_from_gain(58 * cnv.dB, 10 * apu.GHz, 20 * apu.Ohm),
            -1.8971548526625104 * cnv.dB_1_m
            )

    def test_gain_from_antfactor(self):

        # args_list = [
        #     (1.e-30, None, 1. / apu.m),
        #     (0, None, apu.Hz),
        #     (0, None, apu.Ohm),
        #     ]
        # check_astro_quantities(cnv.gain_from_antfactor, args_list)

        assert_quantity_allclose(
            cnv.gain_from_antfactor(
                -0.39200487434260217 * cnv.dB_1_m, 1 * apu.GHz, 10 * apu.Ohm
                ),
            38 * cnv.dB
            )
        assert_quantity_allclose(
            cnv.gain_from_antfactor(
                -1.8971548526625104 * cnv.dB_1_m, 10 * apu.GHz, 20 * apu.Ohm
                ),
            58 * cnv.dB
            )

    def test_powerflux_from_efield(self):

        args_list = [
            (1.e-60, None, apu.V / apu.meter),
            ]
        check_astro_quantities(cnv.powerflux_from_efield, args_list)

        assert_quantity_allclose(
            cnv.powerflux_from_efield(20 * cnv.dB_uV_m),
            2.654418729345079e-13 * apu.W / apu.m ** 2
            )
        assert_quantity_allclose(
            cnv.powerflux_from_efield(10 * apu.uV / apu.m),
            2.654418729345079e-13 * apu.W / apu.m ** 2
            )

        assert_quantity_allclose(
            cnv.powerflux_from_efield(100 * apu.uV ** 2 / apu.m ** 2),
            2.654418729345079e-13 * apu.W / apu.m ** 2
            )

    def test_efield_from_powerflux(self):

        args_list = [
            (None, None, cnv.dB_W_m2),
            ]
        check_astro_quantities(cnv.efield_from_powerflux, args_list)

        assert_quantity_allclose(
            cnv.efield_from_powerflux(10 * cnv.dB_W_m2).to(cnv.dB_uV_m),
            155.76030566965238 * cnv.dB_uV_m
            )
        assert_quantity_allclose(
            cnv.efield_from_powerflux(10 * apu.W / apu.m ** 2),
            61378360.4762272 * apu.uV / apu.m
            )

    def test_ptx_from_efield(self):

        args_list = [
            (1.e-60, None, apu.V / apu.meter),
            (1.e-30, None, apu.m),
            (1.e-60, None, cnv.dimless),
            ]
        check_astro_quantities(cnv.ptx_from_efield, args_list)

        assert_quantity_allclose(
            cnv.ptx_from_efield(20 * cnv.dB_uV_m, 1 * apu.km, 20 * cnv.dBi),
            3.3356409518646604e-08 * apu.W
            )
        assert_quantity_allclose(
            cnv.ptx_from_efield(10 * apu.uV / apu.m, 1 * apu.km, 20 * cnv.dBi),
            3.3356409518646604e-08 * apu.W
            )

        assert_quantity_allclose(
            cnv.ptx_from_efield(
                100 * apu.uV ** 2 / apu.m ** 2, 1 * apu.km, 20 * cnv.dBi
                ),
            3.3356409518646604e-08 * apu.W
            )

    def test_efield_from_ptx(self):

        args_list = [
            (1.e-60, None, apu.W),
            (1.e-30, None, apu.m),
            (1.e-60, None, cnv.dimless),
            ]
        check_astro_quantities(cnv.efield_from_ptx, args_list)

        assert_quantity_allclose(
            cnv.efield_from_ptx(10 * cnv.dB_W, 1 * apu.km, 20 * cnv.dBi),
            173145.15817963346 * apu.uV / apu.meter
            )
        assert_quantity_allclose(
            cnv.efield_from_ptx(10 * apu.W, 1 * apu.km, 20 * cnv.dBi),
            173145.15817963346 * apu.uV / apu.meter
            )

    def test_powerflux_from_ptx(self):

        args_list = [
            (1.e-60, None, apu.W),
            (1.e-30, None, apu.m),
            (1.e-60, None, cnv.dimless),
            ]
        check_astro_quantities(cnv.powerflux_from_ptx, args_list)

        assert_quantity_allclose(
            cnv.powerflux_from_ptx(10 * cnv.dB_W, 1 * apu.km, 20 * cnv.dBi),
            7.957747154594768e-05 * apu.W / apu.m ** 2
            )
        assert_quantity_allclose(
            cnv.powerflux_from_ptx(10 * apu.W, 1 * apu.km, 20 * cnv.dBi),
            7.957747154594768e-05 * apu.W / apu.m ** 2
            )

    def test_ptx_from_powerflux(self):

        args_list = [
            (1.e-60, None, apu.W / apu.m ** 2),
            (1.e-30, None, apu.m),
            (1.e-60, None, cnv.dimless),
            ]
        check_astro_quantities(cnv.ptx_from_powerflux, args_list)

        assert_quantity_allclose(
            cnv.ptx_from_powerflux(10 * cnv.dB_W_m2, 1 * apu.km, 20 * cnv.dBi),
            1256637.0614359172 * apu.W
            )
        assert_quantity_allclose(
            cnv.ptx_from_powerflux(
                10 * apu.W / apu.m ** 2, 1 * apu.km, 20 * cnv.dBi
                ),
            1256637.0614359172 * apu.W
            )

    def test_powerflux_from_prx(self):

        args_list = [
            (1.e-90, None, apu.W),
            (1.e-30, None, apu.Hz),
            (1.e-60, None, cnv.dimless),
            ]
        check_astro_quantities(cnv.powerflux_from_prx, args_list)

        assert_quantity_allclose(
            cnv.powerflux_from_prx(10 * cnv.dB_W, 1 * apu.GHz, 20 * cnv.dBi),
            13.981972968457278 * apu.W / apu.m ** 2
            )
        assert_quantity_allclose(
            cnv.powerflux_from_prx(10 * apu.W, 1 * apu.GHz, 20 * cnv.dBi),
            13.981972968457278 * apu.W / apu.m ** 2
            )

    def test_prx_from_powerflux(self):

        args_list = [
            (1.e-90, None, apu.W / apu.m ** 2),
            (1.e-30, None, apu.Hz),
            (1.e-60, None, cnv.dimless),
            ]
        check_astro_quantities(cnv.prx_from_powerflux, args_list)

        assert_quantity_allclose(
            cnv.prx_from_powerflux(
                -160 * cnv.dB_W_m2, 1 * apu.Hz, 20 * cnv.dBi
                ),
            71.52066466270222 * apu.W
            )
        assert_quantity_allclose(
            cnv.prx_from_powerflux(
                1E-16 * apu.W / apu.m ** 2, 1 * apu.Hz, 20 * cnv.dBi
                ),
            71.52066466270222 * apu.W
            )

    def test_t_a_from_prx_nu(self):

        args_list = [
            (1.e-60, None, apu.W / apu.Hz),
            ]
        check_astro_quantities(cnv.t_a_from_prx_nu, args_list)

        assert_quantity_allclose(
            cnv.t_a_from_prx_nu(2.76129704e-21 * apu.W / apu.Hz),
            100 * apu.K
            )

    def test_prx_nu_from_t_a(self):

        args_list = [
            (1.e-30, None, apu.K),
            ]
        check_astro_quantities(cnv.prx_nu_from_t_a, args_list)

        assert_quantity_allclose(
            cnv.prx_nu_from_t_a(100 * apu.K),
            2.76129704e-21 * apu.W / apu.Hz
            )

    def test_t_a_from_powerflux_nu(self):

        args_list = [
            (1.e-30, None, apu.Jy),
            (0, None, apu.m ** 2),
            ]
        check_astro_quantities(cnv.t_a_from_powerflux_nu, args_list)

        assert_quantity_allclose(
            cnv.t_a_from_powerflux_nu(1 * apu.Jy, 4000 * apu.m ** 2),
            1.4485946068301296 * apu.K
            )

    def test_powerflux_nu_from_t_a(self):

        args_list = [
            (1.e-30, None, apu.K),
            (0, None, apu.m ** 2),
            ]
        check_astro_quantities(cnv.powerflux_nu_from_t_a, args_list)

        assert_quantity_allclose(
            cnv.powerflux_nu_from_t_a(
                1.4485946068301296 * apu.K, 4000 * apu.m ** 2
                ),
            1 * apu.Jy
            )

    def test_free_space_loss(self):

        args_list = [
            (1.e-30, None, apu.m),
            (1.e-30, None, apu.Hz),
            ]
        check_astro_quantities(cnv.free_space_loss, args_list)

        assert_quantity_allclose(
            cnv.free_space_loss(1 * apu.km, 1 * apu.GHz),
            -92.44778322188337 * cnv.dB
            )

    def test_prx_from_ptx(self):

        args_list = [
            (1.e-60, None, apu.W),
            (1.e-60, None, cnv.dimless),
            (1.e-60, None, cnv.dimless),
            (1.e-30, None, apu.m),
            (1.e-30, None, apu.Hz),
            ]
        check_astro_quantities(cnv.prx_from_ptx, args_list)

        assert_quantity_allclose(
            cnv.prx_from_ptx(
                10 * cnv.dB_W,
                50 * cnv.dBi, 50 * cnv.dBi,
                1 * apu.km, 1 * apu.GHz
                ),
            56.9143365714346 * apu.W
            )
        assert_quantity_allclose(
            cnv.prx_from_ptx(
                10 * apu.W,
                50 * cnv.dBi, 50 * cnv.dBi,
                1 * apu.km, 1 * apu.GHz
                ),
            56.9143365714346 * apu.W
            )

    def test_ptx_from_prx(self):

        args_list = [
            (1.e-60, None, apu.W),
            (1.e-60, None, cnv.dimless),
            (1.e-60, None, cnv.dimless),
            (1.e-30, None, apu.m),
            (1.e-30, None, apu.Hz),
            ]
        check_astro_quantities(cnv.ptx_from_prx, args_list)

        assert_quantity_allclose(
            cnv.ptx_from_prx(
                10 * cnv.dB_W,
                50 * cnv.dBi, 50 * cnv.dBi,
                1 * apu.km, 1 * apu.GHz
                ),
            1.7570265424158553 * apu.W
            )
        assert_quantity_allclose(
            cnv.ptx_from_prx(
                10 * apu.W,
                50 * cnv.dBi, 50 * cnv.dBi,
                1 * apu.km, 1 * apu.GHz
                ),
            1.7570265424158553 * apu.W
            )
