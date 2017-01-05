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
from pycraf.helpers import check_astro_quantities
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

    def test_Aeff_from_Ageom(self):

        args_list = [
            (0, None, apu.m ** 2),
            (0, 100, apu.percent),
            ]
        check_astro_quantities(cnv.Aeff_from_Ageom, args_list)

        Ageom = 50. * apu.m ** 2
        Aeff = 100. * apu.m ** 2

        assert_quantity_allclose(
            cnv.Aeff_from_Ageom(Aeff, 50 * apu.percent),
            Ageom
            )
        assert_quantity_allclose(
            cnv.Aeff_from_Ageom(Aeff, 0.5 * cnv.dimless),
            Ageom
            )

    def test_Ageom_from_Aeff(self):

        args_list = [
            (0, None, apu.m ** 2),
            (0, 100, apu.percent),
            ]
        check_astro_quantities(cnv.Ageom_from_Aeff, args_list)

        Ageom = 50. * apu.m ** 2
        Aeff = 100. * apu.m ** 2

        assert_quantity_allclose(
            cnv.Ageom_from_Aeff(Ageom, 50 * apu.percent),
            Aeff
            )
        assert_quantity_allclose(
            cnv.Ageom_from_Aeff(Ageom, 0.5 * cnv.dimless),
            Aeff
            )

    def test_Gain_from_Aeff(self):

        args_list = [
            (0, None, apu.m ** 2),
            (0, None, apu.Hz),
            ]
        check_astro_quantities(cnv.Gain_from_Aeff, args_list)

        assert_quantity_allclose(
            cnv.Gain_from_Aeff(50. * apu.m ** 2, 1 * apu.GHz),
            38.4453846250226 * cnv.dBi
            )

        assert_quantity_allclose(
            cnv.Gain_from_Aeff(50. * apu.m ** 2, 10 * apu.GHz),
            58.4453846250226 * cnv.dBi
            )

    def test_Aeff_from_Gain(self):

        args_list = [
            (1.e-30, None, cnv.dimless),
            (0, None, apu.Hz),
            ]
        check_astro_quantities(cnv.Aeff_from_Gain, args_list)

        assert_quantity_allclose(
            cnv.Aeff_from_Gain(38.4453846250226 * cnv.dB, 1 * apu.GHz),
            50. * apu.m ** 2
            )
        assert_quantity_allclose(
            cnv.Aeff_from_Gain(58.4453846250226 * cnv.dB, 10 * apu.GHz),
            50. * apu.m ** 2
            )

    def test_S_from_E(self):

        args_list = [
            (1.e-30, None, apu.V / apu.meter),
            ]
        check_astro_quantities(cnv.S_from_E, args_list)

        assert_quantity_allclose(
            cnv.S_from_E(20 * cnv.dB_uV_m),
            2.654418729345079e-13 * apu.W / apu.m ** 2
            )
        assert_quantity_allclose(
            cnv.S_from_E(10 * apu.uV / apu.m),
            2.654418729345079e-13 * apu.W / apu.m ** 2
            )

        assert_quantity_allclose(
            cnv.S_from_E(100 * apu.uV ** 2 / apu.m ** 2),
            2.654418729345079e-13 * apu.W / apu.m ** 2
            )

    def test_E_from_S(self):

        args_list = [
            (None, None, cnv.dB_W_m2),
            ]
        check_astro_quantities(cnv.E_from_S, args_list)

        assert_quantity_allclose(
            cnv.E_from_S(10 * cnv.dB_W_m2).to(cnv.dB_uV_m),
            155.76030566965238 * cnv.dB_uV_m
            )
        assert_quantity_allclose(
            cnv.E_from_S(10 * apu.W / apu.m ** 2),
            61378360.4762272 * apu.uV / apu.m
            )

    def test_Ptx_from_Erx(self):

        args_list = [
            (1.e-30, None, apu.V / apu.meter),
            (1.e-30, None, apu.m),
            (1.e-30, None, cnv.dimless),
            ]
        check_astro_quantities(cnv.Ptx_from_Erx, args_list)

        assert_quantity_allclose(
            cnv.Ptx_from_Erx(20 * cnv.dB_uV_m, 1 * apu.km, 20 * cnv.dBi),
            3.3356409518646604e-08 * apu.W
            )
        assert_quantity_allclose(
            cnv.Ptx_from_Erx(10 * apu.uV / apu.m, 1 * apu.km, 20 * cnv.dBi),
            3.3356409518646604e-08 * apu.W
            )

        assert_quantity_allclose(
            cnv.Ptx_from_Erx(
                100 * apu.uV ** 2 / apu.m ** 2, 1 * apu.km, 20 * cnv.dBi
                ),
            3.3356409518646604e-08 * apu.W
            )

    def test_Erx_from_Ptx(self):

        args_list = [
            (1.e-30, None, apu.W),
            (1.e-30, None, apu.m),
            (1.e-30, None, cnv.dimless),
            ]
        check_astro_quantities(cnv.Erx_from_Ptx, args_list)

        assert_quantity_allclose(
            cnv.Erx_from_Ptx(10 * cnv.dB_W, 1 * apu.km, 20 * cnv.dBi),
            173145.15817963346 * apu.uV / apu.meter
            )
        assert_quantity_allclose(
            cnv.Erx_from_Ptx(10 * apu.W, 1 * apu.km, 20 * cnv.dBi),
            173145.15817963346 * apu.uV / apu.meter
            )

    def test_S_from_Ptx(self):

        args_list = [
            (1.e-30, None, apu.W),
            (1.e-30, None, apu.m),
            (1.e-30, None, cnv.dimless),
            ]
        check_astro_quantities(cnv.S_from_Ptx, args_list)

        assert_quantity_allclose(
            cnv.S_from_Ptx(10 * cnv.dB_W, 1 * apu.km, 20 * cnv.dBi),
            7.957747154594768e-05 * apu.W / apu.m ** 2
            )
        assert_quantity_allclose(
            cnv.S_from_Ptx(10 * apu.W, 1 * apu.km, 20 * cnv.dBi),
            7.957747154594768e-05 * apu.W / apu.m ** 2
            )

    def test_Ptx_from_S(self):

        args_list = [
            (1.e-30, None, apu.W / apu.m ** 2),
            (1.e-30, None, apu.m),
            (1.e-30, None, cnv.dimless),
            ]
        check_astro_quantities(cnv.Ptx_from_S, args_list)

        assert_quantity_allclose(
            cnv.Ptx_from_S(10 * cnv.dB_W_m2, 1 * apu.km, 20 * cnv.dBi),
            1256637.0614359172 * apu.W
            )
        assert_quantity_allclose(
            cnv.Ptx_from_S(10 * apu.W / apu.m ** 2, 1 * apu.km, 20 * cnv.dBi),
            1256637.0614359172 * apu.W
            )

    def test_S_from_Prx(self):

        args_list = [
            (1.e-30, None, apu.W),
            (1.e-30, None, apu.Hz),
            (1.e-30, None, cnv.dimless),
            ]
        check_astro_quantities(cnv.S_from_Prx, args_list)

        assert_quantity_allclose(
            cnv.S_from_Prx(10 * cnv.dB_W, 1 * apu.GHz, 20 * cnv.dBi),
            13.981972968457278 * apu.W / apu.m ** 2
            )
        assert_quantity_allclose(
            cnv.S_from_Prx(10 * apu.W, 1 * apu.GHz, 20 * cnv.dBi),
            13.981972968457278 * apu.W / apu.m ** 2
            )

    def test_Prx_from_S(self):

        args_list = [
            (1.e-30, None, apu.W / apu.m ** 2),
            (1.e-30, None, apu.Hz),
            (1.e-30, None, cnv.dimless),
            ]
        check_astro_quantities(cnv.Prx_from_S, args_list)

        assert_quantity_allclose(
            cnv.Prx_from_S(-160 * cnv.dB_W_m2, 1 * apu.Hz, 20 * cnv.dBi),
            71.52066466270222 * apu.W
            )
        assert_quantity_allclose(
            cnv.Prx_from_S(
                1E-16 * apu.W / apu.m ** 2, 1 * apu.Hz, 20 * cnv.dBi
                ),
            71.52066466270222 * apu.W
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

    def test_Prx_from_Ptx(self):

        args_list = [
            (1.e-30, None, apu.W),
            (1.e-30, None, cnv.dimless),
            (1.e-30, None, cnv.dimless),
            (1.e-30, None, apu.m),
            (1.e-30, None, apu.Hz),
            ]
        check_astro_quantities(cnv.Prx_from_Ptx, args_list)

        assert_quantity_allclose(
            cnv.Prx_from_Ptx(
                10 * cnv.dB_W,
                50 * cnv.dBi, 50 * cnv.dBi,
                1 * apu.km, 1 * apu.GHz
                ),
            56.9143365714346 * apu.W
            )
        assert_quantity_allclose(
            cnv.Prx_from_Ptx(
                10 * apu.W,
                50 * cnv.dBi, 50 * cnv.dBi,
                1 * apu.km, 1 * apu.GHz
                ),
            56.9143365714346 * apu.W
            )

    def test_Ptx_from_Prx(self):

        args_list = [
            (1.e-30, None, apu.W),
            (1.e-30, None, cnv.dimless),
            (1.e-30, None, cnv.dimless),
            (1.e-30, None, apu.m),
            (1.e-30, None, apu.Hz),
            ]
        check_astro_quantities(cnv.Ptx_from_Prx, args_list)

        assert_quantity_allclose(
            cnv.Ptx_from_Prx(
                10 * cnv.dB_W,
                50 * cnv.dBi, 50 * cnv.dBi,
                1 * apu.km, 1 * apu.GHz
                ),
            1.7570265424158553 * apu.W
            )
        assert_quantity_allclose(
            cnv.Ptx_from_Prx(
                10 * apu.W,
                50 * cnv.dBi, 50 * cnv.dBi,
                1 * apu.km, 1 * apu.GHz
                ),
            1.7570265424158553 * apu.W
            )
