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

        # first test, if assert Quantity works
        with pytest.raises(TypeError):

            cnv.Aeff_from_Ageom(1., 50 * apu.percent)
            cnv.Aeff_from_Ageom(1. * apu.m ** 2, 0.5)
            cnv.Aeff_from_Ageom(1., 0.5)

        with pytest.raises(apu.UnitsError):

            cnv.Aeff_from_Ageom(1. * apu.Hz, 50 * apu.percent)
            cnv.Aeff_from_Ageom(1. * apu.m ** 2, 0.5 * apu.m)

        Ageom = 50. * apu.m ** 2
        Aeff = 100. * apu.m ** 2
        assert cnv.Aeff_from_Ageom(Aeff, 50 * apu.percent).unit == apu.m ** 2

        assert_quantity_allclose(
            cnv.Aeff_from_Ageom(Aeff, 50 * apu.percent),
            Ageom
            )
        assert_quantity_allclose(
            cnv.Aeff_from_Ageom(Aeff, 0.5 * cnv.dimless),
            Ageom
            )

    def test_Ageom_from_Aeff(self):

        # first test, if assert Quantity works
        with pytest.raises(TypeError):

            cnv.Ageom_from_Aeff(1., 50 * apu.percent)
            cnv.Ageom_from_Aeff(1. * apu.m ** 2, 0.5)
            cnv.Ageom_from_Aeff(1., 0.5)

        with pytest.raises(apu.UnitsError):

            cnv.Ageom_from_Aeff(1. * apu.Hz, 50 * apu.percent)
            cnv.Ageom_from_Aeff(1. * apu.m ** 2, 0.5 * apu.m)

        Ageom = 50. * apu.m ** 2
        Aeff = 100. * apu.m ** 2
        assert cnv.Ageom_from_Aeff(Ageom, 50 * apu.percent).unit == apu.m ** 2

        assert_quantity_allclose(
            cnv.Ageom_from_Aeff(Ageom, 50 * apu.percent),
            Aeff
            )
        assert_quantity_allclose(
            cnv.Ageom_from_Aeff(Ageom, 0.5 * cnv.dimless),
            Aeff
            )

    def test_Gain_from_Aeff(self):

        # first test, if assert Quantity works
        with pytest.raises(TypeError):

            cnv.Gain_from_Aeff(1., 50 * apu.Hz)
            cnv.Gain_from_Aeff(1. * apu.m ** 2, 0.5)
            cnv.Gain_from_Aeff(1., 0.5)

        with pytest.raises(apu.UnitsError):

            cnv.Gain_from_Aeff(1. * apu.Hz, 50 * apu.Hz)
            cnv.Gain_from_Aeff(1. * apu.m ** 2, 0.5 * apu.m)

        assert_quantity_allclose(
            cnv.Gain_from_Aeff(50. * apu.m ** 2, 1 * apu.GHz),
            38.4453846250226 * cnv.dB
            )

        assert_quantity_allclose(
            cnv.Gain_from_Aeff(50. * apu.m ** 2, 10 * apu.GHz),
            58.4453846250226 * cnv.dB
            )

    def test_Aeff_from_Gain(self):

        # first test, if assert Quantity works
        with pytest.raises(TypeError):

            cnv.Aeff_from_Gain(1., 50 * apu.Hz)
            cnv.Aeff_from_Gain(1. * cnv.dB, 0.5)
            cnv.Aeff_from_Gain(1. * apu.Hz, 0.5)

        with pytest.raises(apu.UnitsError):

            cnv.Aeff_from_Gain(1. * apu.Hz, 50 * apu.Hz)
            cnv.Aeff_from_Gain(1. * cnv.dB, 0.5 * apu.m)

        assert_quantity_allclose(
            cnv.Aeff_from_Gain(38.4453846250226 * cnv.dB, 1 * apu.GHz),
            50. * apu.m ** 2
            )
        assert_quantity_allclose(
            cnv.Aeff_from_Gain(58.4453846250226 * cnv.dB, 10 * apu.GHz),
            50. * apu.m ** 2
            )

    def test_S_from_E(self):

        # first test, if assert Quantity works
        with pytest.raises(TypeError):

            cnv.S_from_E(1.)

        with pytest.raises(apu.UnitsError):

            cnv.S_from_E(1. * apu.Hz)

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

        # first test, if assert Quantity works
        with pytest.raises(TypeError):

            cnv.E_from_S(1.)

        with pytest.raises(apu.UnitsError):

            cnv.E_from_S(1. * apu.Hz)

        assert_quantity_allclose(
            cnv.E_from_S(10 * cnv.dB_W_m2).to(cnv.dB_uV_m),
            155.76030566965238 * cnv.dB_uV_m
            )
        assert_quantity_allclose(
            cnv.E_from_S(10 * apu.W / apu.m ** 2),
            61378360.4762272 * apu.uV / apu.m
            )

    def test_Ptx_from_Erx(self):

        # first test, if assert Quantity works
        with pytest.raises(TypeError):

            cnv.Ptx_from_Erx(1, 1 * apu.km, 20 * cnv.dBi)
            cnv.Ptx_from_Erx(1 * cnv.dB_uV_m, 1, 20 * cnv.dBi)
            cnv.Ptx_from_Erx(1 * cnv.dB_uV_m, 1 * apu.km, 20)

        with pytest.raises(apu.UnitsError):

            cnv.Ptx_from_Erx(1 * apu.Hz, 1 * apu.km, 20 * cnv.dBi)
            cnv.Ptx_from_Erx(1 * cnv.dB_uV_m, 1 * apu.Hz, 20 * cnv.dBi)
            cnv.Ptx_from_Erx(1 * cnv.dB_uV_m, 1 * apu.km, 20 * apu.Hz)

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

        # first test, if assert Quantity works
        with pytest.raises(TypeError):

            cnv.Erx_from_Ptx(1, 1 * apu.km, 20 * cnv.dBi)
            cnv.Erx_from_Ptx(1 * cnv.dB_W, 1, 20 * cnv.dBi)
            cnv.Erx_from_Ptx(1 * cnv.dB_W, 1 * apu.km, 20)

        with pytest.raises(apu.UnitsError):

            cnv.Erx_from_Ptx(1 * apu.Hz, 1 * apu.km, 20 * cnv.dBi)
            cnv.Erx_from_Ptx(1 * cnv.dB_W, 1 * apu.Hz, 20 * cnv.dBi)
            cnv.Erx_from_Ptx(1 * cnv.dB_W, 1 * apu.km, 20 * apu.Hz)

        assert_quantity_allclose(
            cnv.Erx_from_Ptx(10 * cnv.dB_W, 1 * apu.km, 20 * cnv.dBi),
            173145.15817963346 * apu.uV / apu.meter
            )
        assert_quantity_allclose(
            cnv.Erx_from_Ptx(10 * apu.W, 1 * apu.km, 20 * cnv.dBi),
            173145.15817963346 * apu.uV / apu.meter
            )

    def test_S_from_Ptx(self):

        # first test, if assert Quantity works
        with pytest.raises(TypeError):

            cnv.S_from_Ptx(1, 1 * apu.km, 20 * cnv.dBi)
            cnv.S_from_Ptx(1 * cnv.dB_W, 1, 20 * cnv.dBi)
            cnv.S_from_Ptx(1 * cnv.dB_W, 1 * apu.km, 20)

        with pytest.raises(apu.UnitsError):

            cnv.S_from_Ptx(1 * apu.Hz, 1 * apu.km, 20 * cnv.dBi)
            cnv.S_from_Ptx(1 * cnv.dB_W, 1 * apu.Hz, 20 * cnv.dBi)
            cnv.S_from_Ptx(1 * cnv.dB_W, 1 * apu.km, 20 * apu.Hz)

        assert_quantity_allclose(
            cnv.S_from_Ptx(10 * cnv.dB_W, 1 * apu.km, 20 * cnv.dBi),
            7.957747154594768e-05 * apu.W / apu.m ** 2
            )
        assert_quantity_allclose(
            cnv.S_from_Ptx(10 * apu.W, 1 * apu.km, 20 * cnv.dBi),
            7.957747154594768e-05 * apu.W / apu.m ** 2
            )

    def test_Ptx_from_S(self):

        # first test, if assert Quantity works
        with pytest.raises(TypeError):

            cnv.Ptx_from_S(1, 1 * apu.km, 20 * cnv.dBi)
            cnv.Ptx_from_S(1 * cnv.dB_W_m2, 1, 20 * cnv.dBi)
            cnv.Ptx_from_S(1 * cnv.dB_W_m2, 1 * apu.km, 20)

        with pytest.raises(apu.UnitsError):

            cnv.Ptx_from_S(1 * apu.Hz, 1 * apu.km, 20 * cnv.dBi)
            cnv.Ptx_from_S(1 * cnv.dB_W_m2, 1 * apu.Hz, 20 * cnv.dBi)
            cnv.Ptx_from_S(1 * cnv.dB_W_m2, 1 * apu.km, 20 * apu.Hz)

        assert_quantity_allclose(
            cnv.Ptx_from_S(10 * cnv.dB_W_m2, 1 * apu.km, 20 * cnv.dBi),
            1256637.0614359172 * apu.W
            )
        assert_quantity_allclose(
            cnv.Ptx_from_S(10 * apu.W / apu.m ** 2, 1 * apu.km, 20 * cnv.dBi),
            1256637.0614359172 * apu.W
            )

    def test_S_from_Prx(self):

        # first test, if assert Quantity works
        with pytest.raises(TypeError):

            cnv.S_from_Prx(1, 1 * apu.GHz, 20 * cnv.dBi)
            cnv.S_from_Prx(1 * cnv.dB_W, 1, 20 * cnv.dBi)
            cnv.S_from_Prx(1 * cnv.dB_W, 1 * apu.GHz, 20)

        with pytest.raises(apu.UnitsError):

            cnv.S_from_Prx(1 * apu.Hz, 1 * apu.GHz, 20 * cnv.dBi)
            cnv.S_from_Prx(1 * cnv.dB_W, 1 * apu.km, 20 * cnv.dBi)
            cnv.S_from_Prx(1 * cnv.dB_W, 1 * apu.GHz, 20 * apu.Hz)

        assert_quantity_allclose(
            cnv.S_from_Prx(10 * cnv.dB_W, 1 * apu.GHz, 20 * cnv.dBi),
            13.981972968457278 * apu.W / apu.m ** 2
            )
        assert_quantity_allclose(
            cnv.S_from_Prx(10 * apu.W, 1 * apu.GHz, 20 * cnv.dBi),
            13.981972968457278 * apu.W / apu.m ** 2
            )

    def test_Prx_from_S(self):

        # first test, if assert Quantity works
        with pytest.raises(TypeError):

            cnv.Prx_from_S(1, 1 * apu.Hz, 20 * cnv.dBi)
            cnv.Prx_from_S(1 * cnv.dB_W_m2, 1, 20 * cnv.dBi)
            cnv.Prx_from_S(1 * cnv.dB_W_m2, 1 * apu.Hz, 20)

        with pytest.raises(apu.UnitsError):

            cnv.Prx_from_S(1 * apu.Hz, 1 * apu.Hz, 20 * cnv.dBi)
            cnv.Prx_from_S(1 * cnv.dB_W_m2, 1 * apu.m, 20 * cnv.dBi)
            cnv.Prx_from_S(1 * cnv.dB_W_m2, 1 * apu.Hz, 20 * apu.Hz)

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

        # first test, if assert Quantity works
        with pytest.raises(TypeError):

            cnv.free_space_loss(1, 1 * apu.Hz)
            cnv.free_space_loss(1 * apu.m, 1)

        with pytest.raises(apu.UnitsError):

            cnv.free_space_loss(1 * apu.Hz, 1 * apu.Hz)
            cnv.free_space_loss(1 * apu.m, 1 * apu.m)

        assert_quantity_allclose(
            cnv.free_space_loss(1 * apu.km, 1 * apu.GHz),
            -92.44778322188337 * cnv.dB
            )

    def test_Prx_from_Ptx(self):

        # first test, if assert Quantity works
        with pytest.raises(TypeError):

            cnv.Prx_from_Ptx(
                1, 5 * cnv.dBi, 5 * cnv.dBi, 1 * apu.m, 1 * apu.GHz
                )
            cnv.Prx_from_Ptx(
                1 * cnv.dB_W, 5, 5 * cnv.dBi, 1 * apu.m, 1 * apu.GHz
                )
            cnv.Prx_from_Ptx(
                1 * cnv.dB_W, 5 * cnv.dBi, 5, 1 * apu.m, 1 * apu.GHz
                )
            cnv.Prx_from_Ptx(
                1 * cnv.dB_W, 5 * cnv.dBi, 5 * cnv.dBi, 1, 1 * apu.GHz
                )
            cnv.Prx_from_Ptx(
                1 * cnv.dB_W, 5 * cnv.dBi, 5 * cnv.dBi, 1 * apu.m, 1
                )

        with pytest.raises(apu.UnitsError):

            cnv.Prx_from_Ptx(
                1 * apu.V, 5 * cnv.dBi, 5 * cnv.dBi, 1 * apu.m, 1 * apu.GHz
                )
            cnv.Prx_from_Ptx(
                1 * cnv.dB_W, 5 * apu.V, 5 * cnv.dBi, 1 * apu.m, 1 * apu.GHz
                )
            cnv.Prx_from_Ptx(
                1 * cnv.dB_W, 5 * cnv.dBi, 5 * apu.V, 1 * apu.m, 1 * apu.GHz
                )
            cnv.Prx_from_Ptx(
                1 * cnv.dB_W, 5 * cnv.dBi, 5 * cnv.dBi, 1 * apu.V, 1 * apu.GHz
                )
            cnv.Prx_from_Ptx(
                1 * cnv.dB_W, 5 * cnv.dBi, 5 * cnv.dBi, 1 * apu.m, 1 * apu.V
                )

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

        # first test, if assert Quantity works
        with pytest.raises(TypeError):

            cnv.Ptx_from_Prx(
                1, 5 * cnv.dBi, 5 * cnv.dBi, 1 * apu.m, 1 * apu.GHz
                )
            cnv.Ptx_from_Prx(
                1 * cnv.dB_W, 5, 5 * cnv.dBi, 1 * apu.m, 1 * apu.GHz
                )
            cnv.Ptx_from_Prx(
                1 * cnv.dB_W, 5 * cnv.dBi, 5, 1 * apu.m, 1 * apu.GHz
                )
            cnv.Ptx_from_Prx(
                1 * cnv.dB_W, 5 * cnv.dBi, 5 * cnv.dBi, 1, 1 * apu.GHz
                )
            cnv.Ptx_from_Prx(
                1 * cnv.dB_W, 5 * cnv.dBi, 5 * cnv.dBi, 1 * apu.m, 1
                )

        with pytest.raises(apu.UnitsError):

            cnv.Ptx_from_Prx(
                1 * apu.V, 5 * cnv.dBi, 5 * cnv.dBi, 1 * apu.m, 1 * apu.GHz
                )
            cnv.Ptx_from_Prx(
                1 * cnv.dB_W, 5 * apu.V, 5 * cnv.dBi, 1 * apu.m, 1 * apu.GHz
                )
            cnv.Ptx_from_Prx(
                1 * cnv.dB_W, 5 * cnv.dBi, 5 * apu.V, 1 * apu.m, 1 * apu.GHz
                )
            cnv.Ptx_from_Prx(
                1 * cnv.dB_W, 5 * cnv.dBi, 5 * cnv.dBi, 1 * apu.V, 1 * apu.GHz
                )
            cnv.Ptx_from_Prx(
                1 * cnv.dB_W, 5 * cnv.dBi, 5 * cnv.dBi, 1 * apu.m, 1 * apu.V
                )

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



# from astropy import units as apu
# from astropy.units import Quantity
# from pycraf import conversions as cnv

