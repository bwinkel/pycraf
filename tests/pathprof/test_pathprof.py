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
