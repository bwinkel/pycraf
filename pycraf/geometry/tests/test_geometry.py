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
from ... import geometry
from ...utils import check_astro_quantities
from astropy.utils.misc import NumpyRNGContext


TOL_KWARGS = {'atol': 1.e-4, 'rtol': 1.e-4}


def test_true_angular_distance():

    pfunc = geometry.true_angular_distance
    args_list = [
        (None, None, apu.deg),
        (-90, 90, apu.deg),
        (None, None, apu.deg),
        (-90, 90, apu.deg),
        ]
    check_astro_quantities(pfunc, args_list)

    l1 = Quantity([10., 20., 30., 140., 350.], apu.deg)
    b1 = Quantity([0., 20., 60., 40., 80.], apu.deg)
    l2 = Quantity([-40., 200., 30.1, 130., 320.], apu.deg)
    b2 = Quantity([0., -30., 60.05, 30., 10.], apu.deg)

    adist = Quantity(
        [
            5.00000000e+01, 1.70000000e+02, 7.06839454e-02,
            1.29082587e+01, 7.13909421e+01
            ], apu.deg
        )
    assert_quantity_allclose(
        pfunc(l1, b1, l2, b2),
        adist,
        )


def test_great_circle_bearing():

    pfunc = geometry.great_circle_bearing
    args_list = [
        (None, None, apu.deg),
        (-90, 90, apu.deg),
        (None, None, apu.deg),
        (-90, 90, apu.deg),
        ]
    check_astro_quantities(pfunc, args_list)

    l1 = Quantity([10., 20., 30., 140., 350.], apu.deg)
    b1 = Quantity([0., 20., 60., 40., 80.], apu.deg)
    l2 = Quantity([-40., 200., 30.1, 130., 320.], apu.deg)
    b2 = Quantity([0., -30., 60.05, 30., 10.], apu.deg)

    adist = Quantity(
        [
            -90., 180., 44.935034, -137.686456, -148.696726
            ], apu.deg
        )
    assert_quantity_allclose(
        pfunc(l1, b1, l2, b2),
        adist,
        )


def test_cart_to_sphere():

    pfunc = geometry.cart_to_sphere
    args_list = [
        (None, None, apu.m),
        (None, None, apu.m),
        (None, None, apu.m),
        ]
    check_astro_quantities(pfunc, args_list)

    x = Quantity([0., 20., 60., 40., 80.], apu.m)
    y = Quantity([10., 20., 30., 140., 350.], apu.m)
    z = Quantity([-40., 200., 30.1, 130., 320.], apu.m)

    r = Quantity([
        41.23105626, 201.99009877, 73.52557378, 195.19221296,
        480.93658626
        ], apu.m)
    theta = Quantity([
        -75.96375653, 81.95053302, 24.16597925, 41.75987004,
        41.71059314
        ], apu.deg)
    phi = Quantity([90., 45., 26.56505118, 74.0546041, 77.12499844], apu.deg)

    _r, _theta, _phi = pfunc(x, y, z)
    assert_quantity_allclose(_r, r)
    assert_quantity_allclose(_theta, theta)
    assert_quantity_allclose(_phi, phi)


def test_sphere_to_cart():

    pfunc = geometry.sphere_to_cart
    args_list = [
        (None, None, apu.m),
        (-90, 90, apu.deg),
        (None, None, apu.deg),
        ]
    check_astro_quantities(pfunc, args_list)

    x = Quantity([0., 20., 60., 40., 80.], apu.m)
    y = Quantity([10., 20., 30., 140., 350.], apu.m)
    z = Quantity([-40., 200., 30.1, 130., 320.], apu.m)

    r = Quantity([
        41.23105626, 201.99009877, 73.52557378, 195.19221296,
        480.93658626
        ], apu.m)
    theta = Quantity([
        -75.96375653, 81.95053302, 24.16597925, 41.75987004,
        41.71059314
        ], apu.deg)
    phi = Quantity([90., 45., 26.56505118, 74.0546041, 77.12499844], apu.deg)

    _x, _y, _z = pfunc(r, theta, phi)
    assert_quantity_allclose(_x, x, atol=1.e-8 * apu.m)
    assert_quantity_allclose(_y, y)
    assert_quantity_allclose(_z, z)
