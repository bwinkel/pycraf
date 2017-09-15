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

    _r, _phi, _theta = pfunc(x, y, z)
    assert_quantity_allclose(_r, r)
    assert_quantity_allclose(_theta, theta)
    assert_quantity_allclose(_phi, phi)


def test_cart_to_sphere_broadcast():

    pfunc = geometry.cart_to_sphere

    x = Quantity([10., 20.], apu.m)[:, np.newaxis, np.newaxis]
    y = Quantity([30., 40.], apu.m)[np.newaxis, :, np.newaxis]
    z = Quantity([50., 60.], apu.m)[np.newaxis, np.newaxis, :]

    r = Quantity([
        [[59.16079783, 67.82329983],
         [64.80740698, 72.80109889]],
        [[61.64414003, 70.],
         [67.08203932, 74.83314774]]
        ], apu.m)
    theta = Quantity([
        [[57.68846676, 62.20869436],
         [50.49028771, 55.50376271]],
        [[54.20424009, 58.99728087],
         [48.1896851, 53.3007748]]
        ], apu.deg)
    phi = Quantity([
        [[71.56505118, 71.56505118],
         [75.96375653, 75.96375653]],
        [[56.30993247, 56.30993247],
         [63.43494882, 63.43494882]]
        ], apu.deg)

    _r, _phi, _theta = pfunc(x, y, z)
    assert_quantity_allclose(_r, r)
    assert_quantity_allclose(_theta, theta)
    assert_quantity_allclose(_phi, phi)

    phi = Quantity([
        [[71.56505118],
         [75.96375653]],
        [[56.30993247],
         [63.43494882]]
        ], apu.deg)
    _r, _phi, _theta = pfunc(x, y, z, broadcast_arrays=False)
    assert_quantity_allclose(_r, r)
    assert_quantity_allclose(_theta, theta)
    assert_quantity_allclose(_phi, phi)


def test_sphere_to_cart():

    pfunc = geometry.sphere_to_cart
    args_list = [
        (None, None, apu.m),
        (None, None, apu.deg),
        (-90, 90, apu.deg),
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

    _x, _y, _z = pfunc(r, phi, theta)
    assert_quantity_allclose(_x, x, atol=1.e-8 * apu.m)
    assert_quantity_allclose(_y, y)
    assert_quantity_allclose(_z, z)


def test_sphere_to_cart_broadcast():

    pfunc = geometry.sphere_to_cart

    r = Quantity([10, 20], apu.m)[:, np.newaxis, np.newaxis]
    theta = Quantity([20, 30], apu.deg)[np.newaxis, :, np.newaxis]
    phi = Quantity([40, 50], apu.deg)[np.newaxis, np.newaxis, :]

    x = Quantity([
        [[7.1984631, 6.04022774],
         [6.63413948, 5.56670399]],
        [[14.39692621, 12.08045547],
         [13.26827896, 11.13340798]]
        ], apu.m)
    y = Quantity([
        [[6.04022774, 7.1984631],
         [5.56670399, 6.63413948]],
        [[12.08045547, 14.39692621],
         [11.13340798, 13.26827896]]
        ], apu.m)
    z = Quantity([
        [[3.42020143, 3.42020143],
         [5., 5.]],
        [[6.84040287, 6.84040287],
         [10., 10.]]
        ], apu.m)

    _x, _y, _z = pfunc(r, phi, theta)
    assert_quantity_allclose(_x, x, atol=1.e-8 * apu.m)
    assert_quantity_allclose(_y, y)
    assert_quantity_allclose(_z, z)

    z = Quantity([
        [[3.42020143],
         [5.]],
        [[6.84040287],
         [10.]]
        ], apu.m)

    _x, _y, _z = pfunc(r, phi, theta, broadcast_arrays=False)
    assert_quantity_allclose(_x, x, atol=1.e-8 * apu.m)
    assert_quantity_allclose(_y, y)
    assert_quantity_allclose(_z, z)
