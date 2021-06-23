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
from astropy.utils.data import get_pkg_data_filename
from ... import conversions as cnv
from ... import geometry
from ...utils import check_astro_quantities
from astropy.utils.misc import NumpyRNGContext


def produce_rotmat_test_cases():

    with NumpyRNGContext(1):
        # initial (temporary) axis values
        rx_i, ry_i, rz_i = np.random.uniform(-1, 1, (3, 50))
        angle_i = np.random.uniform(-180., 180., 50)

    Rs_i = geometry.geometry._rotmat_from_rotaxis(rx_i, ry_i, rz_i, angle_i)

    rx, ry, rz, angle = geometry.geometry._rotaxis_from_rotmat(Rs_i)
    # Note that the rotaxis_from_rotmat result is not unique, therefore
    # we have to calculate the Rs again from the returned values and
    # compare with the original Rs

    Rs = geometry.geometry._rotmat_from_rotaxis(rx, ry, rz, angle)

    assert np.allclose(Rs_i, Rs)

    a3_xyz, a2_xyz, a1_xyz = geometry.geometry._eulerangle_from_rotmat(
        Rs, etype='xyz'
        )

    Rxyz = geometry.multiply_matrices(
        geometry.geometry._Rz(a3_xyz),
        geometry.geometry._Ry(a2_xyz),
        geometry.geometry._Rx(a1_xyz),
        )

    assert np.allclose(Rs, Rxyz)

    a3_zxz, a2_zxz, a1_zxz = geometry.geometry._eulerangle_from_rotmat(
        Rs, etype='zxz'
        )

    Rzxz = geometry.multiply_matrices(
        geometry.geometry._Rz(a3_zxz),
        geometry.geometry._Rx(a2_zxz),
        geometry.geometry._Rz(a1_zxz),
        )

    assert np.allclose(Rs, Rzxz)

    np.savez(
        '/tmp/rotmat_cases.npz',
        rx=rx, ry=ry, rz=rz, angle=angle,
        Rs=Rs, Rxyz=Rxyz, Rzxz=Rzxz,
        a1_xyz=a1_xyz, a2_xyz=a2_xyz, a3_xyz=a3_xyz,
        a1_zxz=a1_zxz, a2_zxz=a2_zxz, a3_zxz=a3_zxz,
        )


# Warning: if you want to produce new test cases (replacing the old ones)
# you better make sure, that everything is 100% correct
# produce_rotmat_test_cases()


TOL_KWARGS = {'atol': 1.e-4, 'rtol': 1.e-4}
R_X = np.array([
    [[[1., 0., 0.],
      [0., 1., 0.],
      [0., 0., 1.]],
     [[1., 0., 0.],
      [0., 0.9961947, -0.08715574],
      [0., 0.08715574, 0.9961947]]],
    [[[1., 0., 0.],
      [0., 0.98480775, -0.17364818],
      [0., 0.17364818, 0.98480775]],
     [[1., 0., 0.],
      [0., 0.96592583, -0.25881905],
      [0., 0.25881905, 0.96592583]]]
    ])
R_Y = np.array([
    [[[1., 0., 0.],
      [0., 1., 0.],
      [0., 0., 1.]],
     [[0.9961947, 0., 0.08715574],
      [0., 1., 0.],
      [-0.08715574, 0., 0.9961947]]],
    [[[0.98480775, 0., 0.17364818],
      [0., 1., 0.],
      [-0.17364818, 0., 0.98480775]],
     [[0.96592583, 0., 0.25881905],
      [0., 1., 0.],
      [-0.25881905, 0., 0.96592583]]]
    ])
R_Z = np.array([
    [[[1., 0., 0.],
      [0., 1., 0.],
      [0., 0., 1.]],
     [[0.9961947, -0.08715574, 0.],
      [0.08715574, 0.9961947, 0.],
      [0., 0., 1.]]],
    [[[0.98480775, -0.17364818, 0.],
      [0.17364818, 0.98480775, 0.],
      [0., 0., 1.]],
     [[0.96592583, -0.25881905, 0.],
      [0.25881905, 0.96592583, 0.],
      [0., 0., 1.]]]
    ])


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


def test_rotation_matrix_generation():

    angle = Quantity([[0., 5.], [10., 15.]], apu.deg)

    for k, R in zip(['Rx', 'Ry', 'Rz'], [R_X, R_Y, R_Z]):

        pfunc = getattr(geometry, k)
        args_list = [
            (None, None, apu.deg),
            ]
        check_astro_quantities(pfunc, args_list)

        _R = pfunc(angle)
        assert_allclose(_R, R)


def test_rotation_matrix_righthandedness():

    Rx = geometry.Rx(90 * apu.deg)
    Ry = geometry.Ry(90 * apu.deg)
    Rz = geometry.Rz(90 * apu.deg)

    e_x = np.array([1., 0., 0.])
    e_y = np.array([0., 1., 0.])
    e_z = np.array([0., 0., 1.])

    assert_allclose(np.matmul(Rx, e_y), e_z, atol=1e-12)
    assert_allclose(np.matmul(Rx, e_z), -e_y, atol=1e-12)
    assert_allclose(np.matmul(Ry, e_x), -e_z, atol=1e-12)
    assert_allclose(np.matmul(Ry, e_z), e_x, atol=1e-12)
    assert_allclose(np.matmul(Rz, e_x), e_y, atol=1e-12)
    assert_allclose(np.matmul(Rz, e_y), -e_x, atol=1e-12)


def test_multiply_matrices():

    assert_allclose(
        geometry.multiply_matrices(R_X[0, 1], R_Y[1, 1], R_Z[1, 0]),
        np.array([
            [0.95125124, -0.16773126, 0.25881905],
            [0.19520226, 0.97714318, -0.08418598],
            [-0.23878265, 0.13060408, 0.96225019]
            ]),
        atol=1.e-6
        )

    assert_allclose(
        geometry.multiply_matrices(R_X[0, 1], R_X[1, :]),
        np.array([
            [[1., 0., 0.],
             [0., 0.96592583, -0.25881904],
             [0., 0.25881904, 0.96592583]],
            [[1., 0., 0.],
             [0., 0.93969263, -0.34202015],
             [0., 0.34202015, 0.93969263]]
            ]),
        atol=1.e-6
        )


def test_rotmat_from_rotaxis():

    pfunc = geometry.rotmat_from_rotaxis
    args_list = [
        (None, None, apu.m),
        (None, None, apu.m),
        (None, None, apu.m),
        (None, None, apu.deg),
        ]
    check_astro_quantities(pfunc, args_list)

    dat = np.load(get_pkg_data_filename('data/rotmat_cases.npz'))

    rx, ry, rz, angle, Rs = (
        dat[k] for k in ['rx', 'ry', 'rz', 'angle', 'Rs']
        )

    Rs2 = pfunc(rx * apu.m, ry * apu.m, rz * apu.m, angle * apu.deg)
    assert np.allclose(Rs2, Rs)

    rx, ry, rz, angle = (
        dat[k].reshape((10, 5)) for k in ['rx', 'ry', 'rz', 'angle']
        )

    Rs2 = pfunc(rx * apu.m, ry * apu.m, rz * apu.m, angle * apu.deg)
    assert np.allclose(Rs2, Rs.reshape((10, 5, 3, 3)))


def test_rotaxis_from_rotmat():

    pfunc = geometry.rotaxis_from_rotmat

    dat = np.load(get_pkg_data_filename('data/rotmat_cases.npz'))

    rx, ry, rz, angle, Rs = (
        dat[k] for k in ['rx', 'ry', 'rz', 'angle', 'Rs']
        )

    rx2, ry2, rz2, angle2 = pfunc(Rs)
    assert_quantity_allclose(rx2, rx * apu.m)
    assert_quantity_allclose(ry2, ry * apu.m)
    assert_quantity_allclose(rz2, rz * apu.m)
    assert_quantity_allclose(angle2, angle * apu.deg)

    rx, ry, rz, angle = (
        dat[k].reshape((10, 5)) for k in ['rx', 'ry', 'rz', 'angle']
        )

    rx2, ry2, rz2, angle2 = pfunc(Rs.reshape((10, 5, 3, 3)))
    assert_quantity_allclose(rx2, rx * apu.m)
    assert_quantity_allclose(ry2, ry * apu.m)
    assert_quantity_allclose(rz2, rz * apu.m)
    assert_quantity_allclose(angle2, angle * apu.deg)


def test_eulerangle_from_rotmat():

    pfunc = geometry.eulerangle_from_rotmat

    dat = np.load(get_pkg_data_filename('data/rotmat_cases.npz'))

    a1_xyz, a2_xyz, a3_xyz, Rs = (
        dat[k] for k in ['a1_xyz', 'a2_xyz', 'a3_xyz', 'Rs']
        )

    a3_xyz2, a2_xyz2, a1_xyz2 = pfunc(Rs, etype='xyz')
    assert_quantity_allclose(a1_xyz2, a1_xyz * apu.deg)
    assert_quantity_allclose(a2_xyz2, a2_xyz * apu.deg)
    assert_quantity_allclose(a3_xyz2, a3_xyz * apu.deg)

    a1_xyz, a2_xyz, a3_xyz = (
        dat[k].reshape((10, 5)) for k in ['a1_xyz', 'a2_xyz', 'a3_xyz']
        )

    a3_xyz2, a2_xyz2, a1_xyz2 = pfunc(Rs.reshape((10, 5, 3, 3)), etype='xyz')
    assert_quantity_allclose(a1_xyz2, a1_xyz * apu.deg)
    assert_quantity_allclose(a2_xyz2, a2_xyz * apu.deg)
    assert_quantity_allclose(a3_xyz2, a3_xyz * apu.deg)

    a1_zxz, a2_zxz, a3_zxz, Rs = (
        dat[k] for k in ['a1_zxz', 'a2_zxz', 'a3_zxz', 'Rs']
        )

    a3_zxz2, a2_zxz2, a1_zxz2 = pfunc(Rs, etype='zxz')
    assert_quantity_allclose(a1_zxz2, a1_zxz * apu.deg)
    assert_quantity_allclose(a2_zxz2, a2_zxz * apu.deg)
    assert_quantity_allclose(a3_zxz2, a3_zxz * apu.deg)

    a1_zxz, a2_zxz, a3_zxz = (
        dat[k].reshape((10, 5)) for k in ['a1_zxz', 'a2_zxz', 'a3_zxz']
        )

    a3_zxz2, a2_zxz2, a1_zxz2 = pfunc(Rs.reshape((10, 5, 3, 3)), etype='zxz')
    assert_quantity_allclose(a1_zxz2, a1_zxz * apu.deg)
    assert_quantity_allclose(a2_zxz2, a2_zxz * apu.deg)
    assert_quantity_allclose(a3_zxz2, a3_zxz * apu.deg)
