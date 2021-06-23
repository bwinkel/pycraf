#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

from functools import reduce
from astropy import units as apu
import numpy as np
from .cygeometry import (
    true_angular_distance_cython, great_circle_bearing_cython
    )
from .. import utils


__all__ = [
    'true_angular_distance', 'great_circle_bearing',
    'cart_to_sphere', 'sphere_to_cart',
    'Rx', 'Ry', 'Rz', 'multiply_matrices',
    'rotaxis_from_rotmat', 'rotmat_from_rotaxis', 'eulerangle_from_rotmat',
    ]


@utils.ranged_quantity_input(
    l1=(None, None, apu.deg),
    b1=(-90, 90, apu.deg),
    l2=(None, None, apu.deg),
    b2=(-90, 90, apu.deg),
    strip_input_units=True, output_unit=apu.deg,
    )
def true_angular_distance(l1, b1, l2, b2):
    '''
    True angular distance between points (l1, b1) and (l2, b2).

    Based on Vincenty formula
    (http://en.wikipedia.org/wiki/Great-circle_distance).
    This was spotted in astropy source code.

    Parameters
    ----------
    l1, b1 : `~astropy.units.Quantity`
        Longitude/Latitude of point 1 [deg]
    l2, b2 : `~astropy.units.Quantity`
        Longitude/Latitude of point 2 [deg]

    Returns
    -------
    adist : `~astropy.units.Quantity`
        True angular distance [deg]
    '''

    # sin_diff_lon = np.sin(np.radians(l2 - l1))
    # cos_diff_lon = np.cos(np.radians(l2 - l1))
    # sin_lat1 = np.sin(np.radians(b1))
    # sin_lat2 = np.sin(np.radians(b2))
    # cos_lat1 = np.cos(np.radians(b1))
    # cos_lat2 = np.cos(np.radians(b2))

    # num1 = cos_lat2 * sin_diff_lon
    # num2 = cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_diff_lon
    # denominator = sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_diff_lon

    # return np.degrees(np.arctan2(
    #     np.sqrt(num1 ** 2 + num2 ** 2), denominator
    #     ))

    return true_angular_distance_cython(l1, b1, l2, b2)


@utils.ranged_quantity_input(
    l1=(None, None, apu.deg),
    b1=(-90, 90, apu.deg),
    l2=(None, None, apu.deg),
    b2=(-90, 90, apu.deg),
    strip_input_units=True, output_unit=apu.deg,
    )
def great_circle_bearing(l1, b1, l2, b2):
    '''
    Great circle bearing between points (l1, b1) and (l2, b2).

    Parameters
    ----------
    l1, b1 : `~astropy.units.Quantity`
        Longitude/Latitude of point 1 [deg]
    l2, b2 : `~astropy.units.Quantity`
        Longitude/Latitude of point 2 [deg]

    Returns
    -------
    bearing : `~astropy.units.Quantity`
        Great circle bearing [deg]
    '''

    # diff_lon_rad = np.radians(l2 - l1)
    # b1_rad, b2_rad = np.radians(b1), np.radians(b2)

    # sin_diff_lon = np.sin(diff_lon_rad)
    # cos_diff_lon = np.cos(diff_lon_rad)
    # sin_lat1 = np.sin(b1_rad)
    # sin_lat2 = np.sin(b2_rad)
    # cos_lat1 = np.cos(b1_rad)
    # cos_lat2 = np.cos(b2_rad)

    # a = cos_lat2 * sin_diff_lon
    # b = cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_diff_lon

    # return np.degrees(np.arctan2(a, b))

    return great_circle_bearing_cython(l1, b1, l2, b2)


def _cart_to_sphere(x, y, z, broadcast_arrays=True):

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = 90 - np.degrees(np.arccos(z / r))
    phi = np.degrees(np.arctan2(y, x))

    if broadcast_arrays:
        r, phi, theta = np.broadcast_arrays(r, phi, theta)

    return r, phi, theta


@utils.ranged_quantity_input(
    x=(None, None, apu.m),
    y=(None, None, apu.m),
    z=(None, None, apu.m),
    strip_input_units=True, output_unit=(apu.m, apu.deg, apu.deg)
    )
def cart_to_sphere(x, y, z, broadcast_arrays=True):
    '''
    Spherical coordinates from Cartesian representation.

    Parameters
    ----------
    x, y, z : `~astropy.units.Quantity`
        Cartesian position [m]

    Returns
    -------
    r : `~astropy.units.Quantity`
        Radial distance [m]
    phi : `~astropy.units.Quantity`
        Azimuth [deg]
    theta : `~astropy.units.Quantity`
        Elevation [deg]
    broadcast_arrays : boolean, optional
        If 'True', output arrays will be broadcasted to dense
        matrices, otherwise the returned arrays will be a
        sparse representation (default: True)

    Notes
    -----
    Unlike with the mathematical definition, `theta` is not the angle
    to the (positive) `z` axis, but the elevation above the `x`-`y` plane.
    '''

    return _cart_to_sphere(x, y, z, broadcast_arrays=broadcast_arrays)


def _sphere_to_cart(r, phi, theta, broadcast_arrays=True):

    c_t, s_t = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    c_p, s_p = np.cos(np.radians(phi)), np.sin(np.radians(phi))

    x = r * c_t * c_p
    y = r * c_t * s_p
    z = r * s_t

    if broadcast_arrays:
        x, y, z = np.broadcast_arrays(x, y, z)

    return x, y, z


@utils.ranged_quantity_input(
    r=(None, None, apu.m),
    theta=(-90, 90, apu.deg),
    phi=(None, None, apu.deg),
    strip_input_units=True, output_unit=(apu.m, apu.m, apu.m)
    )
def sphere_to_cart(r, phi, theta, broadcast_arrays=True):
    '''
    Spherical coordinates from Cartesian representation.

    Parameters
    ----------
    r : `~astropy.units.Quantity`
        Radial distance [m]
    phi : `~astropy.units.Quantity`
        Azimuth [deg]
    theta : `~astropy.units.Quantity`
        Elevation [deg]
    broadcast_arrays : boolean, optional
        If 'True', output arrays will be broadcasted to dense
        matrices, otherwise the returned arrays will be a
        sparse representation (default: True)

    Returns
    -------
    x, y, z : `~astropy.units.Quantity`
        Cartesian position [m]

    Notes
    -----
    Unlike with the mathematical definition, `theta` is not the angle
    to the (positive) `z` axis, but the elevation above the `x`-`y` plane.
    '''

    return _sphere_to_cart(r, phi, theta, broadcast_arrays=broadcast_arrays)


def _Rx(angle):

    angle_rad = np.radians(angle)
    sin_a, cos_a = np.sin(angle_rad.flat), np.cos(angle_rad.flat)
    o, z = np.ones_like(sin_a), np.zeros_like(sin_a)

    R = np.array([[o, z, z], [z, cos_a, -sin_a], [z, sin_a, cos_a]])

    new_sh = angle_rad.shape + (3, 3)
    return np.moveaxis(R, [0, 1, 2], [1, 2, 0]).reshape(new_sh)


def _Ry(angle):

    angle_rad = np.radians(angle)
    sin_a, cos_a = np.sin(angle_rad.flat), np.cos(angle_rad.flat)
    o, z = np.ones_like(sin_a), np.zeros_like(sin_a)

    R = np.array([[cos_a, z, sin_a], [z, o, z], [-sin_a, z, cos_a]])

    new_sh = angle_rad.shape + (3, 3)
    return np.moveaxis(R, [0, 1, 2], [1, 2, 0]).reshape(new_sh)


def _Rz(angle):

    angle_rad = np.radians(angle)
    sin_a, cos_a = np.sin(angle_rad.flat), np.cos(angle_rad.flat)
    o, z = np.ones_like(sin_a), np.zeros_like(sin_a)

    R = np.array([[cos_a, -sin_a, z], [sin_a, cos_a, z], [z, z, o]])

    new_sh = angle_rad.shape + (3, 3)
    return np.moveaxis(R, [0, 1, 2], [1, 2, 0]).reshape(new_sh)


@utils.ranged_quantity_input(
    angle=(None, None, apu.deg),
    strip_input_units=True
    )
def Rx(angle):
    '''
    Construct rotation matrix about x-axis.

    Parameters
    ----------
    angle : `~astropy.units.Quantity`
        Rotation angle [deg]

    Returns
    -------
    Rx : `~numpy.array`
        Rotation matrix [no units!]

    Notes
    -----
    Broadcasting is supported. Rotation matrices can be (matrix-multiplied)
    with `~numpy.dot`, however, if you want to multiply stacks of matrices
    use the new `~numpy.matmul` function.
    '''

    return _Rx(angle)


@utils.ranged_quantity_input(
    angle=(None, None, apu.deg),
    strip_input_units=True
    )
def Ry(angle):
    '''
    Construct rotation matrix about y-axis.

    Parameters
    ----------
    angle : `~astropy.units.Quantity`
        Rotation angle [deg]

    Returns
    -------
    Ry : `~numpy.array`
        Rotation matrix [no units!]

    Notes
    -----
    Broadcasting is supported. Rotation matrices can be (matrix-multiplied)
    with `~numpy.dot`, however, if you want to multiply stacks of matrices
    use the new `~numpy.matmul` function.
    '''

    return _Ry(angle)


@utils.ranged_quantity_input(
    angle=(None, None, apu.deg),
    strip_input_units=True
    )
def Rz(angle):
    '''
    Construct rotation matrix about z-axis.

    Parameters
    ----------
    angle : `~astropy.units.Quantity`
        Rotation angle [deg]

    Returns
    -------
    Rz : `~numpy.array`
        Rotation matrix [no units!]

    Notes
    -----
    Broadcasting is supported. Rotation matrices can be (matrix-multiplied)
    with `~numpy.dot`, however, if you want to multiply stacks of matrices
    use the new `~numpy.matmul` function.
    '''

    return _Rz(angle)


def multiply_matrices(*matrices):
    '''
    Matrix-multiply the matrices in the given list.

    Parameters
    ----------
    matrices : list of `~numpy.array`
        List of (stacks of) rotation matrices. The order of the
        multiplication is `R = R1.R2.R3`, if `*matrices = (R1, R2, R3)`.
        (This means, that `R3` is applied first.)

    Returns
    -------
    R : `~numpy.array`
        Resulting rotation matrix [no units!]

    Notes
    -----
    Broadcasting is supported, i.e., each element of `matrices` can be
    a stack of rotation matrices (that are of course not multiplied
    internally).
    '''

    # Note, rotation matrices are associative, therefore we can replace
    # return reduce(lambda a, b: np.matmul(b, a), matrices[::-1])
    # by
    return reduce(np.matmul, matrices)


def _rotmat_from_rotaxis(rotax_x, rotax_y, rotax_z, angle_deg):

    sh = np.broadcast(angle_deg, rotax_x, rotax_y, rotax_z).shape
    angle_rad = np.radians(angle_deg)

    rot_axes = np.array(np.broadcast_arrays(
        np.atleast_1d(rotax_x),
        np.atleast_1d(rotax_y),
        np.atleast_1d(rotax_z),
        )).astype(np.float64)

    d = rot_axes
    d /= np.linalg.norm(d, axis=0)

    W = np.zeros(rot_axes.shape[1:] + (3, 3))
    W[..., 0, 1] = -d[2]
    W[..., 0, 2] = d[1]
    W[..., 1, 0] = d[2]
    W[..., 1, 2] = -d[0]
    W[..., 2, 0] = -d[1]
    W[..., 2, 1] = d[0]
    W2 = np.matmul(W, W)

    _a = angle_rad[..., np.newaxis, np.newaxis]
    eye = np.eye(3, dtype=np.float64)
    mtxs = eye + np.sin(_a) * W + 2. * np.sin(_a / 2.) ** 2 * W2

    return mtxs.reshape(sh + (3, 3))


@utils.ranged_quantity_input(
    rotax_x=(None, None, apu.m),
    rotax_y=(None, None, apu.m),
    rotax_z=(None, None, apu.m),
    angle=(None, None, apu.deg),
    strip_input_units=True
    )
def rotmat_from_rotaxis(rotax_x, rotax_y, rotax_z, angle):
    '''
    Construct rotation matrix from cartesian rotation axis and angle.

    Parameters
    ----------
    rotax_{x,y,z} : `~astropy.units.Quantity`
        Cartesian components (x, y, z) of rotation axis [m]
    angle : `~astropy.units.Quantity`
        Rotation angle [deg]

    Returns
    -------
    R : `~numpy.array`
        Rotation matrix [no units!]

    Notes
    -----
    Broadcasting is supported. The rotation axis vector will be
    normalized in the function (otherwise it's magnitude would change
    the effective rotation angle).
    '''

    return _rotmat_from_rotaxis(rotax_x, rotax_y, rotax_z, angle)


def _rotaxis_from_rotmat(R):

    # see Wikipedia (https://en.wikipedia.org/wiki/Axis–angle_representation)
    R = np.asarray(R)

    rot_angle = np.arccos(
        0.5 * (np.trace(R, axis1=-2, axis2=-1) - 1)
        )
    norm = 1. / 2. / np.sin(rot_angle)

    rotax_x = norm * (R[..., 2, 1] - R[..., 1, 2])
    rotax_y = norm * (R[..., 0, 2] - R[..., 2, 0])
    rotax_z = norm * (R[..., 1, 0] - R[..., 0, 1])

    return rotax_x, rotax_y, rotax_z, np.degrees(rot_angle)


@utils.ranged_quantity_input(
    output_unit=(apu.m, apu.m, apu.m, apu.deg)
    )
def rotaxis_from_rotmat(R):
    '''
    Cartesian rotation axis and angle from rotation matrix.

    Parameters
    ----------
    R : `~numpy.array`
        Rotation matrix [no units!]

    Returns
    -------
    rotax_{x,y,z} : `~astropy.units.Quantity`
        Cartesian components (x, y, z) of rotation axis [m]
    angle : `~astropy.units.Quantity`
        Rotation angle [deg]

    Notes
    -----
    Broadcasting is supported. The returned rotation axis vector is
    normalized (otherwise it's magnitude would change the effective
    rotation angle).
    '''

    return _rotaxis_from_rotmat(R)


def _eulerangle_from_rotmat(R, etype='xyz'):
    '''
    Euler angles from a rotation matrix.
    '''

    # see https://www.geometrictools.com/Documentation/EulerAngles.pdf

    assert etype in ['xyz', 'zxz']

    R = np.asarray(R)
    assert R.shape[-2:] == (3, 3)

    do_squeeze = False
    if R.ndim < 3:
        R = R.reshape((1, 3, 3))
        do_squeeze = True

    alpha_1 = np.empty(R.shape[:-2], dtype=np.float64)
    alpha_2 = np.empty(R.shape[:-2], dtype=np.float64)
    alpha_3 = np.empty(R.shape[:-2], dtype=np.float64)
    if etype == 'xyz':

        mask1 = R[..., 2, 0] == 1
        sub_R = R[mask1]
        alpha_1[mask1] = 0
        alpha_2[mask1] = np.arctan2(-sub_R[:, 1, 2], sub_R[:, 1, 1])
        alpha_3[mask1] = -np.pi / 2

        mask2 = R[..., 2, 0] == -1
        sub_R = R[mask2]
        alpha_1[mask2] = 0
        alpha_2[mask2] = np.arctan2(-sub_R[:, 1, 2], sub_R[:, 1, 1])
        alpha_3[mask2] = np.pi / 2

        mask = (~mask1) & (~mask2)
        sub_R = R[mask]
        alpha_1[mask] = np.arctan2(sub_R[:, 2, 1], sub_R[:, 2, 2])
        alpha_2[mask] = np.arcsin(-sub_R[:, 2, 0])
        alpha_3[mask] = np.arctan2(sub_R[:, 1, 0], sub_R[:, 0, 0])

    elif etype == 'zxz':

        mask1 = R[..., 2, 2] == 1
        sub_R = R[mask1]
        alpha_1[mask1] = 0
        alpha_2[mask1] = 0
        alpha_3[mask1] = np.arctan2(-sub_R[:, 0, 1], sub_R[:, 0, 0])

        mask2 = R[..., 2, 2] == -1
        sub_R = R[mask2]
        alpha_1[mask2] = 0
        alpha_2[mask2] = np.pi
        alpha_3[mask2] = -np.arctan2(-sub_R[:, 1, 2], sub_R[:, 1, 1])

        mask = (~mask1) & (~mask2)
        sub_R = R[mask]
        alpha_1[mask] = np.arctan2(sub_R[:, 2, 0], sub_R[:, 2, 1])
        alpha_2[mask] = np.arccos(sub_R[:, 2, 2])
        alpha_3[mask] = np.arctan2(sub_R[:, 0, 2], -sub_R[:, 1, 2])

    alpha_1, alpha_2, alpha_3 = (
        np.degrees(alpha_1), np.degrees(alpha_2), np.degrees(alpha_3)
        )
    if do_squeeze:
        alpha_1, alpha_2, alpha_3 = (
            alpha_1.squeeze(), alpha_2.squeeze(), alpha_3.squeeze()
            )

    return alpha_3, alpha_2, alpha_1


@utils.ranged_quantity_input(
    output_unit=(apu.deg, apu.deg, apu.deg)
    )
def eulerangle_from_rotmat(R, etype='xyz'):
    '''
    Cartesian rotation axis and angle from rotation matrix.

    Parameters
    ----------
    R : `~numpy.array`
        Rotation matrix [no units!]
    etype : str, optional, 'xyz' or 'zxz'
        Desired Euler-angle ordering. (default: 'xyz')

        The ordering 'xyz' refers to the case where first a rotation
        about the x-axis is performed, then about y and last about
        z-axis, i.e., R = Rz(alpha3).Ry(alpha2).Rx(alpha1). Likewise
        for ordering 'zxz' it is R = Rz(alpha3).Rx(alpha2).Rz(alpha1)

    Returns
    -------
    angle_{1, 2, 3} : `~astropy.units.Quantity`
        Euler rotation angles [deg]

    Notes
    -----
    Broadcasting is supported. Note the returned Euler angles are
    not unique.
    '''

    return _eulerangle_from_rotmat(R, etype=etype)
