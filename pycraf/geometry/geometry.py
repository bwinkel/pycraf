#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

from astropy import units as apu
import numpy as np
from .. import utils


__all__ = [
    'true_angular_distance', 'great_circle_bearing',
    'cart_to_sphere', 'sphere_to_cart',
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

    sin_diff_lon = np.sin(np.radians(l2 - l1))
    cos_diff_lon = np.cos(np.radians(l2 - l1))
    sin_lat1 = np.sin(np.radians(b1))
    sin_lat2 = np.sin(np.radians(b2))
    cos_lat1 = np.cos(np.radians(b1))
    cos_lat2 = np.cos(np.radians(b2))

    num1 = cos_lat2 * sin_diff_lon
    num2 = cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_diff_lon
    denominator = sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_diff_lon

    return np.degrees(np.arctan2(
        np.sqrt(num1 ** 2 + num2 ** 2), denominator
        ))


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

    sin_diff_lon = np.sin(np.radians(l2 - l1))
    cos_diff_lon = np.cos(np.radians(l2 - l1))
    sin_lat1 = np.sin(np.radians(b1))
    sin_lat2 = np.sin(np.radians(b2))
    cos_lat1 = np.cos(np.radians(b1))
    cos_lat2 = np.cos(np.radians(b2))

    a = cos_lat2 * sin_diff_lon
    b = cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_diff_lon

    return np.degrees(np.arctan2(a, b))


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

    theta = 90. - theta

    x = r * np.sin(np.radians(theta)) * np.cos(np.radians(phi))
    y = r * np.sin(np.radians(theta)) * np.sin(np.radians(phi))
    z = r * np.cos(np.radians(theta))

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


