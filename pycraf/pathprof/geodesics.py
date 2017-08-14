#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

from astropy import units as apu
import numpy as np
from .. import utils
from .cygeodesics import inverse_cython, direct_cython, area_wgs84_cython


__all__ = ['geoid_inverse', 'geoid_direct', 'geoid_area']


@utils.ranged_quantity_input(
    lon1=(-np.pi, np.pi, apu.rad),
    lat1=(-np.pi / 2, np.pi / 2, apu.rad),
    lon2=(-np.pi, np.pi, apu.rad),
    lat2=(-np.pi / 2, np.pi / 2, apu.rad),
    strip_input_units=True,
    output_unit=(apu.m, apu.rad, apu.rad)
    )
def geoid_inverse(
        lon1, lat1,
        lon2, lat2,
        eps=1.e-12,  # corresponds to approximately 0.06mm
        maxiter=50,
        ):
    '''
    Solve inverse Geodesics problem using Vincenty's formulae.

    Using an iterative approach, the distance and relative bearings between
    two points (P1, and P2) on the Geoid (Earth ellipsoid) are determined;
    see also `Wikipedia <https://en.wikipedia.org/wiki/Vincenty's_formulae>`_.

    Parameters
    ----------
    lon1 : `~astropy.units.Quantity`
        Geographic longitude of P1 [rad]
    lat1 : `~astropy.units.Quantity`
        Geographic latitude of P1 [rad]
    lon2 : `~astropy.units.Quantity`
        Geographic longitude of P2 [rad]
    lat2 : `~astropy.units.Quantity`
        Geographic latitude of P2 [rad]
    eps : float, optional
        Accuracy of calculation (default: 1.e-12)
    maxiter : int, optional
        Maximum number of iterations to perform (default: 50)

    Returns
    -------
    distance : `~astropy.units.Quantity`
        Distance between P1 and P2 [m]
    bearing1 : `~astropy.units.Quantity`
        Start bearing [rad]
    bearing2 : `~astropy.units.Quantity`
        Back-bearing [rad]

    Notes
    -----
    The iteration will stop if either the desired accuracy (`eps`) is reached
    or the number of iterations exceeds `maxiter`.
    '''

    return inverse_cython(lon1, lat1, lon2, lat2, eps=eps, maxiter=maxiter)


@utils.ranged_quantity_input(
    lon1=(-np.pi, np.pi, apu.rad),
    lat1=(-np.pi / 2, np.pi / 2, apu.rad),
    bearing1=(-np.pi, np.pi, apu.rad),
    dist=(0.1, None, apu.m),
    strip_input_units=True,
    output_unit=(apu.rad, apu.rad, apu.rad)
    )
def geoid_direct(
        lon1, lat1,
        bearing1, dist,
        eps=1.e-12,  # corresponds to approximately 0.06mm
        maxiter=50,
        ):
    '''
    Solve direct Geodesics problem using Vincenty's formulae.

    From starting point P1, given a start bearing, find
    point P2 located at a certain distance from P1 on the Geoid (Earth
    ellipsoid). As for the inverse problem, an iterative approach is used;
    see also `Wikipedia <https://en.wikipedia.org/wiki/Vincenty's_formulae>`_.

    Parameters
    ----------
    lon1 : `~astropy.units.Quantity`
        Geographic longitude of P1 [rad]
    lat1 : `~astropy.units.Quantity`
        Geographic latitude of P1 [rad]
    bearing1 : `~astropy.units.Quantity`
        Start bearing [rad]
    distance : `~astropy.units.Quantity`
        Distance between P1 and P2 [m]
    eps : float, optional
        Accuracy of calculation (default: 1.e-12)
    maxiter : int, optional
        Maximum number of iterations to perform (default: 50)

    Returns
    -------
    lon2 : `~astropy.units.Quantity`
        Geographic longitude of P2 [rad]
    lat2 : `~astropy.units.Quantity`
        Geographic latitude of P2 [rad]
    bearing2 : `~astropy.units.Quantity`
        Back-bearing [rad]

    Notes
    -----
    The iteration will stop if either the desired accuracy (`eps`) is reached
    or the number of iterations exceeds `maxiter`.
    '''

    return direct_cython(
        lon1, lat1, bearing1, dist, eps=eps, maxiter=maxiter, wrap=True
        )


@utils.ranged_quantity_input(
    lon1=(-np.pi, np.pi, apu.rad),
    lon2=(-np.pi, np.pi, apu.rad),
    lat1=(-np.pi / 2, np.pi / 2, apu.rad),
    lat2=(-np.pi / 2, np.pi / 2, apu.rad),
    strip_input_units=True,
    output_unit=(apu.m ** 2)
    )
def geoid_area(lon1, lon2, lat1, lat2):
    '''
    Calculate WGS84 surface area over interval [lon1, lon2] and [lat1, lat2].

    Parameters
    ----------
    lon1 : `~astropy.units.Quantity`
        Geographic longitude of lower bound [rad]
    lon2 : `~astropy.units.Quantity`
        Geographic longitude of upper bound [rad]
    lat1 : `~astropy.units.Quantity`
        Geographic latitude of lower bound [rad]
    lat2 : `~astropy.units.Quantity`
        Geographic latitude of upper bound [rad]

    Returns
    -------
    area : `~astropy.units.Quantity`
        Surface area in given interval [m^2]

    Notes
    -----
    This was adapted from a thread on `math.stackexchange.com <https://math.stackexchange.com/questions/1379341/how-to-find-the-surface-area-of-revolution-of-an-ellipsoid-from-ellipse-rotating>`__.
    '''

    return area_wgs84_cython(lon1, lon2, lat1, lat2)


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
