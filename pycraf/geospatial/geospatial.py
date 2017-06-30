#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

from functools import partial, lru_cache
# import numpy as np
import numbers
import pyproj


# Note: UTM zones extend over 6d in longitude and a full hemisphere in
# latitude. However, the NATO system further refines the zones by
# putting them into latitude zones, to form a grid. pyproj uses the former
# approach, though. See "https://en.wikipedia.org/wiki/
# Universal_Transverse_Mercator_coordinate_system" for further details.
# In the following, we use "N" and "S" to make a distinction between
# Northern and Southern hemisphere. Don't mix this up with the grid
# nomenclature (where xxS is a cell on the Northern hemisphere...).


__all__ = [
    'utm_to_wgs84', 'wgs84_to_utm', 'utm_to_wgs84_32N',
    'etrs89_to_wgs84', 'wgs84_to_etrs89',
    ]


@lru_cache(maxsize=16, typed=True)
def _create_proj(sys1, sys2, zone=None, south=False):
    '''
    Helper function to create and cache pyproj.Proj instances.
    '''

    if sys1 == 'UTM' and sys2 == 'WGS84':

        if not isinstance(zone, numbers.Integral):
            raise TypeError('zone must be an integer')

        _proj_str = (
            '+proj=utm +ellps=WGS84 +datum=WGS84 +units=m +no_defs '
            '+zone={:d}'.format(zone)
            )
        if south:
            _proj_str += '+south'

    elif sys1 == 'ETRS89' and sys2 == 'WGS84':

        _proj_str = (
            '+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 '
            '+ellps=GRS80 +units=m +no_defs'
            )
    else:
        raise AssertionError(
            'requested conversion not supported {} {}'.format(sys1, sys2)
            )

    return pyproj.Proj(_proj_str)


def utm_to_wgs84(ulon, ulat, zone, south=False):
    '''
    Convert UTM coordinates to GPS/WGS84.

    Parameters
    ----------
    ulon, ulat - UTM longitude and latitude [deg]
    zone - UTM zone (e.g., 32 for Effelsberg, with south == False)
    south - set to True if on southern hemisphere

    Returns
    -------
    glon, glat - GPS/WGS84 longitude and latitude [deg]

    Notes
    -----
    Uses
        +proj=utm +zone=xx +ellps=WGS84 +datum=WGS84 +units=m +no_defs [+south]
    for pyproj setup. Only one zone per function call is allowed.
    '''

    _proj = _create_proj('UTM', 'WGS84', zone, south)

    return _proj(ulon, ulat, inverse=True)


def wgs84_to_utm(glon, glat, zone, south=False):
    '''
    Convert GPS/WGS84 coordinates to UTM.

    Parameters
    ----------
    glon, glat - GPS/WGS84 longitude and latitude [deg]
    zone - UTM zone (e.g., 32 for Effelsberg)

    Returns
    -------
    ulon, ulat - UTM longitude and latitude [deg]

    Notes
    -----
    Uses
        +proj=utm +zone=xx +ellps=WGS84 +datum=WGS84 +units=m +no_defs [+south]
    for pyproj setup.
    '''

    _proj = _create_proj('UTM', 'WGS84', zone, south)

    return _proj(glon, glat, inverse=False)


# This is for Western Germany (Effelsberg)
utm_to_wgs84_32N = partial(utm_to_wgs84, zone=32, south=False)
wgs84_to_utm_32N = partial(wgs84_to_utm, zone=32, south=False)


def etrs89_to_wgs84(elon, elat):
    '''
    Convert ETSR89 coordinates to GPS/WGS84.

    ETRS89 is the European Terrestrial Reference System.
    (Using a Lambert Equal Area projection.)

    Parameters
    ----------
    elon, elat - ETRS89 longitude and latitude [deg]

    Returns
    -------
    glon, glat - GPS/WGS84 longitude and latitude [deg]

    Notes
    -----
    Uses
        +proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000
        +ellps=GRS80 +units=m +no_defs
    for pyproj setup.
    '''

    _proj = _create_proj('ETRS89', 'WGS84')

    return _proj(elon, elat, inverse=True)


def wgs84_to_etrs89(glon, glat):
    '''
    Convert GPS/WGS84 coordinates to ETSR89.

    ETRS89 is the European Terrestrial Reference System.
    (Using a Lambert Equal Area projection.)

    Parameters
    ----------
    glon, glat - GPS/WGS84 longitude and latitude [deg]

    Returns
    -------
    elon, elat - ETRS89 longitude and latitude [deg]

    Notes
    -----
    Uses
        +proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000
        +ellps=GRS80 +units=m +no_defs
    for pyproj setup.
    '''

    _proj = _create_proj('ETRS89', 'WGS84')

    return _proj(glon, glat, inverse=False)


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
