#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

from functools import partial, lru_cache
# import numpy as np
import numbers
from astropy import units as apu
import pyproj
from .. import utils


# Note: UTM zones extend over 6d in longitude and a full hemisphere in
# latitude. However, the NATO system further refines the zones by
# putting them into latitude zones, to form a grid. pyproj uses the former
# approach, though. See "https://en.wikipedia.org/wiki/
# Universal_Transverse_Mercator_coordinate_system" for further details.
# In the following, we use "N" and "S" to make a distinction between
# Northern and Southern hemisphere. Don't mix this up with the grid
# nomenclature (where xxS is a cell on the Northern hemisphere...).


__all__ = [
    'utm_to_wgs84', 'wgs84_to_utm',  # 'utm_to_wgs84_32N',
    'etrs89_to_wgs84', 'wgs84_to_etrs89',
    # 'itrf2005_to_wgs84',
    # 'wgs84_to_itrf2005',
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
    elif sys1 == 'ITRF05' and sys2 == 'WGS84':

        _proj_str = (
            '+proj=longlat +datum=WGS84 +no_defs'
            '+to +proj=geocent +ellps=GRS80 +units=m +no_defs'
            )
    else:
        raise AssertionError(
            'requested conversion not supported {} {}'.format(sys1, sys2)
            )

    return pyproj.Proj(_proj_str)


@utils.ranged_quantity_input(
    ulon=(None, None, apu.m),
    ulat=(None, None, apu.m),
    strip_input_units=True, output_unit=(apu.deg, apu.deg)
    )
def utm_to_wgs84(ulon, ulat, zone, south=False):
    '''
    Convert UTM coordinates to GPS/WGS84.

    Parameters
    ----------
    ulon, ulat : `~astropy.units.Quantity`
        UTM longitude and latitude [m]
    zone : int
        UTM zone (e.g., 32 for longitudes 6 E to 12 E); see `Wikipedia
        <https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system>`_
        for more information on the zones
    south : bool
        Hemisphere; set to True for southern hemisphere (default: False)

    Returns
    -------
    glon, glat : `~astropy.units.Quantity`
        GPS/WGS84 longitude and latitude [deg]

    Notes
    -----
    - Uses

      .. code-block:: bash

          +proj=utm +zone=xx +ellps=WGS84 +datum=WGS84 +units=m +no_defs [+south]

      for `pyproj` setup (inverse transformation). Only one zone per function
      call is allowed.
    - This function uses only the longitudal zone scheme. There is also
      the NATO system, which introduces latitude bands.
    '''

    _proj = _create_proj('UTM', 'WGS84', zone, south)

    return _proj(ulon, ulat, inverse=True)


@utils.ranged_quantity_input(
    glon=(None, None, apu.deg),
    glat=(None, None, apu.deg),
    strip_input_units=True, output_unit=(apu.m, apu.m)
    )
def wgs84_to_utm(glon, glat, zone, south=False):
    '''
    Convert GPS/WGS84 coordinates to UTM.

    Parameters
    ----------
    glon, glat : `~astropy.units.Quantity`
        GPS/WGS84 longitude and latitude [deg]
    zone : int
        UTM zone (e.g., 32 for longitudes 6 E to 12 E); see `Wikipedia
        <https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system>`_
        for more information on the zones
    south : bool
        Hemisphere; set to True for southern hemisphere (default: False)

    Returns
    -------
    ulon, ulat : `~astropy.units.Quantity`
        UTM longitude and latitude [m]

    Notes
    -----
    - Uses

      .. code-block:: bash

          +proj=utm +zone=xx +ellps=WGS84 +datum=WGS84 +units=m +no_defs [+south]

      for `pyproj` setup (forward transformation). Only one zone per function
      call is allowed.
    - This function uses only the longitudal zone scheme. There is also
      the NATO system, which introduces latitude bands.
    '''

    _proj = _create_proj('UTM', 'WGS84', zone, south)

    return _proj(glon, glat, inverse=False)


# This is for Western Germany (Effelsberg)
utm_to_wgs84_32N = partial(utm_to_wgs84, zone=32, south=False)
wgs84_to_utm_32N = partial(wgs84_to_utm, zone=32, south=False)


@utils.ranged_quantity_input(
    elon=(None, None, apu.m),
    elat=(None, None, apu.m),
    strip_input_units=True, output_unit=(apu.deg, apu.deg)
    )
def etrs89_to_wgs84(elon, elat):
    '''
    Convert ETSR89 coordinates to GPS/WGS84.

    ETRS89 is the European Terrestrial Reference System.
    (Using a Lambert Equal Area projection.)

    Parameters
    ----------
    elon, elat : `~astropy.units.Quantity`
        ETRS89 longitude and latitude [m]

    Returns
    -------
    glon, glat : `~astropy.units.Quantity`
        GPS/WGS84 longitude and latitude [deg]

    Notes
    -----
    Uses

    .. code-block:: bash

        +proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000
        +ellps=GRS80 +units=m +no_defs

    for pyproj setup (inverse transformation).
    '''

    _proj = _create_proj('ETRS89', 'WGS84')

    return _proj(elon, elat, inverse=True)


@utils.ranged_quantity_input(
    glon=(None, None, apu.deg),
    glat=(None, None, apu.deg),
    strip_input_units=True, output_unit=(apu.m, apu.m)
    )
def wgs84_to_etrs89(glon, glat):
    '''
    Convert GPS/WGS84 coordinates to ETSR89.

    ETRS89 is the European Terrestrial Reference System.
    (Using a Lambert Equal Area projection.)

    Parameters
    ----------
    glon, glat : `~astropy.units.Quantity`
        GPS/WGS84 longitude and latitude [deg]

    Returns
    -------
    elon, elat : `~astropy.units.Quantity`
        ETRS89 longitude and latitude [m]

    Notes
    -----
    Uses

    .. code-block:: bash

        +proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000
        +ellps=GRS80 +units=m +no_defs

    for pyproj setup (forward transformation).
    '''

    _proj = _create_proj('ETRS89', 'WGS84')

    return _proj(glon, glat, inverse=False)


def _wgs84_to_itrf2005(glon, glat):
    '''
    BROKEN!

    Convert GPS/WGS84 coordinates to ITRF.

    Parameters
    ----------
    glon, glat - GPS/WGS84 longitude and latitude [deg]

    Returns
    -------
    x, y, z - ITRF cartesian coordinates, geocentric [m]

    Notes
    -----
    Uses
        +proj=utm +zone=xx +ellps=WGS84 +datum=WGS84 +units=m +no_defs [+south]
    for pyproj setup.
    '''

    _proj = _create_proj('ITRF05', 'WGS84')

    return _proj(glon, glat, inverse=False)


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
