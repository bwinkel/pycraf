#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

# from functools import partial, lru_cache
import os
import re
import glob
from functools import lru_cache
from astropy import units as apu
import numpy as np
from scipy.interpolate import RegularGridInterpolator
# from geographiclib.geodesic import Geodesic
from . import cygeodesics
from .. import conversions as cnv
from .. import utils


__all__ = [
    'srtm_height_profile', 'srtm_height_map',
    ]


_HGT_RES = 90.  # m; equivalent to 3 arcsec resolution


def _extract_hgt_coords(hgt_name):
    '''
    Extract coordinates from hgt-filename (lower left corner).

    Properly handles EW and NS substrings. Longitude range: -180 .. 179 deg
    '''

    _codes = {'E': 1, 'W': -1, 'N': 1, 'S': -1}

    yc, wy0, xc, wx0 = re.search(
        ".*([NS])(-?\d*)([EW])(\d*).hgt.*", hgt_name
        ).groups()

    return _codes[xc] * int(wx0), _codes[yc] * int(wy0)


def _find_hgt_files(basepath=None):

    if basepath is None:
        try:
            basepath = os.environ['SRTMDATA']
        except KeyError:
            print('Warning, SRTMDATA environment variable not set.')
            return {}

    hgt_files = glob.glob(
        os.path.join(basepath, '**', '*.hgt*'),
        recursive=True
        )
    if len(hgt_files) < 0:
        print('Warning, no SRTM data found.')
        print('Was looking in {} directory.')
        return {}

    hgt_dict = {}
    for hgt_file in hgt_files:
        hgt_name = os.path.basename(hgt_file)
        l1, b1 = _extract_hgt_coords(hgt_name)
        hgt_dict[(l1, b1)] = hgt_file

    return hgt_dict


# search for hgt tiles during module load
HGT_DICT = _find_hgt_files()


@lru_cache(maxsize=30, typed=False)
def _get_tile_data(ilon, ilat):
    # angles in deg

    tile_size = 1201
    dx = dy = 1. / (tile_size - 1)
    x, y = np.ogrid[0:tile_size, 0:tile_size]
    lons, lats = x * dx + ilon, y * dy + ilat

    try:
        hgt_file = HGT_DICT[ilon, ilat]
        tile = np.fromfile(hgt_file, dtype='>i2')
        tile = tile.reshape((tile_size, tile_size))[::-1]
    except KeyError:
        print(list(HGT_DICT.keys()))
        print(
            'warning, no SRTM tile data found for '
            'ilon, ilat = ({}, {})'.format(
                ilon, ilat
                )
            )
        tile = np.zeros((tile_size, tile_size), dtype=np.int16)

    bad_mask = (tile == 32768) | (tile == -32768)
    tile = tile.astype(np.float32)
    tile[bad_mask] = np.nan

    return lons, lats, tile


@lru_cache(maxsize=30, typed=False)
def _get_tile_interpolator(ilon, ilat):
    # angles in deg

    lons, lats, tile = _get_tile_data(ilon, ilat)
    # have to treat NaNs in some way; set to zero for now
    tile = np.nan_to_num(tile)

    _tile_interpolator = RegularGridInterpolator(
        (lons[:, 0], lats[0]), tile.T
        )

    return _tile_interpolator


def _get_interpolated_data(lons, lats):
    # angles in deg, heights in m

    # coordinates could span different tiles, so first get the unique list
    lons = np.atleast_1d(lons)
    lats = np.atleast_1d(lats)

    assert lons.ndim == 1 and lats.ndim == 1

    heights = np.empty(lons.shape, dtype=np.float32)

    ilons = np.floor(lons).astype(np.int32)
    ilats = np.floor(lats).astype(np.int32)

    uilonlats = set((a, b) for a, b in zip(ilons, ilats))

    for uilon, uilat in uilonlats:

        mask = (ilons == uilon) & (ilats == uilat)
        heights[mask] = _get_tile_interpolator(uilon, uilat)(
            (lons[mask], lats[mask])
            )

    return heights


def _srtm_height_profile(lon_t, lat_t, lon_r, lat_r, step):
    # angles in rad; lengths in m

    # first find start bearing (and backward bearing for the curious people)
    # and distance

    # import time
    # t = time.time()
    lon_t_rad, lat_t_rad = np.radians(lon_t), np.radians(lat_t)
    lon_r_rad, lat_r_rad = np.radians(lon_r), np.radians(lat_r),
    distance, bearing_1_rad, bearing_2_rad = cygeodesics.inverse_cython(
        lon_t_rad, lat_t_rad, lon_r_rad, lat_r_rad,
        )
    bearing_1 = np.degrees(bearing_1_rad)
    bearing_2 = np.degrees(bearing_2_rad)
    back_bearing = bearing_2 % 360 - 180

    distances = np.arange(0., distance + step, step)  # [m]

    lons_rad, lats_rad, bearing_2s_rad = cygeodesics.direct_cython(
        lon_t_rad, lat_t_rad, bearing_1_rad, distances
        )
    lons = np.degrees(lons_rad)
    lats = np.degrees(lats_rad)
    bearing_2s = np.degrees(bearing_2s_rad)

    back_bearings = bearing_2s % 360 - 180

    # important: unless the requested resolution is super-fine, we always
    # have to query the raw height profile data using sufficient resolution,
    # to acquire all features
    # only afterwards, we may smooth the data to the desired distance-step
    # resolution

    # print(time.time() - t)
    # t = time.time()

    if step > _HGT_RES / 1.5:
        hdistances = np.arange(0., distance + _HGT_RES / 3., _HGT_RES / 3.)
        hlons, hlats, _ = cygeodesics.direct_cython(
            lon_t_rad, lat_t_rad, bearing_1_rad, hdistances
            )
        hlons = np.degrees(hlons)
        hlats = np.degrees(hlats)

        hheights = _get_interpolated_data(hlons, hlats).astype(np.float64)
        heights = np.empty_like(distances)
        # now smooth/interpolate this to the desired step width
        cygeodesics.regrid1d_with_x(
            hdistances, hheights, distances, heights,
            step / 2.35, regular=True
            )

    else:

        heights = _get_interpolated_data(lons, lats).astype(np.float64)

    return (
        lons, lats,
        distance * 1.e-3,
        distances * 1.e-3, heights,
        bearing_1, back_bearing, back_bearings
        )


@utils.ranged_quantity_input(
    lon_t=(-180, 180, apu.deg),
    lat_t=(-90, 90, apu.deg),
    lon_r=(-180, 180, apu.deg),
    lat_r=(-90, 90, apu.deg),
    step=(1., 1.e5, apu.m),
    strip_input_units=True,
    output_unit=(
        apu.deg, apu.deg, apu.km, apu.km, apu.m, apu.deg, apu.deg, apu.deg
        )
    )
def srtm_height_profile(lon_t, lat_t, lon_r, lat_r, step):
    '''
    Extract a height profile from SRTM data.

    Parameters
    ----------
    lon_t, lat_t : `~astropy.units.Quantity`
        Geographic longitude/latitude of start point (transmitter) [deg]
    lon_r, lat_r : `~astropy.units.Quantity`
        Geographic longitude/latitude of end point (receiver) [deg]
    step : `~astropy.units.Quantity`
        Distance resolution of height profile along path [m]

    Returns
    -------
    lons : `~astropy.units.Quantity` 1D array
        Geographic longitudes of path.
    lats : `~astropy.units.Quantity` 1D array
        Geographic latitudes of path.
    distance : `~astropy.units.Quantity` scalar
        Distance between start and end point of path.
    distances : `~astropy.units.Quantity` 1D array
        Distances along the path (with respect to start point).
    heights : `~astropy.units.Quantity` 1D array
        Terrain height along the path (aka Height profile).
    bearing : `~astropy.units.Quantity` scalar
        Start bearing of path.
    backbearing : `~astropy.units.Quantity` scalar
        Back-bearing at end point of path.
    backbearings : `~astropy.units.Quantity` 1D array
        Back-bearings for each point on the path.

    Notes
    -----
    - `distances` contains distances from Transmitter.
    - `SRTM data <https://www2.jpl.nasa.gov/srtm/>`_ need to be downloaded
      manually by the user. An environment variable `SRTMDATA` has to be
      set to point to the directory containing the .hgt files; see
      :ref:`srtm_data`.
    '''

    return _srtm_height_profile(lon_t, lat_t, lon_r, lat_r, step)


@utils.ranged_quantity_input(
    lon_c=(-180, 180, apu.deg),
    lat_c=(-90, 90, apu.deg),
    map_size_lon=(0.002, 90, apu.deg),
    map_size_lat=(0.002, 90, apu.deg),
    map_resolution=(0.0001, 0.1, apu.deg),
    hprof_step=(0.01, 30, apu.km),
    strip_input_units=True, allow_none=True,
    output_unit=(apu.deg, apu.deg, apu.m)
    )
def srtm_height_map(
        lon_c, lat_c,
        map_size_lon, map_size_lat,
        map_resolution=3. * apu.arcsec,
        hprof_step=None,
        do_cos_delta=True,
        do_coords_2d=False,
        ):
    '''
    Extract terrain map from SRTM data.

    Parameters
    ----------
    lon_t, lat_t : `~astropy.units.Quantity`
        Geographic longitude/latitude of map center [deg]
    map_size_lon, map_size_lat : `~astropy.units.Quantity`
        Map size in longitude/latitude[deg]
    map_resolution : `~astropy.units.Quantity`, optional
        Pixel resolution of map [deg] (default: 3 arcsec)
    hprof_step : `~astropy.units.Quantity`, optional
        Pixel resolution of map [m] (default: None)
        Overrides `map_resolution` if given!
    do_cos_delta : bool, optional
        If True, divide `map_size_lon` by `cos(lat_c)` to produce a more
        square-like map. (default: True)
    do_coords_2d : bool, optional
        If True, return 2D coordinate arrays (default: False)

    Returns
    -------
    lons : `~astropy.units.Quantity`, 1D or 2D
        Geographic longitudes [deg]
    lats : `~astropy.units.Quantity`, 1D or 2D
        Geographic latitudes [deg]
    heights : `~astropy.units.Quantity`, 1D or 2D
        Height map [m]

    Notes
    -----
    - `SRTM data <https://www2.jpl.nasa.gov/srtm/>`_ need to be downloaded
      manually by the user. An environment variable `SRTMDATA` has to be
      set to point to the directory containing the .hgt files; see
      :ref:`srtm_data`.
    '''

    if hprof_step is None:
        hprof_step = map_resolution * 3600. / 1. * 30.

    cosdelta = 1. / np.cos(np.radians(lat_c)) if do_cos_delta else 1.

    # construction map arrays
    xcoords = np.arange(
        lon_c - cosdelta * map_size_lon / 2,
        lon_c + cosdelta * map_size_lon / 2 + 1.e-6,
        cosdelta * map_resolution,
        )
    ycoords = np.arange(
        lat_c - map_size_lat / 2,
        lat_c + map_size_lat / 2 + 1.e-6,
        map_resolution,
        )
    xcoords2d, ycoords2d = np.meshgrid(xcoords, ycoords)

    heightmap = _get_interpolated_data(
        xcoords2d.flatten(), ycoords2d.flatten()
        ).reshape(xcoords2d.shape)

    if do_coords_2d:
        xcoords, ycoords = xcoords2d, ycoords2d

    return xcoords, ycoords, heightmap


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
