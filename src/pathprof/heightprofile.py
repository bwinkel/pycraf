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
from . import geodesics
from .. import conversions as cnv
from .. import helpers


__all__ = [
    'srtm_height_profile',
    ]


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
            print('Warning, no SRTM data found.')

    hgt_files = glob.glob(
        os.path.join(basepath, '**', '*.hgt*'),
        recursive=True
        )

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

    hgt_file = HGT_DICT[ilon, ilat]

    tile_size = 1201
    dx = dy = 1. / (tile_size - 1)
    x, y = np.ogrid[0:tile_size, 0:tile_size]
    lons, lats = x * dx + ilon, y * dy + ilat

    tile = np.fromfile(hgt_file, dtype='>i2')
    tile = tile.reshape((tile_size, tile_size))[::-1]
    bad_mask = (tile == 32768) | (tile == -32768)
    tile = tile.astype(np.float32)
    tile[bad_mask] = np.nan

    return lons, lats, tile


@lru_cache(maxsize=30, typed=False)
def _get_tile_interpolator(ilon, ilat):

    lons, lats, tile = _get_tile_data(ilon, ilat)

    _tile_interpolator = RegularGridInterpolator(
        (lons[:, 0], lats[0]), tile
        )

    return _tile_interpolator


def _get_interpolated_data(lons, lats):

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

    # first find start bearing (and backward bearing for the curious people)
    # and distance

    distance, bearing_1, bearing_2 = geodesics.inverse(
        lon_t, lat_t, lon_r, lat_r
        )
    back_bearing = bearing_1 % 360. - 180.

    distances = np.arange(0., distance + step, step)  # [m]
    lons = np.empty_like(distances)
    lats = np.empty_like(distances)

    for idx, d in enumerate(distances):
        # TODO: build a numpy interface in Geodesics
        lons[idx], lats[idx], _ = geodesics.direct(lon_t, lat_t, bearing_1, d)

    heights = _get_interpolated_data(lons, lats)

    return (
        lons,
        lats,
        distances * 1.e-3,
        heights,
        bearing_1,
        back_bearing,
        distance * 1.e-3,
        )


@helpers.ranged_quantity_input(
    lon_t=(0, 360, apu.deg),
    lat_t=(-90, 90, apu.deg),
    lon_r=(0, 360, apu.deg),
    lat_r=(-90, 90, apu.deg),
    step=(1., 1.e5, apu.m),
    strip_input_units=True,
    output_unit=(apu.deg, apu.deg, apu.km, apu.m, apu.deg, apu.deg, apu.km)
    )
def srtm_height_profile(lon_t, lat_t, lon_r, lat_r, step):
    '''
    Extract a height profile from SRTM data.

    Parameters
    ----------
    lon_t, lat_t - Transmitter coordinates [deg]
    lon_r, lat_r - Receiver coordinates [deg]

    Returns
    -------
    d_vec, h_vec - Path dist/height profile vectors [m]

    Notes
    -----
    - d_vec contains distances from Transmitter.
    '''

    return _srtm_height_profile(lon_t, lat_t, lon_r, lat_r, step)


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
