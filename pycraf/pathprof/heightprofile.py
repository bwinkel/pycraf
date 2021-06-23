#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

from astropy import units as apu
import numpy as np
from . import cygeodesics
from . import srtm
from .. import utils


__all__ = [
    'srtm_height_profile', 'srtm_height_map',
    ]


def _srtm_height_profile(
        lon_t, lat_t, lon_r, lat_r, step, generic_heights=False
        ):
    # angles in deg; lengths in m

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

    if generic_heights:
        heights = np.zeros_like(lons)

    else:
        # important: unless the requested resolution is super-fine, we always
        # have to query the raw height profile data using sufficient
        # resolution, to acquire all features
        # only afterwards, we may smooth the data to the desired distance-step
        # resolution

        # print(time.time() - t)
        # t = time.time()

        # hgt_res may not be set correctly yet, if call to srtm wasn't made
        # before
        # let's do a simple query to make sure, it is set
        srtm._srtm_height_data(lon_t, lat_t)
        hgt_res = srtm.SrtmConf.hgt_res
        if step > hgt_res / 1.5:
            hdistances = np.arange(
                0., distance + hgt_res / 3., hgt_res / 3.
                )
            hlons, hlats, _ = cygeodesics.direct_cython(
                lon_t_rad, lat_t_rad, bearing_1_rad, hdistances
                )
            hlons = np.degrees(hlons)
            hlats = np.degrees(hlats)

            hheights = srtm._srtm_height_data(hlons, hlats).astype(np.float64)
            heights = np.empty_like(distances)
            # now smooth/interpolate this to the desired step width
            cygeodesics.regrid1d_with_x(
                hdistances, hheights, distances, heights,
                step / 2.35, regular=True
                )

        else:

            heights = srtm._srtm_height_data(lons, lats).astype(np.float64)

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
def srtm_height_profile(
        lon_t, lat_t, lon_r, lat_r, step, generic_heights=False
        ):
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
    generic_heights : bool
        If `generic_heights` is set to True, heights will be set to zero.
        This can be useful for generic (aka flat-Earth) computations.
        (Default: False)

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
    - `SRTM <https://www2.jpl.nasa.gov/srtm/>`_ data tiles (`*.hgt`) need
      to be accessible by `pycraf`.  It is assumed that these are either
      present in the current working directory or in the path defined by the
      `SRTMDATA` environment variable (sub-directories are also parsed).
      Alternatively, use the `~pycraf.pathprof.SrtmConf` manager to
      change the directory, where `pycraf` looks for SRTM data, during
      run-time. The `~pycraf.pathprof.SrtmConf` manager also offers
      additional features such as automatic downloading of missing
      tiles or applying different interpolation methods (e.g., splines).
      For details see :ref:`working_with_srtm`.
    '''

    return _srtm_height_profile(
        lon_t, lat_t, lon_r, lat_r, step, generic_heights
        )


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
    - `SRTM <https://www2.jpl.nasa.gov/srtm/>`_ data tiles (`*.hgt`) need
      to be accessible by `pycraf`.  It is assumed that these are either
      present in the current working directory or in the path defined by the
      `SRTMDATA` environment variable (sub-directories are also parsed).
      Alternatively, use the `~pycraf.pathprof.SrtmConf` manager to
      change the directory, where `pycraf` looks for SRTM data, during
      run-time. The `~pycraf.pathprof.SrtmConf` manager also offers
      additional features such as automatic downloading of missing
      tiles or applying different interpolation methods (e.g., splines).
      For details see :ref:`working_with_srtm`.
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

    heightmap = srtm._srtm_height_data(
        xcoords2d.flatten(), ycoords2d.flatten()
        ).reshape(xcoords2d.shape)

    if do_coords_2d:
        xcoords, ycoords = xcoords2d, ycoords2d

    return xcoords, ycoords, heightmap


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
