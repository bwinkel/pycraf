#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Note, there are various versions of SRTM data. Quasi-official are Versions 1
and 2.1 available on https://dds.cr.usgs.gov/srtm/. There is even a NASA
version 3, but we couldn't find a site for direct download. It may work
with an EarthData Account on https://lpdaac.usgs.gov/data_access/data_pool.

Then, there is V4.1 by CGIAR
(ftp://srtm.csi.cgiar.org/SRTM_V41/SRTM_Data_GeoTiff/)
and an unofficial version by viewfinderpanoramas.org (see
http://viewfinderpanoramas.org/Coverage%20map%20viewfinderpanoramas_org3.htm).

For automatic download we should use the 2.1 version by NASA. V4.1 is in
GeoTiff format, which we currently don't support. viewfinderpanoramas.org
is probably superior to 2.1 (maybe even to V4.1), but not official.

V4.1 and viewfinderpanoramas forbid commercial use (without explicit
permission).
'''


from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

# from functools import partial, lru_cache
import os
import warnings
import shutil
from zipfile import ZipFile
import re
import json
import glob
from functools import lru_cache
import numpy as np
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from astropy.utils.data import get_pkg_data_filename, download_file
from astropy import units as apu
from .. import utils


__all__ = [
    'TileNotAvailableOnServerError',
    'TileNotAvailableOnDiskError',
    'TileNotAvailableOnDiskWarning',
    'TilesSizeError',
    'SrtmConf', 'srtm_height_data'
    ]


_NASA_JSON_NAME = get_pkg_data_filename('data/nasa.json')
_VIEWPANO_NAME = get_pkg_data_filename('data/viewpano.npy')

with open(_NASA_JSON_NAME, 'r') as f:
    NASA_TILES = json.load(f)

VIEWPANO_TILES = np.load(_VIEWPANO_NAME)


class TileNotAvailableOnServerError(Exception):

    pass


class TileNotAvailableOnDiskError(Exception):

    pass


class TileNotAvailableOnDiskWarning(UserWarning):

    pass


class TilesSizeError(Exception):

    pass


class SrtmConf(utils.MultiState):
    '''
    Provide a global state to adjust SRTM configuration.

    By default, `~pycraf` will look for SRTM '.hgt' files (the terrain data)
    in the SRTMDATA environment variable. If this is not defined, the
    local directory ('./') is used for look-up. It is possible during
    run-time to change the directory where to look for '.hgt' files
    with the `SrtmConf` manager::

        from pycraf.pathprof import SrtmConf
        SrtmConf.set(srtm_dir='/path/to/srtmdir')

    This will also check, if all '.hgt' files have the same size. If not
    an error is raised.

    Alternatively, if only a temporary change of the config is desired,
    one can use `SrtmConf` as a context manager::

        with SrtmConf.set(srtm_dir='/path/to/srtmdir'):
            # do stuff

    Afterwards, the old settings will be re-established. It is also possible
    to allow downloading of missing '.hgt' files::

        SrtmConf.set(download='missing')

    The default behavior is to not download anything (`download='never'`).
    There is even an option, to always force download (`download='always'`).

    The default download server will be `server='nasa_v2.1'`. One could
    also use the (very old) data (`server='nasa_v1.0'`) or inofficial
    tiles from viewfinderpanorama (`server='viewpano'`).

    Of course, one can set several of these options simultaneously::

        with SrtmConf.set(
                srtm_dir='/path/to/srtmdir',
                download='missing',
                server='viewpano'
                ):

            # do stuff

    Last, but not least, it is possible to use different interpolation methods.
    The default method uses bi-linear interpolation (`interp='linear'`). One
    can also have nearest-neighbor (`interp='nearest'`) or spline
    (`interp='spline'`) interpolation. The two former internally use
    `~scipy.interpolate.RegularGridInterpolator`, the latter employs
    `~scipy.interpolate.RectBivariateSpline` that also allows custom
    spline degrees (`kx` and `ky`, default: 3) and smoothing factor (`s`,
    default: 0.). To change these use::

        SrtmConf.set(interp='spline', spline_opts=(k, s))

    We refer to `~scipy.interpolate.RectBivariateSpline` description for
    further information.

    Two read-only attributes are present, `tile_size` (pixels) and
    `hgt_res` (m), which are automatically inferred from the tile data.

    URLS:

    - `nasa_v2.1 <https://dds.cr.usgs.gov/srtm/version2_1/SRTM3/>`__
    - `nasa_v1.0 <https://dds.cr.usgs.gov/srtm/version1/>`__
    - `viewpano <http://www.viewfinderpanoramas.org/Coverage%20map%20viewfinderpanoramas_org3.htm>`__

    Note: As of Spring 2021, NASA decided to put all SRTM data products
    behind a log-in page, such that automatic download ceases to work.
    If you prefer to use NASA tiles (over viewpano), please use their
    services, e.g., the `Land Processes Distributed Active Archive Center
    <https://lpdaac.usgs.gov/>`

    '''

    _attributes = (
        'srtm_dir', 'download', 'server', 'interp', 'spline_opts',
        'tile_size', 'hgt_res'
        )

    srtm_dir = os.environ.get('SRTMDATA', '.')
    download = 'never'
    server = 'viewpano'
    interp = 'linear'
    spline_opts = (3, 0)
    tile_size = 1201
    hgt_res = 90.  # m; basic SRTM resolution (refers to 3 arcsec resolution)

    @classmethod
    def validate(cls, **kwargs):
        '''
        This checks, if the provided inputs for `download` and `server` are
        allowed. Possible values are:

        - `download`:  'never', 'missing', 'always'
        - `server`:  'viewpano'  # removed: 'nasa_v2.1', 'nasa_v1.0'
        - `interp`:  'nearest', 'linear', 'spline'
        - `spline_opts`:  tuple(k, s) (k = degree, s = smoothing factor)

        '''

        for k, v in kwargs.items():

            if k == 'srtm_dir':
                if not isinstance(v, str):
                    raise ValueError(
                        '"srtm_dir" option must be a string.'
                        )

            if k == 'download':
                if v not in ['never', 'missing', 'always']:
                    raise ValueError(
                        'Only the values "never", "missing", and "always" '
                        'are supported for "download" option.'
                        )
            if k == 'server':
                if v not in ['viewpano']:
                    raise ValueError(
                        'Only the value "viewpano" is currently '
                        'supported for "server" option.'
                        )

            if k == 'interp':
                if v not in ['nearest', 'linear', 'spline']:
                    raise ValueError(
                        'Only the values "nearest", "linear", and '
                        '"spline" are supported for "interp" option.'
                        )

            if k == 'spline_opts':
                if not isinstance(v, tuple):
                    raise ValueError(
                        '"spline_opts" option must be a tuple (k, s).'
                        )

                if not len(v) == 2:
                    raise ValueError(
                        '"spline_opts" option must be a tuple (k, s).'
                        )

                if not isinstance(v[0], int):
                    raise ValueError(
                        '"spline_opts" k-value must be an int.'
                        )

                if not isinstance(v[1], (int, float)):
                    raise ValueError(
                        '"spline_opts" s-value must be a float.'
                        )
            if k in ['tile_size', 'hgt_res']:

                raise KeyError(
                    'Setting the {} manually not allowed! '
                    '(This is automatically inferred from data.)'.format(k)
                    )

        return kwargs

    @classmethod
    def hook(cls, **kwargs):

        if 'srtm_dir' in kwargs:
            # check if srtm_dir changed and clear cache
            if kwargs['srtm_dir'] != cls.srtm_dir:
                get_tile_interpolator.cache_clear()

        if 'download' in kwargs:
            # check if 'download' strategy was changed and clear cache
            # this is necessary, because missing tiles will lead to
            # zero heights in the tile cache (for that tile) and if user
            # later sets the option to download missing tiles, the reading
            # routine needs to run again
            if kwargs['download'] != cls.download:
                get_tile_interpolator.cache_clear()

        if 'server' in kwargs:
            # dito
            if kwargs['server'] != cls.server:
                get_tile_interpolator.cache_clear()

    @classmethod
    def __repr__(cls):
        return (
            '<SrtmConf dir: {}, download: {}, server: {}, '
            'interp: {}, spline_opts: {}>'.format(
                cls.srtm_dir, cls.download, cls.server,
                cls.interp, cls.spline_opts
                ))

    @classmethod
    def __str__(cls):
        return (
            'SrtmConf\n  directory: {}\n  download: {}\n  server: {}\n'
            '  interp: {}\n  spline_opts: {}'.format(
                cls.srtm_dir, cls.download, cls.server,
                cls.interp, cls.spline_opts
                ))


def _hgt_filename(ilon, ilat):
    # construct proper hgt-file name

    return '{:1s}{:02d}{:1s}{:03d}.hgt'.format(
        'N' if ilat >= 0 else 'S',
        abs(ilat),
        'E' if ilon >= 0 else 'W',
        abs(ilon),
        )


def _check_availability(ilon, ilat):
    # check availability of a tile on download servers
    # returns continent name (for NASA server) or zip file name (Pano)

    server = SrtmConf.server
    tile_name = _hgt_filename(ilon, ilat)

    if server.startswith('nasa_v'):

        for continent, tiles in NASA_TILES.items():
            if tile_name in tiles:
                break
        else:
            raise TileNotAvailableOnServerError(
                'No tile found for ({}d, {}d) in list of available '
                'tiles.'.format(
                    ilon, ilat
                    ))

        return continent

    elif server == 'viewpano':

        tiles = VIEWPANO_TILES['tile']
        idx = np.where(tiles == tile_name)

        if len(tiles[idx]) == 0:
            raise TileNotAvailableOnServerError(
                'No tile found for ({}d, {}d) in list of available '
                'tiles.'.format(
                    ilon, ilat
                    ))

        return VIEWPANO_TILES['zipfile'][idx][0]

    return None  # should not happen


def _check_consistent_tile_sizes(srtm_dir):

    all_files = glob.glob(
        os.path.join(srtm_dir, '**', '*.hgt'),
        recursive=True
        )
    file_sizes = set(os.stat(fname).st_size for fname in all_files)

    if len(file_sizes) == 0:
        raise OSError('No .hgt tiles found in given srtm path.')
    elif len(file_sizes) > 1:
        raise TilesSizeError(
            'Inconsistent tile sizes found in given srtm path. '
            'All tiles must be the same size!'
            )

    tile_size = int(np.sqrt(file_sizes.pop() / 2) + 0.5)

    return tile_size


def _download(ilon, ilat):
    # download the tile to path

    srtm_dir = SrtmConf.srtm_dir
    server = SrtmConf.server

    tile_name = _hgt_filename(ilon, ilat)
    tile_path = os.path.join(srtm_dir, tile_name)

    # Unfortunately, each server has a different structure.
    # NASA stores them in sub-directories (by continents)
    # Panoramic-Viewfinders has a flat structure but has several hgt tiles
    # zipped in a file

    # Furthermore, we need to check against the available tiles
    # (ocean tiles and polar caps are not present); we also do this
    # in the _get_hgt_file function (because it's not only important
    # for downloading). However, we have to figure out, in which
    # subdirectory/zip-file a tile is located.

    if server.startswith('nasa_v'):

        if server == 'nasa_v1.0':
            base_url = 'https://dds.cr.usgs.gov/srtm/version1/'
        elif server == 'nasa_v2.1':
            base_url = 'https://dds.cr.usgs.gov/srtm/version2_1/SRTM3/'

        continent = _check_availability(ilon, ilat)

        # downloading
        full_url = base_url + continent + '/' + tile_name + '.zip'
        tmp_path = download_file(full_url)

        # move to srtm_dir
        shutil.move(tmp_path, tile_path + '.zip')

        # unpacking
        with ZipFile(tile_path + '.zip', 'r') as zf:
            zf.extractall(srtm_dir)

        try:
            os.remove(tile_path + '.zip')
        except (FileNotFoundError, PermissionError):
            # someone else was faster to delete or still accessing?
            pass

    elif server == 'viewpano':

        base_url = 'http://viewfinderpanoramas.org/dem3/'

        zipfile_name = _check_availability(ilon, ilat)
        super_tile_path = os.path.join(srtm_dir, zipfile_name)

        # downloading
        full_url = base_url + zipfile_name
        tmp_path = download_file(full_url)

        # move to srtm_dir
        shutil.move(tmp_path, super_tile_path)

        # unpacking
        with ZipFile(super_tile_path, 'r') as zf:
            zf.extractall(srtm_dir)

        try:
            os.remove(super_tile_path)
        except (FileNotFoundError, PermissionError):
            # someone else was faster to delete or still accessing?
            pass


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


def _get_hgt_diskpath(tile_name):
    # check, if a tile already exists in srtm directory (recursive)

    srtm_dir = SrtmConf.srtm_dir
    _files = glob.glob(os.path.join(srtm_dir, '**', tile_name), recursive=True)

    if len(_files) > 1:
        raise IOError(
            '{} exists {} times in {} and its sub-directories'.format(
                tile_name, len(_files), srtm_dir
                ))
    elif len(_files) == 0:
        return None
    else:
        return _files[0]


def get_hgt_file(ilon, ilat):

    _check_availability(ilon, ilat)

    srtm_dir = SrtmConf.srtm_dir
    tile_name = _hgt_filename(ilon, ilat)
    hgt_file = _get_hgt_diskpath(tile_name)

    download = SrtmConf.download
    if download == 'always' or (hgt_file is None and download == 'missing'):

        _download(ilon, ilat)

    hgt_file = _get_hgt_diskpath(tile_name)
    if hgt_file is None:
        raise TileNotAvailableOnDiskError(
            'No hgt-file found for ({}d, {}d), was looking for {}\n'
            'in directory: {}'.format(
                ilon, ilat, tile_name, srtm_dir
                ))

    return hgt_file


def get_tile_data(ilon, ilat):
    # angles in deg

    try:
        hgt_file = get_hgt_file(ilon, ilat)
        # need to run check after get_hgt_file, because download could happen
        _check_consistent_tile_sizes(SrtmConf.srtm_dir)
        tile = np.fromfile(hgt_file, dtype='>i2')
        tile_size = int(np.sqrt(tile.size) + 0.5)
        hgt_res = 90. * 1200 / (tile_size - 1)
        SrtmConf.set(tile_size=tile_size, _do_validate=False)
        SrtmConf.set(hgt_res=hgt_res, _do_validate=False)
        tile = tile.reshape((tile_size, tile_size))[::-1]

        bad_mask = (tile == 32768) | (tile == -32768)
        tile = tile.astype(np.float32)
        tile[bad_mask] = np.nan

    except TileNotAvailableOnServerError:
        # always use very small tile size for zero tiles
        # (just enough to make spline interpolation work)
        tile_size = 5
        tile = np.zeros((tile_size, tile_size), dtype=np.float32)

    except TileNotAvailableOnDiskError:
        # also set to zero, but raise a warning
        tile_size = 5
        tile = np.zeros((tile_size, tile_size), dtype=np.float32)

        tile_name = _hgt_filename(ilon, ilat)
        srtm_dir = SrtmConf.srtm_dir
        warnings.warn(
            '''
No hgt-file found for ({}d, {}d) - was looking for file {}
in directory: {}
Will set terrain heights in this area to zero. Note, you can have pycraf
download missing tiles automatically - just use "pycraf.pathprof.SrtmConf"
(see its documentation).'''.format(ilon, ilat, tile_name, srtm_dir),
            category=TileNotAvailableOnDiskWarning,
            stacklevel=1,
            )

    dx = dy = 1. / (tile_size - 1)
    x, y = np.ogrid[0:tile_size, 0:tile_size]
    lons, lats = x * dx + ilon, y * dy + ilat
    return lons, lats, tile


# cannot use SrtmConf inside to query interp and spline_opts, because
# caching might cause problems
@lru_cache(maxsize=36, typed=False)
def get_tile_interpolator(ilon, ilat, interp, spline_opts):
    # angles in deg

    lons, lats, tile = get_tile_data(ilon, ilat)
    # have to treat NaNs in some way; set to zero for now
    tile = np.nan_to_num(tile)

    if interp in ['nearest', 'linear']:
        _tile_interpolator = RegularGridInterpolator(
            (lons[:, 0], lats[0]), tile.T, method=interp,
            )
    elif interp == 'spline':
        kx = ky = spline_opts[0]
        s = spline_opts[1]
        _tile_interpolator = RectBivariateSpline(
            lons[:, 0], lats[0], tile.T, kx=kx, ky=ky, s=s,
            )

    return _tile_interpolator


def _srtm_height_data(lons, lats):
    # angles in deg

    # is there no way around constructing the full lon/lat grid?
    lons_g, lats_g = np.broadcast_arrays(lons, lats)
    heights = np.empty(lons_g.shape, dtype=np.float32)

    ilons = np.floor(lons).astype(np.int32)
    ilats = np.floor(lats).astype(np.int32)

    interp = SrtmConf.interp
    spl_opts = SrtmConf.spline_opts

    for uilon in np.unique(ilons):
        for uilat in np.unique(ilats):

            mask = (ilons == uilon) & (ilats == uilat)

            if interp in ['nearest', 'linear']:
                ifunc = get_tile_interpolator(uilon, uilat, interp, None)
                heights[mask] = ifunc((lons_g[mask], lats_g[mask]))
            elif interp == 'spline':
                ifunc = get_tile_interpolator(uilon, uilat, interp, spl_opts)
                heights[mask] = ifunc(lons_g[mask], lats_g[mask], grid=False)

    return heights


@utils.ranged_quantity_input(
    lons=(-180, 180, apu.deg),
    lats=(-90, 90, apu.deg),
    strip_input_units=True,
    output_unit=apu.m
    )
def srtm_height_data(lons, lats):
    '''
    Interpolated SRTM terrain data extracted from ".hgt" files.

    Parameters
    ----------
    lons, lats : `~astropy.units.Quantity`
        Geographic longitudes/latitudes for which to return height data [deg]

    Returns
    -------
    heights : `~astropy.units.Quantity`
        SRTM heights [m]

    Raises
    ------
    TileNotAvailableOnDiskWarning : UserWarning
        If a tile is requested that should exist on the chosen server
        but is not available on disk (at least not in the search path)
        a warning is raised. In this case, the tile height data is set
        to Zeros.

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

    return _srtm_height_data(lons, lats)


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
