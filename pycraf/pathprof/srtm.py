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

Copernicus DEM
--------------

In addition to the SRTM '.hgt' tiles, `pycraf` can use the Copernicus DEM
(GLO-90 and GLO-30) as a terrain source. These are hosted as Cloud-Optimised
GeoTIFFs on the AWS Open Data buckets (no authentication required) and, unlike
the retired SRTM auto-download servers, provide a reliable download path. The
Copernicus DEM is global (pole-to-pole, whereas SRTM only covers 60N to 56S)
and is void-free over water. To use it::

    from pycraf.pathprof import SrtmConf
    SrtmConf.set(
        srtm_dir='/path/to/copernicus', download='missing',
        server='copernicus_glo90',
        )

Reading the GeoTIFF tiles requires the (optional) `rasterio` package.

Model type: the Copernicus DEM is a Digital Surface Model (DSM), i.e. it
"represents the surface of the Earth including buildings, infrastructure and
vegetation" (source: Copernicus DEM readme,
https://copernicus-dem-30m.s3.amazonaws.com/readme.html; see also the
Copernicus DEM Product Handbook). It is derived from the TanDEM-X radar
mission (X-band). It is therefore NOT a bare-earth terrain model; canopy and
building heights are included. This matches SRTM, which is also a radar DSM
(C-band), so the surface-vs-terrain character is unchanged when switching
between the two. Note that P.452 additionally models clutter separately (via
the `zone_t` / `zone_r` options), so assigning clutter over a DSM in built-up
or forested terminals can double-count vegetation/building height.

Attribution (required when using or redistributing Copernicus DEM data):

    Produced using Copernicus WorldDEM-90 (c) DLR e.V. 2010-2014 and
    (c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by
    the European Union and ESA; all rights reserved.

Note, the Copernicus DEM uses the EGM2008 geoid as vertical datum (SRTM uses
EGM96); the difference is at the metre level and is not converted here, as
propagation profiles only depend on relative terrain heights.
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


# Copernicus DEM (GLO-90 / GLO-30) on the AWS Open Data buckets.
# The "code" is the internal arc-second*10/3 identifier that appears in the
# tile names ('30' -> GLO-90 = 3 arcsec, '10' -> GLO-30 = 1 arcsec); it is
# *not* the resolution in metres. `hgt_res` is the nominal (equatorial)
# latitude pixel spacing in metres, used to pick the height-profile sampling.
COPERNICUS_SERVERS = {
    'copernicus_glo90': {
        'base_url': 'https://copernicus-dem-90m.s3.amazonaws.com/',
        'code': '30',
        'hgt_res': 90.,
        },
    'copernicus_glo30': {
        'base_url': 'https://copernicus-dem-30m.s3.amazonaws.com/',
        'code': '10',
        'hgt_res': 30.,
        },
    }

# cache of the authoritative tile inventories (server -> set of tile names),
# lazily populated from a local copy of the bucket "tileList.txt" or, if
# downloading is enabled, from the bucket root
_COPERNICUS_TILE_LISTS = {}


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

    The default download server is `server='viewpano'` (inofficial tiles
    from viewfinderpanorama). Alternatively, one can use the Copernicus DEM
    (`server='copernicus_glo90'` or `server='copernicus_glo30'`), which -
    unlike the retired SRTM servers - offers a reliably working download
    path (AWS Open Data, no authentication), is global (pole-to-pole) and
    void-free over water. The Copernicus tiles are Cloud-Optimised GeoTIFFs
    and require the optional `rasterio` package (a clear error is raised if
    a Copernicus server is selected without it). See the module
    documentation for the required attribution statement.

    Two further options control the behaviour when data is missing or
    invalid. `on_missing` decides what happens if a tile that *should*
    exist on the chosen server is not found on disk (and cannot be
    downloaded): `on_missing='zeros'` (default) sets the terrain of that
    tile to zero and emits a `TileNotAvailableOnDiskWarning` (the historic
    behaviour), whereas `on_missing='raise'` raises a
    `TileNotAvailableOnDiskError` instead - useful to avoid silently
    computing over zero-terrain. `void_fill` controls how void pixels
    (SRTM data gaps; the Copernicus DEM is void-free) are treated:
    `void_fill='zero'` (default) replaces voids with zero,
    `void_fill='nan'` keeps them as `NaN` (so they can be detected
    downstream), and `void_fill='interp'` fills them from the nearest
    valid pixels. To change these use::

        SrtmConf.set(on_missing='raise', void_fill='interp')

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
        'on_missing', 'void_fill', 'tile_size', 'hgt_res'
        )

    srtm_dir = os.environ.get('SRTMDATA', '.')
    download = 'never'
    server = 'viewpano'
    interp = 'linear'
    spline_opts = (3, 0)
    on_missing = 'zeros'
    void_fill = 'zero'
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
                if v not in [
                        'viewpano',
                        'copernicus_glo90', 'copernicus_glo30',
                        ]:
                    raise ValueError(
                        'Only the values "viewpano", "copernicus_glo90", '
                        'and "copernicus_glo30" are currently supported for '
                        'the "server" option.'
                        )

            if k == 'on_missing':
                if v not in ['zeros', 'raise']:
                    raise ValueError(
                        'Only the values "zeros" and "raise" are supported '
                        'for the "on_missing" option.'
                        )

            if k == 'void_fill':
                if v not in ['zero', 'nan', 'interp']:
                    raise ValueError(
                        'Only the values "zero", "nan", and "interp" are '
                        'supported for the "void_fill" option.'
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
                # the Copernicus tile inventory is per directory
                _COPERNICUS_TILE_LISTS.clear()

        if 'server' in kwargs and kwargs['server'] != cls.server:
            # the cached Copernicus inventory is per server
            _COPERNICUS_TILE_LISTS.clear()

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

        if 'on_missing' in kwargs:
            # changes whether a missing tile becomes zeros or an error,
            # i.e. the cached tile data would differ
            if kwargs['on_missing'] != cls.on_missing:
                get_tile_interpolator.cache_clear()

        if 'void_fill' in kwargs:
            # changes how void pixels are filled in the cached interpolator
            if kwargs['void_fill'] != cls.void_fill:
                get_tile_interpolator.cache_clear()

    @classmethod
    def __repr__(cls):
        return (
            '<SrtmConf dir: {}, download: {}, server: {}, '
            'interp: {}, spline_opts: {}, on_missing: {}, void_fill: {}>'
            ''.format(
                cls.srtm_dir, cls.download, cls.server,
                cls.interp, cls.spline_opts, cls.on_missing, cls.void_fill
                ))

    @classmethod
    def __str__(cls):
        return (
            'SrtmConf\n  directory: {}\n  download: {}\n  server: {}\n'
            '  interp: {}\n  spline_opts: {}\n  on_missing: {}\n'
            '  void_fill: {}'.format(
                cls.srtm_dir, cls.download, cls.server,
                cls.interp, cls.spline_opts, cls.on_missing, cls.void_fill
                ))


def _hgt_filename(ilon, ilat):
    # construct proper hgt-file name

    return '{:1s}{:02d}{:1s}{:03d}.hgt'.format(
        'N' if ilat >= 0 else 'S',
        abs(ilat),
        'E' if ilon >= 0 else 'W',
        abs(ilon),
        )


def _copernicus_tilename(ilon, ilat, server=None):
    # construct the Copernicus DEM tile (base) name for the tile whose
    # south-west corner is at the integer degree (ilon, ilat)

    if server is None:
        server = SrtmConf.server
    code = COPERNICUS_SERVERS[server]['code']

    return (
        'Copernicus_DSM_COG_{code}_{ns:1s}{ilat:02d}_00_{ew:1s}{ilon:03d}'
        '_00_DEM'.format(
            code=code,
            ns='N' if ilat >= 0 else 'S', ilat=abs(ilat),
            ew='E' if ilon >= 0 else 'W', ilon=abs(ilon),
            )
        )


def _copernicus_tile_set(server=None):
    # return the authoritative set of tile names for a Copernicus server,
    # or None if the inventory is not available (and cannot be fetched)

    if server is None:
        server = SrtmConf.server

    if server in _COPERNICUS_TILE_LISTS:
        return _COPERNICUS_TILE_LISTS[server]

    srtm_dir = SrtmConf.srtm_dir
    list_name = os.path.join(srtm_dir, server + '_tileList.txt')

    tiles = None
    if os.path.exists(list_name):
        with open(list_name, 'r') as f:
            tiles = set(line.strip() for line in f if line.strip())
    elif SrtmConf.download in ['missing', 'always']:
        # fetch the authoritative list from the bucket root and cache it
        base_url = COPERNICUS_SERVERS[server]['base_url']
        tmp_path = download_file(base_url + 'tileList.txt')
        try:
            os.makedirs(srtm_dir, exist_ok=True)
            shutil.copyfile(tmp_path, list_name)
        except OSError:
            # srtm_dir not writable - keep the in-memory copy only
            pass
        with open(tmp_path, 'r') as f:
            tiles = set(line.strip() for line in f if line.strip())

    _COPERNICUS_TILE_LISTS[server] = tiles
    return tiles


def _check_availability(ilon, ilat):
    # check availability of a tile on download servers
    # returns continent name (for NASA server) or zip file name (Pano)

    server = SrtmConf.server
    tile_name = _hgt_filename(ilon, ilat)

    if server.startswith('copernicus'):

        cop_name = _copernicus_tilename(ilon, ilat)
        tiles = _copernicus_tile_set()

        # if the inventory is unknown (download='never' and no cached list),
        # we cannot rule the tile out - defer the decision to the disk lookup
        if tiles is not None and cop_name not in tiles:
            raise TileNotAvailableOnServerError(
                'No tile found for ({}d, {}d) in list of available '
                'tiles.'.format(
                    ilon, ilat
                    ))

        return cop_name

    elif server.startswith('nasa_v'):

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

    if server.startswith('copernicus'):

        # Copernicus tiles are single Cloud-Optimised GeoTIFFs, stored as
        # "<TileName>/<TileName>.tif" in the bucket; we keep them flat on disk
        base_url = COPERNICUS_SERVERS[server]['base_url']
        cop_name = _copernicus_tilename(ilon, ilat)
        full_url = base_url + cop_name + '/' + cop_name + '.tif'
        tile_path = os.path.join(srtm_dir, cop_name + '.tif')

        tmp_path = download_file(full_url)
        os.makedirs(srtm_dir, exist_ok=True)
        shutil.move(tmp_path, tile_path)

        return

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
        r".*([NS])(-?\d*)([EW])(\d*).hgt.*", hgt_name
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


def get_copernicus_file(ilon, ilat):
    # locate (and, if requested, download) the Copernicus GeoTIFF tile whose
    # south-west corner is at the integer degree (ilon, ilat)

    _check_availability(ilon, ilat)

    srtm_dir = SrtmConf.srtm_dir
    tif_name = _copernicus_tilename(ilon, ilat) + '.tif'
    tif_file = _get_hgt_diskpath(tif_name)

    download = SrtmConf.download
    if download == 'always' or (tif_file is None and download == 'missing'):

        _download(ilon, ilat)

    tif_file = _get_hgt_diskpath(tif_name)
    if tif_file is None:
        raise TileNotAvailableOnDiskError(
            'No Copernicus tile found for ({}d, {}d), was looking for {}\n'
            'in directory: {}'.format(
                ilon, ilat, tif_name, srtm_dir
                ))

    return tif_file


# metres per degree of latitude (mean, spherical Earth); only used to turn the
# tile's latitude pixel spacing into an approximate resolution for choosing the
# height-profile sampling step
_M_PER_DEG_LAT = 111120.


def _read_copernicus_cog(tif_file):
    # read a Copernicus DEM Cloud-Optimised GeoTIFF and return coordinate and
    # height arrays in pycraf's tile convention:
    #   lons  -> shape (nlon, 1), ascending west -> east
    #   lats  -> shape (1, nlat), ascending south -> north
    #   tile  -> shape (nlat, nlon), tile[lat_idx, lon_idx], voids set to NaN
    # Copernicus tiles use pixel-centre (area) registration and their
    # longitude spacing widens above |lat| 50 deg, so the actual geotransform
    # is read from the file rather than assumed.

    try:
        import rasterio
    except ImportError as e:
        raise ImportError(
            'The "rasterio" package is required to read Copernicus DEM '
            '(GeoTIFF) tiles. Install it (e.g. "pip install rasterio") or '
            'select a different "server" in pycraf.pathprof.SrtmConf.'
            ) from e

    with rasterio.open(tif_file) as ds:
        tile = ds.read(1).astype(np.float32)  # (nlat, nlon), row 0 = north
        transf = ds.transform
        nodata = ds.nodata

    nlat, nlon = tile.shape
    # pixel-centre coordinates from the affine transform
    lon0 = transf.c + 0.5 * transf.a  # centre of column 0
    lat0 = transf.f + 0.5 * transf.e  # centre of row 0 (northern-most)
    lon_axis = lon0 + np.arange(nlon) * transf.a         # west -> east
    lat_axis = lat0 + np.arange(nlat) * transf.e         # north -> south

    # flip rows to go south -> north, matching the '.hgt' convention
    tile = tile[::-1]
    lat_axis = lat_axis[::-1]

    # NoData handling: Copernicus is void-free over water, but coastal tiles
    # can carry NaN or negative sentinels; mask conservatively (sea -> 0 m is
    # applied later via "void_fill", consistent with the SRTM convention)
    bad_mask = ~np.isfinite(tile) | (tile < -500.)
    if nodata is not None and np.isfinite(nodata):
        bad_mask |= (tile == np.float32(nodata))
    tile[bad_mask] = np.nan

    hgt_res = abs(transf.e) * _M_PER_DEG_LAT
    SrtmConf.set(tile_size=nlat, _do_validate=False)
    SrtmConf.set(hgt_res=hgt_res, _do_validate=False)

    lons = lon_axis[:, np.newaxis]
    lats = lat_axis[np.newaxis, :]
    return lons, lats, tile


def _missing_tile_warning(ilon, ilat, tile_name):
    # emit the historic "tile not on disk -> zeros" warning

    srtm_dir = SrtmConf.srtm_dir
    warnings.warn(
        '''
No tile found for ({}d, {}d) - was looking for file {}
in directory: {}
Will set terrain heights in this area to zero. Note, you can have pycraf
download missing tiles automatically - just use "pycraf.pathprof.SrtmConf"
(see its documentation). To turn this into an error instead, set
"SrtmConf.set(on_missing='raise')".'''.format(
            ilon, ilat, tile_name, srtm_dir),
        category=TileNotAvailableOnDiskWarning,
        stacklevel=1,
        )


def _zero_tile(ilon, ilat):
    # a minimal (5x5) zero tile, just big enough for spline interpolation
    tile_size = 5
    tile = np.zeros((tile_size, tile_size), dtype=np.float32)
    dx = dy = 1. / (tile_size - 1)
    x, y = np.ogrid[0:tile_size, 0:tile_size]
    lons, lats = x * dx + ilon, y * dy + ilat
    return lons, lats, tile


def _get_srtm_tile_data(ilon, ilat):

    hgt_file = get_hgt_file(ilon, ilat)
    # need to run check after get_hgt_file, because download could happen
    _check_consistent_tile_sizes(SrtmConf.srtm_dir)
    tile = np.fromfile(hgt_file, dtype='>i2')
    tile_size = int(np.sqrt(tile.size) + 0.5)
    hgt_res = 90. * 1200 / (tile_size - 1)
    SrtmConf.set(tile_size=tile_size, _do_validate=False)
    SrtmConf.set(hgt_res=hgt_res, _do_validate=False)
    tile = tile.reshape((tile_size, tile_size))[::-1]

    # void/NoData sentinels: the canonical SRTM void is -32768 (0x8000);
    # -32767 and +32767 are also seen in some products. (The historic code
    # masked only -32767/+32767, so genuine -32768 voids leaked through and
    # were linearly blended with valid neighbours, producing spurious pits.)
    bad_mask = (tile == -32768) | (tile == -32767) | (tile == 32767)
    tile = tile.astype(np.float32)
    tile[bad_mask] = np.nan

    dx = dy = 1. / (tile_size - 1)
    x, y = np.ogrid[0:tile_size, 0:tile_size]
    lons, lats = x * dx + ilon, y * dy + ilat
    return lons, lats, tile


def get_tile_data(ilon, ilat):
    # angles in deg

    server = SrtmConf.server

    try:
        if server.startswith('copernicus'):
            tif_file = get_copernicus_file(ilon, ilat)
            return _read_copernicus_cog(tif_file)
        else:
            return _get_srtm_tile_data(ilon, ilat)

    except TileNotAvailableOnServerError:
        # tile is genuinely absent from the server (ocean, polar cap): use a
        # zero tile silently, as before
        return _zero_tile(ilon, ilat)

    except TileNotAvailableOnDiskError:
        # tile should exist but is not on disk (and wasn't downloaded); either
        # raise or fall back to zeros (+ warning), depending on "on_missing"
        if SrtmConf.on_missing == 'raise':
            raise

        if server.startswith('copernicus'):
            tile_name = _copernicus_tilename(ilon, ilat) + '.tif'
        else:
            tile_name = _hgt_filename(ilon, ilat)
        _missing_tile_warning(ilon, ilat, tile_name)
        return _zero_tile(ilon, ilat)


def _fill_voids(tile, void_fill):
    # resolve NaN void pixels in a tile according to the "void_fill" policy;
    # returns a float array free of NaNs unless void_fill == 'nan'

    if void_fill == 'nan':
        return tile

    bad_mask = ~np.isfinite(tile)
    if not bad_mask.any():
        return tile

    if void_fill == 'interp':
        if bad_mask.all():
            return np.nan_to_num(tile)
        from scipy import ndimage
        # fill each void with the value of the nearest valid pixel
        idx = ndimage.distance_transform_edt(
            bad_mask, return_distances=False, return_indices=True
            )
        return tile[tuple(idx)]

    # void_fill == 'zero'
    return np.nan_to_num(tile)


# cannot use SrtmConf inside to query interp and spline_opts, because
# caching might cause problems (changing them does not clear this cache).
# "void_fill" is safe to query here, because the SrtmConf.hook clears this
# cache whenever "void_fill" (or the tile source) changes.
@lru_cache(maxsize=36, typed=False)
def get_tile_interpolator(ilon, ilat, interp, spline_opts):
    # angles in deg

    lons, lats, tile = get_tile_data(ilon, ilat)
    # resolve voids (NaNs); default replaces them with zero
    tile = _fill_voids(tile, SrtmConf.void_fill)

    if interp in ['nearest', 'linear']:
        # bounds_error=False + fill_value=None extrapolates for query points
        # that fall just outside the pixel-centre grid. This is needed for the
        # Copernicus (area-registered) tiles, whose pixel centres do not reach
        # the southern/eastern tile edge, and is a no-op for the (node-
        # registered) SRTM tiles, which always cover the assigned degree cell.
        _tile_interpolator = RegularGridInterpolator(
            (lons[:, 0], lats[0]), tile.T, method=interp,
            bounds_error=False, fill_value=None,
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
