#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import importlib
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from astropy.tests.helper import assert_quantity_allclose
from astropy import units as apu
from ...pathprof import srtm
from ...utils import check_astro_quantities


TOL_KWARGS = {'atol': 1.e-4, 'rtol': 1.e-4}

# skip Copernicus (GeoTIFF) reading tests, if rasterio is not installed
skip_rasterio = pytest.mark.skipif(
    importlib.util.find_spec('rasterio') is None,
    reason='"rasterio" package not installed'
    )


class TestSrtmConf:

    def setup_method(self):

        srtm.SrtmConf.set(
            srtm_dir=os.environ.get('SRTMDATA', '.'),
            download='never',
            server='viewpano',
            interp='linear',
            )

    def test_context_manager(self):

        srtm_dir = srtm.SrtmConf.srtm_dir
        download = srtm.SrtmConf.download

        with srtm.SrtmConf.set(srtm_dir='bar', download='always'):
            pass

        assert srtm_dir == srtm.SrtmConf.srtm_dir
        assert download == srtm.SrtmConf.download

    def test_getter(self):

        assert srtm.SrtmConf.srtm_dir == os.environ.get('SRTMDATA', '.')
        assert srtm.SrtmConf.download == 'never'
        assert srtm.SrtmConf.server == 'viewpano'

    def test_setter(self):

        with srtm.SrtmConf.set(srtm_dir='foo'):
            assert srtm.SrtmConf.srtm_dir == 'foo'
            assert srtm.SrtmConf.download == 'never'
            assert srtm.SrtmConf.server == 'viewpano'

        with srtm.SrtmConf.set(download='missing'):
            assert srtm.SrtmConf.srtm_dir == os.environ.get('SRTMDATA', '.')
            assert srtm.SrtmConf.download == 'missing'
            assert srtm.SrtmConf.server == 'viewpano'

        with srtm.SrtmConf.set(srtm_dir='bar', download='always'):
            assert srtm.SrtmConf.srtm_dir == 'bar'
            assert srtm.SrtmConf.download == 'always'
            assert srtm.SrtmConf.server == 'viewpano'

        with pytest.raises(RuntimeError):
            srtm.SrtmConf.srtm_dir = 'bar'

        with pytest.raises(RuntimeError):
            srtm.SrtmConf()

    def test_validation(self):

        with pytest.raises(TypeError):
            with srtm.SrtmConf.set(1):
                pass

        with pytest.raises(ValueError):
            with srtm.SrtmConf.set(srtm_dir=1):
                pass

        with pytest.raises(ValueError):
            with srtm.SrtmConf.set(foo='bar'):
                pass

        with pytest.raises(ValueError):
            with srtm.SrtmConf.set(download='bar'):
                pass

        with pytest.raises(ValueError):
            with srtm.SrtmConf.set(server='bar'):
                pass


def test_hgt_filename():

    cases = [
        (10, 10, 'N10E010.hgt'),
        (0, 20, 'N20E000.hgt'),
        (0, 0, 'N00E000.hgt'),
        (-1, -1, 'S01W001.hgt'),
        (-10, -1, 'S01W010.hgt'),
        (10, -1, 'S01E010.hgt'),
        (19, 18, 'N18E019.hgt'),
        (28, 35, 'N35E028.hgt'),
        (-24, -1, 'S01W024.hgt'),
        (-111, -40, 'S40W111.hgt'),
        (119, 12, 'N12E119.hgt'),
        (86, -46, 'S46E086.hgt'),
        (147, -54, 'S54E147.hgt'),
        (-20, -71, 'S71W020.hgt'),
        (-46, -79, 'S79W046.hgt'),
        (-46, -22, 'S22W046.hgt'),
        (6, 25, 'N25E006.hgt'),
        (67, -22, 'S22E067.hgt'),
        (63, -38, 'S38E063.hgt'),
        (-97, 51, 'N51W097.hgt'),
        (148, -38, 'S38E148.hgt'),
        (53, 39, 'N39E053.hgt'),
        (27, -67, 'S67E027.hgt'),
        (57, 20, 'N20E057.hgt'),
        (109, -31, 'S31E109.hgt'),
        (-143, 74, 'N74W143.hgt'),
        ]

    for ilon, ilat, name in cases:
        assert srtm._hgt_filename(ilon, ilat) == name


def test_extract_hgt_coords():

    cases = [
        (10, 10, 'N10E010.hgt'),
        (0, 20, 'N20E000.hgt'),
        (0, 0, 'N00E000.hgt'),
        (-1, -1, 'S01W001.hgt'),
        (-10, -1, 'S01W010.hgt'),
        (10, -1, 'S01E010.hgt'),
        (19, 18, 'N18E019.hgt'),
        (28, 35, 'N35E028.hgt'),
        (-24, -1, 'S01W024.hgt'),
        (-111, -40, 'S40W111.hgt'),
        (119, 12, 'N12E119.hgt'),
        (86, -46, 'S46E086.hgt'),
        (147, -54, 'S54E147.hgt'),
        (-20, -71, 'S71W020.hgt'),
        (-46, -79, 'S79W046.hgt'),
        (-46, -22, 'S22W046.hgt'),
        (6, 25, 'N25E006.hgt'),
        (67, -22, 'S22E067.hgt'),
        (63, -38, 'S38E063.hgt'),
        (-97, 51, 'N51W097.hgt'),
        (148, -38, 'S38E148.hgt'),
        (53, 39, 'N39E053.hgt'),
        (27, -67, 'S67E027.hgt'),
        (57, 20, 'N20E057.hgt'),
        (109, -31, 'S31E109.hgt'),
        (-143, 74, 'N74W143.hgt'),
        ]

    for ilon, ilat, name in cases:
        assert srtm._extract_hgt_coords(name) == (ilon, ilat)


@pytest.mark.skip(reason="NASA tiles not available without log-in anymore")
def test_check_availability_nasa():

    nasa_tiles = [
        ('Australia', 1060),
        ('South_America', 1807),
        ('Islands', 141),
        ('Africa', 3250),
        ('Eurasia', 5876),
        ('North_America', 2412),
        ]

    for k, v in nasa_tiles:

        assert v == len(srtm.NASA_TILES[k])

    nasa_cases = [
        (19, 18, 'Africa'),
        (28, 35, None),
        (-24, -1, None),
        (-111, -40, None),
        (119, 12, 'Eurasia'),
        (86, -46, None),
        (147, -54, None),
        (-20, -71, None),
        (-46, -79, None),
        (-46, -22, 'South_America'),
        (6, 25, 'Africa'),
        (67, -22, None),
        (63, -38, None),
        (-97, 51, 'North_America'),
        (148, -38, 'Australia'),
        (53, 39, 'Eurasia'),
        (27, -67, None),
        (57, 20, 'Africa'),
        (109, -31, None),
        (-143, 74, None),
        ]

    for ilon, ilat, name in nasa_cases:

        if name is None:
            with pytest.raises(srtm.TileNotAvailableOnServerError):
                srtm._check_availability(ilon, ilat)
        else:
            assert srtm._check_availability(ilon, ilat) == name


def test_check_availability_pano():

    assert srtm.VIEWPANO_TILES.size == 19297

    pano_cases = [
        (19, 18, 'E34.zip'),
        (28, 35, None),
        (-24, -1, None),
        (-111, -40, None),
        (119, 12, 'D50.zip'),
        (86, -46, None),
        (147, -54, None),
        (-20, -71, None),
        (-46, -79, None),
        (-46, -22, 'SF23.zip'),
        (6, 25, 'G32.zip'),
        (67, -22, None),
        (63, -38, None),
        (-97, 51, 'M14.zip'),
        (148, -38, 'SJ55.zip'),
        (53, 39, 'J39.zip'),
        (27, -67, None),
        (57, 20, 'F40.zip'),
        (109, -31, None),
        (-143, 74, None),
        ]

    with srtm.SrtmConf.set(server='viewpano'):

        for ilon, ilat, name in pano_cases:

            if name is None:
                with pytest.raises(srtm.TileNotAvailableOnServerError):
                    srtm._check_availability(ilon, ilat)
            else:
                assert srtm._check_availability(ilon, ilat) == name


@pytest.mark.skip(reason="NASA tiles not available without log-in anymore")
@pytest.mark.remote_data
def test_download_nasa(srtm_temp_dir):

    ilon, ilat = 6, 50
    tile_name = srtm._hgt_filename(ilon, ilat)

    with srtm.SrtmConf.set(srtm_dir=srtm_temp_dir, server='nasa_v2.1'):

        srtm._download(ilon, ilat)

        dl_path = srtm._get_hgt_diskpath(tile_name)

        assert dl_path is not None

        assert dl_path.startswith(srtm_temp_dir)
        assert dl_path.endswith(tile_name)


@pytest.mark.remote_data
def test_download_pano(srtm_temp_dir):

    ilon, ilat = 6, 50
    tile_name = srtm._hgt_filename(ilon, ilat)

    with srtm.SrtmConf.set(srtm_dir=srtm_temp_dir, server='viewpano'):

        srtm._download(ilon, ilat)

        dl_path = srtm._get_hgt_diskpath(tile_name)

        assert dl_path is not None

        assert dl_path.startswith(srtm_temp_dir)
        assert dl_path.endswith(tile_name)

    ilon, ilat = -175, -4
    tile_name = srtm._hgt_filename(ilon, ilat)

    with srtm.SrtmConf.set(srtm_dir=srtm_temp_dir, server='viewpano'):

        srtm._download(ilon, ilat)

        dl_path = srtm._get_hgt_diskpath(tile_name)

        assert dl_path is not None

        assert dl_path.startswith(srtm_temp_dir)
        assert dl_path.endswith(tile_name)


def test_get_hgt_diskpath(srtm_temp_dir):

    # getting the correct files was already tested above
    # checking the behavior for problematic cases

    with srtm.SrtmConf.set(srtm_dir=srtm_temp_dir):

        assert srtm._get_hgt_diskpath('foo.hgt') is None

        os.makedirs(os.path.join(srtm_temp_dir, 'd1'))
        os.makedirs(os.path.join(srtm_temp_dir, 'd2'))
        open(os.path.join(srtm_temp_dir, 'd1', 'foo.hgt'), 'w').close()
        open(os.path.join(srtm_temp_dir, 'd2', 'foo.hgt'), 'w').close()

        with pytest.raises(IOError, match=r'.* exists .* times in .*'):
            srtm._get_hgt_diskpath('foo.hgt')

        # cleaning up
        os.remove(os.path.join(srtm_temp_dir, 'd1', 'foo.hgt'))
        os.remove(os.path.join(srtm_temp_dir, 'd2', 'foo.hgt'))


@pytest.mark.remote_data
def test_get_hgt_file_download_never(srtm_temp_dir):

    print(srtm.SrtmConf.srtm_dir)
    with srtm.SrtmConf.set(srtm_dir=srtm_temp_dir):

        ilon, ilat = 6, 50
        tile_name = srtm._hgt_filename(ilon, ilat)
        tile_path = srtm.get_hgt_file(ilon, ilat)

        assert tile_path.endswith(tile_name)

        ilon, ilat = -175, -4
        tile_name = srtm._hgt_filename(ilon, ilat)
        tile_path = srtm.get_hgt_file(ilon, ilat)

        assert tile_path.endswith(tile_name)

        ilon, ilat = 12, 50
        tile_name = srtm._hgt_filename(ilon, ilat)

        with pytest.raises(
                srtm.TileNotAvailableOnDiskError,
                match=r'.*No hgt-file found for .*'
                ):
            srtm.get_hgt_file(ilon, ilat)


@pytest.mark.remote_data
def test_get_hgt_file_download_missing(srtm_temp_dir):

    print(srtm.SrtmConf.srtm_dir)
    with srtm.SrtmConf.set(srtm_dir=srtm_temp_dir, download='missing'):

        ilon, ilat = 12, 50
        tile_name = srtm._hgt_filename(ilon, ilat)
        tile_path = srtm.get_hgt_file(ilon, ilat)

        assert tile_path.endswith(tile_name)


@pytest.mark.remote_data
def test_get_hgt_file_download_always(srtm_temp_dir):

    # note, previously, we checked the file's mtime to do this check
    # however, on macos, the mtime is often the same (perhaps because
    # of bad granularity?)
    ilon, ilat = 12, 50
    dat1 = b'W\x04'  # == 1111 as short (struct type: 'h')
    with srtm.SrtmConf.set(srtm_dir=srtm_temp_dir):

        tile_path = srtm.get_hgt_file(ilon, ilat)
        # manually modify the first datum:
        with open(tile_path, 'r+b') as f:
            f.write(dat1)

        _, _, tile1 = srtm.get_tile_data(ilon, ilat)

    with srtm.SrtmConf.set(srtm_dir=srtm_temp_dir, download='always'):

        ilon, ilat = 12, 50
        tile_path = srtm.get_hgt_file(ilon, ilat)
        with open(tile_path, 'rb') as f:
            dat2 = f.read(2)

    print(dat1, dat2)
    assert dat1 != dat2


@pytest.mark.remote_data
def test_get_tile_data(srtm_temp_dir):

    with srtm.SrtmConf.set(srtm_dir=srtm_temp_dir):

        ilon, ilat = 12, 50
        lons, lats, tile = srtm.get_tile_data(ilon, ilat)

        assert_allclose(lons[::250, 0], np.array([
            12., 12.20833333, 12.41666667, 12.625, 12.83333333
            ]))
        assert_allclose(lats[0, ::250], np.array([
            50., 50.20833333, 50.41666667, 50.625, 50.83333333
            ]))
        assert_allclose(tile[::250, ::250], np.array([
            [776., 543., 542., 622., 652.],
            [562., 641., 470., 471., 480.],
            [522., 487., 733., 939., 970.],
            [466., 359., 454., 518., 560.],
            [335., 319., 255., 342., 339.]
            ]))


def test_get_tile_zero(srtm_temp_dir):

    with srtm.SrtmConf.set(srtm_dir=srtm_temp_dir):

        # ilon, ilat = 6, 54
        ilon, ilat = 28, 35
        lons, lats, tile = srtm.get_tile_data(ilon, ilat)

        assert_allclose(lons[:, 0], np.array([
            28., 28.25, 28.5, 28.75, 29.
            ]))
        assert_allclose(lats[0, :], np.array([
            35., 35.25, 35.5, 35.75, 36.
            ]))
        assert_allclose(tile, np.zeros((5, 5), dtype=np.float32))


def test_get_tile_warning(srtm_temp_dir):

    with srtm.SrtmConf.set(srtm_dir=srtm_temp_dir):

        # ilon, ilat = 6, 54
        ilon, ilat = 15, 47
        with pytest.warns(srtm.TileNotAvailableOnDiskWarning):
            lons, lats, tile = srtm.get_tile_data(ilon, ilat)

        assert_allclose(lons[:, 0], np.array([
            15., 15.25, 15.5, 15.75, 16.
            ]))
        assert_allclose(lats[0, :], np.array([
            47., 47.25, 47.5, 47.75, 48.
            ]))
        assert_allclose(tile, np.zeros((5, 5), dtype=np.float32))


@pytest.mark.remote_data
def test_srtm_height_data_linear(srtm_temp_dir):

    args_list = [
        (-180, 180, apu.deg),
        (-90, 90, apu.deg),
        ]
    check_astro_quantities(srtm.srtm_height_data, args_list)

    with srtm.SrtmConf.set(srtm_dir=srtm_temp_dir, interp='linear'):

        # lons = np.arange(12.1, 12.91, 0.2) * apu.deg
        # lats = np.arange(50.1, 50.91, 0.2)[:, np.newaxis] * apu.deg

        lons, lats = np.meshgrid(
            np.arange(12.1005, 12.9, 0.2),
            np.arange(50.1005, 50.9, 0.2)
            )
        # heights = srtm.srtm_height_data(lons * apu.deg, lats * apu.deg)
        heights = srtm.srtm_height_data(
            lons.flatten() * apu.deg, lats.flatten() * apu.deg
            ).reshape(lons.shape)

        assert_quantity_allclose(heights, np.array([
            [581.71997070, 484.48001099, 463.79998779, 736.44000244],
            [613.00000000, 549.88000488, 636.52001953, 678.91998291],
            [433.44000244, 416.20001221, 704.52001953, 826.08001709],
            [358.72000122, 395.55999756, 263.83999634, 469.39999390]
            ]) * apu.m)


@pytest.mark.remote_data
def test_srtm_height_data_nearest(srtm_temp_dir):

    args_list = [
        (-180, 180, apu.deg),
        (-90, 90, apu.deg),
        ]
    check_astro_quantities(srtm.srtm_height_data, args_list)

    with srtm.SrtmConf.set(srtm_dir=srtm_temp_dir, interp='nearest'):

        lons, lats = np.meshgrid(
            np.arange(12.1005, 12.9, 0.2),
            np.arange(50.1005, 50.9, 0.2)
            )
        heights = srtm.srtm_height_data(
            lons.flatten() * apu.deg, lats.flatten() * apu.deg
            ).reshape(lons.shape)

        assert_quantity_allclose(heights, np.array([
            [583., 484., 463., 739.],
            [613., 543., 641., 685.],
            [432., 415., 699., 828.],
            [358., 397., 262., 471.]
            ]) * apu.m)


@pytest.mark.remote_data
def test_srtm_height_data_spline(srtm_temp_dir):

    args_list = [
        (-180, 180, apu.deg),
        (-90, 90, apu.deg),
        ]
    check_astro_quantities(srtm.srtm_height_data, args_list)

    with srtm.SrtmConf.set(srtm_dir=srtm_temp_dir, interp='spline'):

        lons, lats = np.meshgrid(
            np.arange(12.1005, 12.9, 0.2),
            np.arange(50.1005, 50.9, 0.2)
            )
        heights = srtm.srtm_height_data(
            lons.flatten() * apu.deg, lats.flatten() * apu.deg
            ).reshape(lons.shape)

        assert_quantity_allclose(heights, np.array([
            [581.39044189, 484.20700073, 463.94418335, 734.95751953],
            [613.10083008, 550.10040283, 637.01745605, 678.44708252],
            [432.46701050, 416.11437988, 704.96179199, 826.47576904],
            [358.81408691, 395.84069824, 262.50534058, 471.27304077]
            ]) * apu.m)


def test_srtm_height_data_zero(srtm_temp_dir):

    with srtm.SrtmConf.set(srtm_dir=srtm_temp_dir):

        lons = np.arange(28.1, 28.91, 0.2)
        lats = np.arange(35.1, 35.91, 0.2)
        heights = srtm._srtm_height_data(lons, lats)

        assert_allclose(heights, np.zeros(5, dtype=np.float32))


@pytest.mark.remote_data
def test_srtm_height_data_broadcasting(srtm_temp_dir):

    args_list = [
        (-180, 180, apu.deg),
        (-90, 90, apu.deg),
        ]
    check_astro_quantities(srtm.srtm_height_data, args_list)

    with srtm.SrtmConf.set(srtm_dir=srtm_temp_dir, interp='linear'):

        lons = np.arange(12.1005, 12.9, 0.2) * apu.deg
        lats = np.arange(50.1005, 50.9, 0.2)[:, np.newaxis] * apu.deg
        heights = srtm.srtm_height_data(lons, lats)

        assert_quantity_allclose(heights, np.array([
            [581.71997070, 484.48001099, 463.79998779, 736.44000244],
            [613.00000000, 549.88000488, 636.52001953, 678.91998291],
            [433.44000244, 416.20001221, 704.52001953, 826.08001709],
            [358.72000122, 395.55999756, 263.83999634, 469.39999390]
            ]) * apu.m)

        heights = srtm.srtm_height_data(lons, lats.reshape((2, 2, 1)))

        assert_quantity_allclose(heights, np.array([
            [[581.71997070, 484.48001099, 463.79998779, 736.44000244],
             [613.00000000, 549.88000488, 636.52001953, 678.91998291]],
            [[433.44000244, 416.20001221, 704.52001953, 826.08001709],
             [358.72000122, 395.55999756, 263.83999634, 469.39999390]]
            ]) * apu.m)


# ---------------------------------------------------------------------------
# Failure-mode hardening (missing tiles, void masking)
# ---------------------------------------------------------------------------

def _write_hgt(path, tile_size, fill, void_positions=(), void_value=-32768):
    '''Write a minimal big-endian int16 SRTM ".hgt" tile.'''

    arr = np.full((tile_size, tile_size), fill, dtype='>i2')
    for pos in void_positions:
        arr[pos] = void_value
    arr.astype('>i2').tofile(path)


@pytest.mark.parametrize('void_value', [-32768, -32767, 32767])
def test_srtm_void_is_masked(srtm_temp_dir, void_value):
    # regression test: the canonical SRTM void sentinel is -32768; make sure
    # all of -32768/-32767/+32767 are masked (the historic code only handled
    # -32767/+32767, so genuine -32768 voids leaked through as huge values)

    tdir = os.path.join(srtm_temp_dir, 'voidtest_{}'.format(void_value))
    os.makedirs(tdir, exist_ok=True)
    # N50E006 (ilon=6, ilat=50) is a valid viewpano tile
    _write_hgt(
        os.path.join(tdir, 'N50E006.hgt'), 6, 100,
        void_positions=[(2, 3)], void_value=void_value,
        )

    with srtm.SrtmConf.set(srtm_dir=tdir, server='viewpano'):
        srtm.get_tile_interpolator.cache_clear()
        _, _, tile = srtm.get_tile_data(6, 50)

        assert np.isnan(tile).sum() == 1
        # no sentinel value survived into the (float) tile
        assert np.nanmin(tile) == 100 and np.nanmax(tile) == 100


@pytest.mark.parametrize('on_missing', ['zeros', 'raise'])
def test_on_missing_behaviour(srtm_temp_dir, on_missing):
    # a tile that is "available on the server" but not on disk, with
    # download='never', either warns+zeros or raises, depending on on_missing

    tdir = os.path.join(srtm_temp_dir, 'missing_{}'.format(on_missing))
    os.makedirs(tdir, exist_ok=True)

    # 15E, 47N is not a viewpano tile -> would be a *server* miss (silent
    # zeros); use 6E, 50N which *is* a viewpano tile -> a *disk* miss
    with srtm.SrtmConf.set(
            srtm_dir=tdir, server='viewpano',
            download='never', on_missing=on_missing,
            ):
        srtm.get_tile_interpolator.cache_clear()

        if on_missing == 'raise':
            with pytest.raises(srtm.TileNotAvailableOnDiskError):
                srtm.get_tile_data(6, 50)
        else:
            with pytest.warns(srtm.TileNotAvailableOnDiskWarning):
                _, _, tile = srtm.get_tile_data(6, 50)
            assert_allclose(tile, np.zeros((5, 5), dtype=np.float32))


# ---------------------------------------------------------------------------
# Copernicus DEM backend
# ---------------------------------------------------------------------------

def test_copernicus_tilename():

    cases = [
        (6, 45, 'copernicus_glo90',
         'Copernicus_DSM_COG_30_N45_00_E006_00_DEM'),
        (109, -31, 'copernicus_glo90',
         'Copernicus_DSM_COG_30_S31_00_E109_00_DEM'),
        (-97, 51, 'copernicus_glo90',
         'Copernicus_DSM_COG_30_N51_00_W097_00_DEM'),
        (-46, -22, 'copernicus_glo90',
         'Copernicus_DSM_COG_30_S22_00_W046_00_DEM'),
        (6, 45, 'copernicus_glo30',
         'Copernicus_DSM_COG_10_N45_00_E006_00_DEM'),
        ]

    for ilon, ilat, server, name in cases:
        assert srtm._copernicus_tilename(ilon, ilat, server) == name


def _write_cop_tile(
        path, ilon, ilat, nlon, nlat, dx_asec, dy_asec, values, nodata=None
        ):
    '''Write a synthetic Copernicus-style COG (pixel-centre registration).

    The pixel *centre* of the north-west pixel is placed exactly at
    (ilon, ilat + 1), matching the real Copernicus tiles (named by their
    south-west corner).
    '''

    import rasterio

    dx = dx_asec / 3600.
    dy = dy_asec / 3600.
    west_edge = ilon - 0.5 * dx
    north_edge = (ilat + 1) + 0.5 * dy
    transform = rasterio.Affine(dx, 0.0, west_edge, 0.0, -dy, north_edge)

    kwargs = dict(
        driver='GTiff', height=nlat, width=nlon, count=1,
        dtype='float32', crs='EPSG:4326', transform=transform,
        )
    if nodata is not None:
        kwargs['nodata'] = nodata

    with rasterio.open(path, 'w', **kwargs) as dst:
        dst.write(values.astype('float32'), 1)


@skip_rasterio
def test_copernicus_registration(srtm_temp_dir):

    tdir = os.path.join(srtm_temp_dir, 'cop_reg')
    os.makedirs(tdir, exist_ok=True)

    # coarse full-degree tile at SW corner (6, 45); values encode (row, col)
    nlon = nlat = 12
    dasec = 300.  # 5 arcmin -> 12 px per degree
    rows, cols = np.mgrid[0:nlat, 0:nlon]
    values = 1000. + 10. * rows + cols  # row 0 = north
    _write_cop_tile(
        os.path.join(tdir, 'Copernicus_DSM_COG_30_N45_00_E006_00_DEM.tif'),
        6, 45, nlon, nlat, dasec, dasec, values,
        )

    with srtm.SrtmConf.set(
            srtm_dir=tdir, server='copernicus_glo90',
            download='never', interp='nearest',
            ):
        srtm.get_tile_interpolator.cache_clear()

        lons, lats, tile = srtm.get_tile_data(6, 45)

        # pixel-centre registration: NW pixel centre at (6, 46)
        assert tile.shape == (nlat, nlon)
        assert_allclose(lons[0, 0], 6.0)             # west-most centre
        assert_allclose(lats[0, -1], 46.0)           # north-most centre
        # hgt_res inferred from the tile (5 arcmin ~ 9.3 km here)
        assert srtm.SrtmConf.hgt_res > 0

        # query an exact pixel centre (stored row 3, col 5 -> value 1035)
        dx = dasec / 3600.
        lon_q = 6.0 + 5 * dx
        lat_q = 46.0 - 3 * dx
        h = srtm.srtm_height_data([lon_q] * apu.deg, [lat_q] * apu.deg)
        assert_quantity_allclose(h, [1035.] * apu.m)


@skip_rasterio
def test_copernicus_variable_spacing(srtm_temp_dir):
    # above |lat| 50 deg the longitude spacing widens; tiles are not square
    # and the reader must use the actual geotransform

    tdir = os.path.join(srtm_temp_dir, 'cop_var')
    os.makedirs(tdir, exist_ok=True)

    nlon, nlat = 4, 12
    dx_asec, dy_asec = 900., 300.  # 9 arcsec would be a GLO-90 70-75 deg tile
    values = np.arange(nlon * nlat, dtype='float32').reshape(nlat, nlon)
    _write_cop_tile(
        os.path.join(tdir, 'Copernicus_DSM_COG_30_N70_00_E020_00_DEM.tif'),
        20, 70, nlon, nlat, dx_asec, dy_asec, values,
        )

    with srtm.SrtmConf.set(
            srtm_dir=tdir, server='copernicus_glo90', download='never',
            ):
        srtm.get_tile_interpolator.cache_clear()

        lons, lats, tile = srtm.get_tile_data(20, 70)

        assert tile.shape == (nlat, nlon)
        assert lons.shape == (nlon, 1)
        assert lats.shape == (1, nlat)
        # longitude spacing != latitude spacing
        assert_allclose((lons[1, 0] - lons[0, 0]) * 3600., dx_asec)
        assert_allclose((lats[0, 1] - lats[0, 0]) * 3600., dy_asec)


@skip_rasterio
@pytest.mark.parametrize('void_fill', ['zero', 'nan', 'interp'])
def test_copernicus_void_fill(srtm_temp_dir, void_fill):

    tdir = os.path.join(srtm_temp_dir, 'cop_void_{}'.format(void_fill))
    os.makedirs(tdir, exist_ok=True)

    nlon = nlat = 12
    values = np.full((nlat, nlon), 200., dtype='float32')
    values[5, 5] = np.nan          # NaN void
    values[6, 6] = -1000.          # sea/void sentinel (< -500)
    _write_cop_tile(
        os.path.join(tdir, 'Copernicus_DSM_COG_30_N45_00_E006_00_DEM.tif'),
        6, 45, nlon, nlat, 300., 300., values,
        )

    with srtm.SrtmConf.set(
            srtm_dir=tdir, server='copernicus_glo90',
            download='never', interp='nearest', void_fill=void_fill,
            ):
        srtm.get_tile_interpolator.cache_clear()

        _, _, tile = srtm.get_tile_data(6, 45)
        # both the NaN and the < -500 sentinel are masked to NaN on read
        assert np.isnan(tile).sum() == 2

        dx = 300. / 3600.
        # stored row 5, col 5 -> the NaN void
        lon_q = 6.0 + 5 * dx
        lat_q = 46.0 - 5 * dx
        h = srtm.srtm_height_data(
            [lon_q] * apu.deg, [lat_q] * apu.deg
            ).to_value(apu.m)

        if void_fill == 'zero':
            assert_allclose(h, [0.])
        elif void_fill == 'nan':
            assert np.isnan(h).all()
        elif void_fill == 'interp':
            # filled from the nearest valid pixel (200 m)
            assert_allclose(h, [200.])


@skip_rasterio
def test_copernicus_check_availability(srtm_temp_dir):
    # with a cached tileList, tiles not in the list are treated as ocean
    # (silent zeros); tiles in the list but missing on disk hit on_missing

    tdir = os.path.join(srtm_temp_dir, 'cop_avail')
    os.makedirs(tdir, exist_ok=True)
    with open(
            os.path.join(tdir, 'copernicus_glo90_tileList.txt'), 'w'
            ) as f:
        f.write('Copernicus_DSM_COG_30_N45_00_E006_00_DEM\n')

    with srtm.SrtmConf.set(
            srtm_dir=tdir, server='copernicus_glo90', download='never',
            ):
        srtm.get_tile_interpolator.cache_clear()

        # not in the list -> ocean -> silent zeros
        _, _, tile = srtm.get_tile_data(0, 0)
        assert_allclose(tile, np.zeros((5, 5), dtype=np.float32))

        # in the list but not on disk -> disk miss
        with srtm.SrtmConf.set(on_missing='raise'):
            with pytest.raises(srtm.TileNotAvailableOnDiskError):
                srtm.get_tile_data(6, 45)


@pytest.mark.remote_data
@skip_rasterio
def test_copernicus_download(srtm_temp_dir):

    tdir = os.path.join(srtm_temp_dir, 'cop_dl')
    os.makedirs(tdir, exist_ok=True)

    ilon, ilat = 6, 45
    tif_name = srtm._copernicus_tilename(
        ilon, ilat, 'copernicus_glo90'
        ) + '.tif'

    with srtm.SrtmConf.set(
            srtm_dir=tdir, server='copernicus_glo90', download='missing',
            ):
        srtm.get_tile_interpolator.cache_clear()

        dl_path = srtm.get_copernicus_file(ilon, ilat)
        assert dl_path is not None
        assert dl_path.endswith(tif_name)


@pytest.mark.remote_data
@skip_rasterio
def test_copernicus_get_tile_data(srtm_temp_dir):
    # value check against a real downloaded GLO-90 tile (Mont Blanc region),
    # analogous to test_get_tile_data for SRTM. Note the pixel-centre (area)
    # registration: the south-most row centre lies half a pixel plus the
    # removed shared edge above the SW corner (45.00083), the north-most row
    # centre exactly at 46.0.

    tdir = os.path.join(srtm_temp_dir, 'cop_tiledata')
    os.makedirs(tdir, exist_ok=True)

    with srtm.SrtmConf.set(
            srtm_dir=tdir, server='copernicus_glo90', download='missing',
            ):
        srtm.get_tile_interpolator.cache_clear()

        ilon, ilat = 6, 45
        lons, lats, tile = srtm.get_tile_data(ilon, ilat)

        assert tile.shape == (1200, 1200)

        assert_allclose(lons[::250, 0], np.array([
            6., 6.20833333, 6.41666667, 6.625, 6.83333333
            ]))
        assert_allclose(lats[0, ::250], np.array([
            45.00083333, 45.20916667, 45.4175, 45.62583333, 45.83416667
            ]))
        assert_allclose(tile[::250, ::250], np.array([
            [2443.5603, 3120.2942, 2775.819, 2283.7593, 1530.1218],
            [2173.671, 2129.5164, 1736.1337, 1493.1702, 1649.1984],
            [248.32872, 2015.355, 2559.966, 1861.7087, 2698.8232],
            [1093.4408, 1177.1719, 1019.66125, 2303.7588, 1472.1299],
            [370., 445.5, 1134.9672, 1501.6011, 3701.4497]
            ]), rtol=1.e-6)


@pytest.mark.remote_data
@skip_rasterio
def test_copernicus_registration_real(srtm_temp_dir):
    # Mont Blanc summit (45.8326 N, 6.8652 E); GLO-90 reads ~4790 m there
    # (a wrong half-pixel registration would miss the narrow summit)

    tdir = os.path.join(srtm_temp_dir, 'cop_real')
    os.makedirs(tdir, exist_ok=True)

    with srtm.SrtmConf.set(
            srtm_dir=tdir, server='copernicus_glo90',
            download='missing', interp='linear',
            ):
        srtm.get_tile_interpolator.cache_clear()

        h = srtm.srtm_height_data(
            [6.8652] * apu.deg, [45.8326] * apu.deg
            ).to_value(apu.m)[0]

        assert 4700. < h < 4810.


@pytest.mark.remote_data
@skip_rasterio
def test_copernicus_vs_srtm_overlap(srtm_temp_dir):
    # cross-check GLO-90 against the viewpano SRTM tile on an overlapping
    # land cell (French Alps). The two independent DEMs must be highly
    # correlated; we use the correlation coefficient rather than an absolute
    # tolerance, so the result is insensitive to the EGM96/EGM2008 datum
    # offset (a few metres) between them.

    lons, lats = np.meshgrid(
        np.arange(6.2, 6.8, 0.02),
        np.arange(45.2, 45.8, 0.02),
        )
    lons = lons.flatten() * apu.deg
    lats = lats.flatten() * apu.deg

    tdir_s = os.path.join(srtm_temp_dir, 'xcheck_srtm')
    tdir_c = os.path.join(srtm_temp_dir, 'xcheck_cop')
    os.makedirs(tdir_s, exist_ok=True)
    os.makedirs(tdir_c, exist_ok=True)

    with srtm.SrtmConf.set(
            srtm_dir=tdir_s, server='viewpano', download='missing',
            ):
        srtm.get_tile_interpolator.cache_clear()
        h_srtm = srtm.srtm_height_data(lons, lats).to_value(apu.m)

    with srtm.SrtmConf.set(
            srtm_dir=tdir_c, server='copernicus_glo90', download='missing',
            ):
        srtm.get_tile_interpolator.cache_clear()
        h_cop = srtm.srtm_height_data(lons, lats).to_value(apu.m)

    corr = np.corrcoef(h_srtm, h_cop)[0, 1]
    assert corr > 0.99
    # and the median absolute difference should be small (metre-to-decametre
    # level over this relief)
    assert np.median(np.abs(h_srtm - h_cop)) < 40.
