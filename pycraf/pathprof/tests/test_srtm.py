#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from astropy.tests.helper import assert_quantity_allclose, remote_data
from astropy import units as apu
from ...pathprof import srtm
from ...utils import check_astro_quantities


TOL_KWARGS = {'atol': 1.e-4, 'rtol': 1.e-4}


class TestSrtmConf:

    def setup(self):

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
@remote_data(source='any')
def test_download_nasa(srtm_temp_dir):

    ilon, ilat = 6, 50
    tile_name = srtm._hgt_filename(ilon, ilat)

    with srtm.SrtmConf.set(srtm_dir=srtm_temp_dir, server='nasa_v2.1'):

        srtm._download(ilon, ilat)

        dl_path = srtm._get_hgt_diskpath(tile_name)

        assert dl_path is not None

        assert dl_path.startswith(srtm_temp_dir)
        assert dl_path.endswith(tile_name)


@remote_data(source='any')
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


@remote_data(source='any')
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


@remote_data(source='any')
def test_get_hgt_file_download_missing(srtm_temp_dir):

    print(srtm.SrtmConf.srtm_dir)
    with srtm.SrtmConf.set(srtm_dir=srtm_temp_dir, download='missing'):

        ilon, ilat = 12, 50
        tile_name = srtm._hgt_filename(ilon, ilat)
        tile_path = srtm.get_hgt_file(ilon, ilat)

        assert tile_path.endswith(tile_name)


@remote_data(source='any')
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


@remote_data(source='any')
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


@remote_data(source='any')
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


@remote_data(source='any')
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


@remote_data(source='any')
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


@remote_data(source='any')
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
