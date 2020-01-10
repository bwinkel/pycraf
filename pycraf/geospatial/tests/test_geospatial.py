#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from astropy.tests.helper import assert_quantity_allclose
from astropy import units as apu
from astropy.units import Quantity
from astropy.utils.misc import NumpyRNGContext
from astropy.utils.data import get_pkg_data_filename
from ... import conversions as cnv
from ... import geospatial as gsp
import importlib


# skip over pyproj related tests (currently all), if package not present:
skip_pyproj = pytest.mark.skipif(
    importlib.util.find_spec('pyproj') is None,
    reason='"pyproj" package not installed'
    )


UTM_ZONES = ['1N', '32N', '60N']
ITRF_YEARS = ['2005', '2008']
GK_EPSG = [31466, 31467, 31468]


def produce_utm_test_cases():

    with NumpyRNGContext(1):
        _glon = np.random.uniform(-180., -174., 20) * apu.deg
        glat = np.random.uniform(0., 84., 20) * apu.deg

    for zone in UTM_ZONES:

        glon = _glon + (int(zone[:-1]) - 1) * 6. * apu.deg
        ulon, ulat = gsp.wgs84_to_utm(glon, glat, zone)

        np.savez(
            '/tmp/wgs84_utm_zone{}.npz'.format(zone),
            glon=glon.value, glat=glat.value,
            ulon=ulon.value, ulat=ulat.value,
            )


def produce_etrs89_test_cases():

    with NumpyRNGContext(1):
        glon = np.random.uniform(-20., 40., 20) * apu.deg
        glat = np.random.uniform(0., 70., 20) * apu.deg

    elon, elat = gsp.wgs84_to_etrs89(glon, glat)

    np.savez(
        '/tmp/wgs84_etrs89.npz',
        glon=glon.value, glat=glat.value,
        elon=elon.value, elat=elat.value,
        )


def produce_itrf_test_cases():

    with NumpyRNGContext(1):
        glon = np.random.uniform(-180., 180., 20) * apu.deg
        glat = np.random.uniform(-90., 90., 20) * apu.deg
        height = np.random.uniform(0., 500., 20) * apu.m

    for year in ITRF_YEARS:

        func = getattr(gsp, 'wgs84_to_itrf{}'.format(year))
        x, y, z = func(glon, glat, height)

        np.savez(
            '/tmp/wgs84_itrf{}.npz'.format(year),
            glon=glon.value, glat=glat.value, height=height.value,
            x=x.value, y=y.value, z=z.value,
            )


def produce_gauss_kruger_case():

    with NumpyRNGContext(1):
        ulon = np.random.uniform(3.41e6, 3.61e6, 20) * apu.m
        ulat = np.random.uniform(5.24e6, 6.10e6, 20) * apu.m

        glon, glat = gsp.transform_factory(
            31467, gsp.geospatial.EPSG.WGS84
            )(ulon, ulat)

        np.savez(
            '/tmp/wgs84_gauss_kruger_epsg31467.npz',
            glon=glon.value, glat=glat.value,
            ulon=ulon.value, ulat=ulat.value,
            )


# Warning: if you want to produce new test cases (replacing the old ones)
# you better make sure, that everything is 100% correct
# produce_utm_test_cases()
# produce_etrs89_test_cases()
# produce_itrf_test_cases()
# produce_gauss_kruger_case()


def test_utm_zone_from_gps():

    for glon, glat, utm_zone in [
            (44.1, 10.3, '38N'),
            (-72.4, -52.7, '18S'),
            (153.1, 52.5, '56N'),
            (156.4, -67.3, '57S'),
            (-175.9, -51.1, '1S'),
            (282.1, 43.1, '18N'),
            (67.9, 4.7, '42N'),
            (130.6, 63.8, '52N'),
            (-152.2, 56.7, '5N'),
            (42.8, -78.5, '38S'),
            ]:

        assert gsp.geospatial.utm_zone_from_gps(
            glon * apu.deg, glat * apu.deg
            ) == utm_zone


def test_utm_zone_from_gps_broadcast():

    glons = np.linspace(-100, 100, 5)
    glats = np.linspace(-50, 50, 5)

    assert_equal(
        gsp.geospatial.utm_zone_from_gps(
            glons * apu.deg, glats[:, np.newaxis] * apu.deg
            ),
        np.array([
            ['14S', '22S', '31S', '39S', '47S'],
            ['14S', '22S', '31S', '39S', '47S'],
            ['14N', '22N', '31N', '39N', '47N'],
            ['14N', '22N', '31N', '39N', '47N'],
            ['14N', '22N', '31N', '39N', '47N']
            ])
        )


def test_epsg_from_utm_zone():

    for utm_zone, epsg in [
            ('38N', 32638), ('18S', 32718), ('56N', 32656),
            ('57S', 32757), ('1S', 32701), ('18N', 32618),
            ]:

        assert gsp.geospatial.epsg_from_utm_zone(utm_zone) == epsg

    for utm_zone in ['-1N', '0N', '61N', '100N', '-1S', '0S', '61S', '100S']:

        with pytest.raises(ValueError):
            gsp.geospatial.epsg_from_utm_zone(utm_zone)


class TestTransformations:

    @skip_pyproj
    def test_create_transform(self):

        import pyproj

        _create_transform = gsp.geospatial._create_transform

        wgs84 = gsp.geospatial.EPSG.WGS84
        etrs89 = gsp.geospatial.EPSG.ETRS89
        etrs89_str = ('+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 '
                      '+y_0=3210000 +ellps=GRS80 +units=m +no_defs')

        with pytest.raises(TypeError):
            _create_transform(None, None)

        with pytest.raises(TypeError):
            _create_transform(1., 1.)

        # For some reason, this is no longer invalid, but returns
        # Proj('+proj=longlat +ellps=bessel +no_defs', preserve_units=True), Proj('+proj=longlat +a=6378249.145 +rf=293.465 +no_defs', preserve_units=True))
        # with pytest.raises(RuntimeError):
        #     _create_transform('FOO', 'BAR')

        with pytest.raises(ValueError):
            _create_transform(wgs84, etrs89, code_in='foo')

        test_list = [
            ((wgs84, etrs89), {}),
            ((wgs84, 3035), {}),
            ((wgs84, etrs89_str), {}),
            ]
        if pyproj.__version__ < '2.0':
            test_list.append(((wgs84, 54004), {'code_out': 'esri'}))
        for args, kwargs in test_list:

            func = _create_transform(*args, **kwargs)[0]
            assert callable(func)

    @skip_pyproj
    def test_wgs84_to_utm(self):

        template = 'data/wgs84_utm_zone{}.npz'

        for zone in UTM_ZONES:

            dat = np.load(get_pkg_data_filename(template.format(zone)))

            ulon, ulat = gsp.wgs84_to_utm(
                dat['glon'] * apu.deg, dat['glat'] * apu.deg, zone
                )

            assert_quantity_allclose(ulon, dat['ulon'] * apu.m)
            assert_quantity_allclose(ulat, dat['ulat'] * apu.m)

    @skip_pyproj
    def test_utm_to_wgs84(self):

        template = 'data/wgs84_utm_zone{}.npz'

        for zone in UTM_ZONES:

            dat = np.load(get_pkg_data_filename(template.format(zone)))

            glon, glat = gsp.utm_to_wgs84(
                dat['ulon'] * apu.m, dat['ulat'] * apu.m, zone
                )

            assert_quantity_allclose(glon, dat['glon'] * apu.deg)
            assert_quantity_allclose(glat, dat['glat'] * apu.deg)

    @skip_pyproj
    def test_wgs84_to_etrs89(self):

        dat = np.load(get_pkg_data_filename('data/wgs84_etrs89.npz'))

        elon, elat = gsp.wgs84_to_etrs89(
            dat['glon'] * apu.deg, dat['glat'] * apu.deg
            )

        assert_quantity_allclose(elon, dat['elon'] * apu.m)
        assert_quantity_allclose(elat, dat['elat'] * apu.m)

    @skip_pyproj
    def test_etrs89_to_wgs84(self):

        dat = np.load(get_pkg_data_filename('data/wgs84_etrs89.npz'))

        glon, glat = gsp.etrs89_to_wgs84(
            dat['elon'] * apu.m, dat['elat'] * apu.m
            )

        assert_quantity_allclose(glon, dat['glon'] * apu.deg)
        assert_quantity_allclose(glat, dat['glat'] * apu.deg)

    @skip_pyproj
    def test_wgs84_to_itrf(self):

        template = 'data/wgs84_itrf{}.npz'
        for year in ITRF_YEARS:

            func = getattr(gsp, 'wgs84_to_itrf{}'.format(year))
            dat = np.load(get_pkg_data_filename(template.format(year)))

            x, y, z = func(
                dat['glon'] * apu.deg, dat['glat'] * apu.deg,
                dat['height'] * apu.m
                )

            assert_quantity_allclose(x, dat['x'] * apu.m, atol=1.e-3 * apu.m)
            assert_quantity_allclose(y, dat['y'] * apu.m, atol=1.e-3 * apu.m)
            assert_quantity_allclose(z, dat['z'] * apu.m, atol=1.e-3 * apu.m)

    @skip_pyproj
    def test_itrf_to_wgs84(self):

        template = 'data/wgs84_itrf{}.npz'

        for year in ITRF_YEARS:

            func = getattr(gsp, 'itrf{}_to_wgs84'.format(year))
            dat = np.load(get_pkg_data_filename(template.format(year)))

            glon, glat, height = func(
                dat['x'] * apu.m, dat['y'] * apu.m, dat['z'] * apu.m
                )

            assert_quantity_allclose(glon, dat['glon'] * apu.deg)
            assert_quantity_allclose(glat, dat['glat'] * apu.deg)
            assert_quantity_allclose(
                height, dat['height'] * apu.m, atol=0.0002 * apu.m
                )

    @skip_pyproj
    @pytest.mark.skip(
        reason="Different proj4 version lead to different results. "
        "Needs check."
        )
    def test_wgs84_to_gk31467(self):

        dat = np.load(get_pkg_data_filename(
            'data/wgs84_gauss_kruger_epsg31467.npz'
            ))

        transform = gsp.transform_factory(
            gsp.geospatial.EPSG.WGS84, 31467
            )
        ulon, ulat = transform(
            dat['glon'] * apu.deg, dat['glat'] * apu.deg
            )

        assert_quantity_allclose(ulon, dat['ulon'] * apu.m)
        assert_quantity_allclose(ulat, dat['ulat'] * apu.m)

    @skip_pyproj
    @pytest.mark.skip(
        reason="Different proj4 version lead to different results. "
        "Needs check."
        )
    def test_gk31467_to_wgs84(self):

        dat = np.load(get_pkg_data_filename(
            'data/wgs84_gauss_kruger_epsg31467.npz'
            ))

        transform = gsp.transform_factory(
            31467, gsp.geospatial.EPSG.WGS84
            )
        glon, glat = transform(
            dat['ulon'] * apu.m, dat['ulat'] * apu.m
            )

        assert_quantity_allclose(glon, dat['glon'] * apu.deg)
        assert_quantity_allclose(glat, dat['glat'] * apu.deg)
