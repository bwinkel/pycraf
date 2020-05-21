#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

import pytest
from functools import partial
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from astropy.tests.helper import assert_quantity_allclose
from astropy import units as u
from astropy.units import Quantity
from ... import conversions as cnv
from ... import pathprof
from ...utils import check_astro_quantities
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.misc import NumpyRNGContext
import importlib


# skip over rasterio related tests, if not package present:
skip_rio = pytest.mark.skipif(
    importlib.util.find_spec('rasterio') is None,
    reason='"rasterio" package not installed'
    )


TOL_KWARGS = {'atol': 1.e-4, 'rtol': 1.e-6}


def test_landcover_to_p452_clutter_zones():

    landcover_map_corine = np.array([
        [111, 112, 121, 122, 123, 124, 131, 132, 133, 141, 142],
        [211, 212, 213, 221, 222, 223, 231, 241, 242, 243, 244],
        [311, 312, 313, 321, 322, 323, 324, 331, 332, 333, 334],
        [335, 411, 412, 421, 422, 423, 511, 512, 521, 522, 523],
        ],
        dtype=np.int16,
        )

    clutter_map_corine = pathprof.landcover_to_p452_clutter_zones(
        landcover_map_corine, pathprof.CORINE_TO_P452_CLASSES
        )

    np.testing.assert_equal(
        clutter_map_corine, np.array([
            [7, 5, 10, 10, 10, 10, 10, 10, 10, 7, 10],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            ],
            dtype=np.int8,
            ))

    landcover_map_igbp = np.arange(17, dtype=np.int8)

    # IGBP id's start at 1!
    clutter_map_igbp = pathprof.landcover_to_p452_clutter_zones(
        landcover_map_igbp + 1, pathprof.IGBP_TO_P452_CLASSES
        )

    np.testing.assert_equal(
        clutter_map_igbp, np.array([
            3, 2, 3, 2, 2, 0, 0, 2, 0, 0, 0, 0, 7, 0, -1, -1, -1,
            ], dtype=np.int8,
            ))


@skip_rio
def test_wgs84_to_geotiff_pixels():

    import rasterio as rio

    args_list = [
        (-180, 180, u.deg),
        (-90, 90, u.deg),
        ]

    corine_test_file = get_pkg_data_filename(
        'corine/CLC2018_CLC2018_V2018_20_cut.tif'
        )

    with NumpyRNGContext(1):

        lons = np.random.uniform(6.64, 7.13, (5, 5)) * u.deg
        lats = np.random.uniform(50.40, 50.64, (5, 5)) * u.deg

    with rio.open(corine_test_file) as geotiff:

        tfunc = partial(pathprof.wgs84_to_geotiff_pixels, geotiff)
        check_astro_quantities(tfunc, args_list)

        xpix, ypix = tfunc(lons, lats)

        np.testing.assert_allclose(
            xpix, np.array([
                [159.87075, 256.16389, 5.05979, 111.72816, 66.04433],
                [37.85589, 74.33290, 135.84304, 148.71861, 199.79925],
                [154.02347, 250.51821, 85.52072, 310.35913, 23.14141],
                [248.71501, 158.29831, 202.10562, 62.78055, 74.70098],
                [288.03113, 351.01327, 117.01755, 248.63167, 310.93909],
                ],
                dtype=np.float64,
                ),
            **TOL_KWARGS
            )

        np.testing.assert_allclose(
            ypix, np.array([
                [45.67619, 266.06945, 267.44347, 237.27898, 45.94546],
                [253.09123, 168.46844, 27.72161, 141.78957, 101.55987],
                [200.18415, 105.11696, 58.44770, 286.10344, 78.21196],
                [24.24330, 84.74479, 211.60570, 69.54968, 253.44213],
                [170.41227, 49.82715, 204.42966, 211.58685, 256.26989],
                ],
                dtype=np.float64,
                ),
            **TOL_KWARGS
            )


@skip_rio
def test_regrid_from_geotiff():

    import rasterio as rio

    args_list = [
        (-180, 180, u.deg),
        (-90, 90, u.deg),
        ]

    corine_test_file = get_pkg_data_filename(
        'corine/CLC2018_CLC2018_V2018_20_cut.tif'
        )

    with NumpyRNGContext(1):

        lons = np.random.uniform(6.64, 7.13, (5, 5)) * u.deg
        lats = np.random.uniform(50.40, 50.64, (5, 5)) * u.deg

    with rio.open(corine_test_file) as geotiff:

        tfunc = partial(pathprof.regrid_from_geotiff, geotiff)
        check_astro_quantities(tfunc, args_list)

        geodata_regridded = tfunc(lons, lats)

        np.testing.assert_equal(
            geodata_regridded, np.array([
                [211, 311, 311, 231, 231],
                [231, 311, 112, 312, 312],
                [312, 231, 112, 312, 312],
                [211, 312, 312, 211, 231],
                [311, 211, 311, 312, 312],
                ],
                dtype=np.int16,
                ),
            )


@skip_rio
def test_regrid_from_geotiff_degenerated():

    import rasterio as rio

    corine_test_file = get_pkg_data_filename(
        'corine/CLC2018_CLC2018_V2018_20_cut.tif'
        )

    lons = [6.7, 6.7] * u.deg
    lats = [50.5, 50.5] * u.deg

    with rio.open(corine_test_file) as geotiff:

        tfunc = partial(pathprof.regrid_from_geotiff, geotiff)

        geodata_regridded = tfunc(lons, lats)

        np.testing.assert_equal(
            geodata_regridded, np.array([211, 211], dtype=np.int16),
            )
