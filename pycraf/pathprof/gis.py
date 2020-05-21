#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

from astropy import units as u
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from astropy.utils.data import get_pkg_data_filename
from . import cyprop
from .. import geospatial
from .. import utils


__all__ = [
    'CORINE_TO_P452_CLASSES', 'IGBP_TO_P452_CLASSES', 'P452_CLUTTER_COLORS',
    'landcover_to_p452_clutter_zones',
    'wgs84_to_geotiff_pixels',
    'regrid_from_geotiff',
    ]


_CORINE_TO_P452_CLASSES_FILE = get_pkg_data_filename(
    'data/corine_to_p452_classes.txt'
    )
CORINE_TO_P452_CLASSES = dict(np.genfromtxt(
    _CORINE_TO_P452_CLASSES_FILE,
    dtype=np.dtype([('CorineID', np.uint16), ('ClutterName', np.str, 100)]),
    delimiter=',',
    ))

_IGBP_TO_P452_CLASSES_FILE = get_pkg_data_filename(
    'data/igbp_to_p452_classes.txt'
    )
IGBP_TO_P452_CLASSES = dict(np.genfromtxt(
    _IGBP_TO_P452_CLASSES_FILE,
    dtype=np.dtype([('IGBP_ID', np.uint8), ('ClutterName', np.str, 100)]),
    delimiter=',',
    ))

P452_CLUTTER_COLORS = {
    'UNKNOWN': (255, 255, 255, 0),
    'SPARSE': (255, 230, 200, 255),
    'VILLAGE': (255, 160, 50, 255),
    'DECIDIOUS_TREES': (128, 255, 0, 255),
    'CONIFEROUS_TREES': (0, 166, 0, 255),
    'TROPICAL_FOREST': (166, 230, 77, 255),
    'SUBURBAN': (255, 100, 0, 255),
    'DENSE_SUBURBAN': (255, 100, 50, 255),
    'URBAN': (255, 0, 0, 255),
    'DENSE_URBAN': (255, 0, 50, 255),
    'HIGH_URBAN': (255, 0, 100, 255),
    'INDUSTRIAL_ZONE': (255, 255, 0, 255),
    }


def landcover_to_p452_clutter_zones(landcover_map, conversion_table):
    '''
    Convert a map of landcover class IDs to P.452 clutter types.

    Two pre-defined mappings (conversion tables) are provided in
    `~pycraf.pathprof`, one for the Corine landcover survey (by the Copernicus
    mission) that is a high-resolution database for Europe, and one for IGBP
    class IDs, which are used e.g., by  the Terra and Aqua combined Moderate
    Resolution Imaging Spectroradiometer, MODIS, which is available for
    several years. One version of that data, the "MOD12C1"
    (Majority_Land_Cover_Type_1) map is a (rather low-resolution)
    version for the full Earth.

    Be careful, some data files seem to contain zero-indexed IGBP IDs! One
    may need to increase the input map values by One in such cases.

    Parameters
    ----------
    landcover_map : 2D `~numpy.ndarray` (usually integer type)
        Map of land cover types (e.g. from Corine land cover survey). Data
        type will usually be some Integer type and must contain land cover
        class IDs (see e.g. Notes below).
    conversion_table : `dict`
        A dictionary with a mapping (conversion) from the land cover ID to
        the P.452 clutter zone type. Two mappings are pre-defined in
        `~pycraf.pathprof`, the `~pycraf.pathprof.CORINE_TO_P452_CLASSES` and
        the `~pycraf.pathprof.IGBP_TO_P452_CLASSES`.

    Returns
    -------
    p452_clutter_zones_map : 2D `~numpy.ndarray` of int8
        Map of P.452 clutter zone types derived from the provided dictionary
        with the landcover class IDs.

    Notes
    -----
    - Corine land cover classes:

      | 111 Continuous urban fabric
      | 112 Discontinuous urban fabric
      | 121 Industrial or commercial units
      | 122 Road and rail networks and associated land
      | 123 Port areas
      | 124 Airports
      | 131 Mineral extraction sites
      | 132 Dump sites
      | 133 Construction sites
      | 141 Green urban areas
      | 142 Sport and leisure facilities
      | 211 Non-irrigated arable land
      | 212 Permanently irrigated land
      | 213 Rice fields
      | 221 Vineyards
      | 222 Fruit trees and berry plantations
      | 223 Olive groves
      | 231 Pastures
      | 241 Annual crops associated with permanent crops
      | 242 Complex cultivation patterns
      | 243 Agricultural land with significant areas of natural vegetation
      | 244 Agro-forestry areas
      | 311 Broad-leaved forest
      | 312 Coniferous forest
      | 313 Mixed forest
      | 321 Natural grasslands
      | 322 Moors and heathland
      | 323 Sclerophyllous vegetation
      | 324 Transitional woodland-shrub
      | 331 Beaches dunes sands
      | 332 Bare rocks
      | 333 Sparsely vegetated areas
      | 334 Burnt areas
      | 335 Glaciers and perpetual snow
      | 411 Inland marshes
      | 412 Peat bogs
      | 421 Salt marshes
      | 422 Salines
      | 423 Intertidal flats
      | 511 Water courses
      | 512 Water bodies
      | 521 Coastal lagoons
      | 522 Estuaries
      | 523 Sea and ocean
      | 999 NODATA

    - IGBP land cover classes:

      | 1  Evergreen needleleaf forests
      | 2  Evergreen broadleaf forests
      | 3  Deciduous needleleaf forests
      | 4  Deciduous broadleaf forests
      | 5  Mixed forests
      | 6  Closed shrublands
      | 7  Open shrublands
      | 8  Woody savannas
      | 9  Savannas
      | 10 Grasslands
      | 11 Permanent wetlands
      | 12 Croplands
      | 13 Urban and built-up lands
      | 14 Cropland/natural vegetation mosaics
      | 15 Snow and ice
      | 16 Barren
      | 17 Water bodies
    '''

    clutter_map = np.full(landcover_map.shape, -1, dtype=np.int8)
    for cor_id, cl_name in conversion_table.items():
        mask = landcover_map == cor_id
        clutter_map[mask] = getattr(cyprop.CLUTTER, cl_name)

    return clutter_map


@utils.ranged_quantity_input(
    lons=(-180, 180, u.deg),
    lats=(-90, 90, u.deg),
    strip_input_units=True,
    output_unit=None,
    )
def wgs84_to_geotiff_pixels(geotiff, lons, lats):
    '''
    Convert WGS84 (longitude, latitude) to pixel space of a GeoTiff.

    This is purely a convenience function, which internally calls `pyproj`
    with the appropriate CRS of the GeoTiff file. The GeoTiff file must be
    opened with the Python package `Rasterio
    <https://rasterio.readthedocs.io/>`_

    Parameters
    ----------
    geotiff : `~rasterio.io.DatasetReader` instance
        A geotiff raster map opened with the Python package `Rasterio
        <https://rasterio.readthedocs.io/>`_.
    lons, lats : `~astropy.units.Quantity`
        Geographic longitudes/latitudes (WGS84) [deg]

    Returns
    -------
    xpix, ypix : `~numpy.ndarray` of float64
        Pixel coordinates (floating point!) of the provided geographic
        position(s) in the given GeoTiff raster map.
    '''

    try:
        import rasterio as rio
    except ImportError as e:
        print('Python package rasterio is needed for this function.')
        raise e

    if not isinstance(geotiff, rio.io.DatasetReader):
        raise TypeError(
            '"geotiff" parameter must be an instance of '
            '"rasterio.io.DatasetReader" (a Rasterio geotiff file object)'
            )

    lons, lats = np.broadcast_arrays(
        lons, lats
        ) * u.deg
    wgs84_to_crs_world = geospatial.transform_factory(
        geospatial.EPSG.WGS84, geotiff.crs.to_proj4()
        )
    wx, wy = wgs84_to_crs_world(lons, lats)
    px, py = (~geotiff.transform) * np.array([
        wx.value.flatten(), wy.value.flatten()
        ])

    return px.reshape(lons.shape), py.reshape(lats.shape)


@utils.ranged_quantity_input(
    lons=(-180, 180, u.deg),
    lats=(-90, 90, u.deg),
    strip_input_units=False,
    output_unit=None,
    )
def regrid_from_geotiff(geotiff, lons, lats, band=1):
    '''
    Retrieve interpolated GeoTiff raster values for given WGS84 coordinates
    (longitude, latitude).

    Most GeoTiff raster maps will be based on reference frames other than
    geographic (WGS84), such that it is often necessary to reproject a map,
    e.g. to get the data values for the positions present in an terrain height
    map such as SRTM, which is the basis for path propagation loss
    calculations. By means of nearest neighbour interpolation one can quickly
    reproject (or regrid) a GeoTiff raster map to the required positions.

    Parameters
    ----------
    geotiff : `~rasterio.io.DatasetReader` instance
        A geotiff raster map opened with the Python package `Rasterio
        <https://rasterio.readthedocs.io/>`_.
    lons, lats : `~astropy.units.Quantity`
        Geographic longitudes/latitudes (WGS84) [deg]
    band : int, Optional (default: 1)
        The GeoTiff band to use.

    Returns
    -------
    geo_data_regridded : `~numpy.ndarray`
        Regridded values of the input raster map on the given longitude and
        latitude positions. If the input GeoTiff has more than one band and
        you need to regrid several of the bands, please run the function
        repeatedly, specifying the band parameter.

    Notes
    -----
    If requested geo positions are all to close to the edge of the geotiff
    file, the interpolation can fail (`~scipy.interpolate` will raise a
    ValueError in such cases).
    '''

    try:
        import rasterio as rio
    except ImportError as e:
        print('Python package rasterio is needed for this function.')
        raise e

    if not isinstance(geotiff, rio.io.DatasetReader):
        raise TypeError(
            '"geotiff" parameter must be an instance of '
            '"rasterio.io.DatasetReader" (a Rasterio geotiff file object)'
            )

    geo_x, geo_y = wgs84_to_geotiff_pixels(geotiff, lons, lats)

    xmin, xmax = np.int32([geo_x.min(), geo_x.max()])
    ymin, ymax = np.int32([geo_y.min(), geo_y.max()])
    # slightly enlarge box to avoid edge effects
    col_off, row_off = xmin - 5, ymin - 5
    col_width, row_width = xmax - xmin + 10, ymax - ymin + 10
    window = rio.windows.Window(col_off, row_off, col_width, row_width)

    geo_data = geotiff.read(band, window=window)

    # are the world coordinates associated with pixel centers?
    # is there a shift about half a pixel necessary?
    # --> according to https://gdal.org/user/raster_data_model.html#affine-geotransform
    # the (0, 0) is the top left corner of the top left pixel
    geo_interp = RegularGridInterpolator(
        (np.arange(col_width), np.arange(row_width)),
        geo_data.T,
        method='nearest', bounds_error=True,
        )

    geo_data_regridded = geo_interp((geo_x - col_off, geo_y - row_off))

    return geo_data_regridded.astype(geo_data.dtype, copy=False)
