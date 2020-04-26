.. pycraf-geospatial:

*****************************************
Geographical frames (`pycraf.geospatial`)
*****************************************

.. currentmodule:: pycraf.geospatial

Introduction
============

GIS software often works with coordinate frames other than GPS/WGS84.
Therefore, the `~pycraf.geospatial` sub-package offers routines
to convert between different geographical projections/frames.
Some of the common use cases would be the transformations between
GPS/WGS84 and UTM, ETSR89, or ITRF. However, the `~pycraf.geospatial` sub-package provides a factory to produce arbitrary transforms,
based on EPSG/ESRI codes or `proj4` strings.

.. note::

    Internally, the `pycraf.geospatial` module uses the `pyproj
    <https://pypi.python.org/pypi/pyproj>`_ package, which itself is a wrapper
    around the `proj.4 <http://proj4.org/>`_ software for the transformation.
    In principle, one could use `pyproj` directly, but `pycraf.geospatial`
    offers unit checking (by making use of the `~astropy.units`
    functionality.)

There are thousands of pre-defined frames (which one can work with by simply
using the correct code, e.g., EPSG or ESRI; `see Wikipedia
<https://en.wikipedia.org/wiki/Spatial_reference_system>`__), but the
underlying `proj4` library also accepts user-defined frames. For example,
the ETRS89 frame is defined with the following:

.. code-block:: bash

    +proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000
    +ellps=GRS80 +units=m +no_defs

Using `pycraf.geospatial`
=========================

The provided functions are just a thin wrapper around `pyproj`, mostly
improving convenience for the user::

    >>> import pycraf.geospatial as geo
    >>> import astropy.units as u

    >>> # coordinates of the 100-m telescope at Effelsberg/Germany
    >>> rt_lon, rt_lat = 6.88361 * u.deg, 50.52483 * u.deg

    >>> etrs_lon, etrs_lat = geo.wgs84_to_etrs89(rt_lon, rt_lat)
    >>> print(etrs_lon, etrs_lat)  # doctest: +FLOAT_CMP
    4100074.484338885 m 3050584.470522307 m

If the built-in conversions are not sufficient, you can simply build your own
transform with the `~pycraf.geospatial.transform_factory`. For example,
if one wants Gauss-Kruger coordinates (for the `Germany tile
<http://spatialreference.org/ref/epsg/31467/>`__)::

    >>> transform = geo.transform_factory(
    ...     geo.geospatial.EPSG.WGS84, 31467
    ...     )
    >>> transform(rt_lon, rt_lat)  # doctest: +FLOAT_CMP
    (<Quantity 3350002.4611806367 m>, <Quantity 5600924.810577951 m>)

The factory even produces a correct doc-string::

    >>> print(transform.__doc__)
    <BLANKLINE>
        Convert coordinates from `sys_in` to `sys_out`.
    <BLANKLINE>
        - `sys_in`: EPSG.WGS84
        - `sys_out`: EPSG:31467
    <BLANKLINE>
        Parameters
        ----------
        lon, lat : `~astropy.units.Quantity`
            `sys_in` longitude and latitude [deg]
    <BLANKLINE>
        Returns
        -------
        x, y : `~astropy.units.Quantity`
            `sys_out` world coordinates [m]
    <BLANKLINE>
    <BLANKLINE>
    <BLANKLINE>

The `~pycraf.geospatial.EPSG` and `~pycraf.geospatial.ESRI` enums contain
some pre-defined projections for convenience.


.. note::

    There are many projections that only produce accurate results over a
    relatively small region. In some cases such as `UTM
    <https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system>`__,
    there is a bulk of pre-defined frames, one for each so-called zone.
    Since UTM is quite often used in GIS applications, `~pycraf.geospatial`
    has a helper function `~pycraf.geospatial.utm_zone_from_gps` to
    get the correct zone name for a certain Geographical position.
    Likewise, with `~pycraf.geospatial.epsg_from_utm_zone` one could
    directly query the EPSG code (if needed)::

        >>> utm_zone = geo.utm_zone_from_gps(rt_lon, rt_lat)
        >>> utm_zone
        '32N'

        >>> utm_lon, utm_lat = geo.wgs84_to_utm(rt_lon, rt_lat, utm_zone)
        >>> print(utm_lon, utm_lat)  # doctest: +FLOAT_CMP
        349988.58854241937 m 5599125.388981281 m

Imperial units
--------------

The `proj4` and `pyproj` software packages allow to work with frames that
are historical or of regional interest, only. Some of these don't work
with units of Meters, but feet, etc. Per default, `pyproj` (versions prior
to 2.0) converts all world (=physical) units to Meters (input and/or
output). One can specifically ask for the original units by doing::

    >>> import pycraf.geospatial as geo
    >>> import astropy.units as u
    >>> import pyproj

    >>> proj_wgs84 = pyproj.Proj('+init=epsg:4326')
    >>> # Louisiana South (ftUS)
    >>> proj_nad83 = pyproj.Proj('+init=epsg:3452', preserve_units=False)

    >>> pyproj.transform(proj_wgs84, proj_nad83, -92.105819, 30.447921)  # doctest: +FLOAT_CMP
    (925806.5486332772, 216168.1432314818)

This is the wrong result. But `pyproj.Proj` has an option::

    >>> proj_nad83 = pyproj.Proj('+init=epsg:3452', preserve_units=True)

that gives the correct result::

    >>> pyproj.transform(proj_wgs84, proj_nad83, -92.105819, 30.447921)  # doctest: +FLOAT_CMP
    (3037416.9849743457, 709211.6499186204)

The `~pycraf.geospatial` sub-package makes life a bit easier,
because we can use the `~astropy.units` conversion::

    >>> transform = geo.transform_factory(4326, 3452)
    >>> x, y = transform(-92.105819 * u.deg, 30.447921 * u.deg)
    >>> x, y  # doctest: +FLOAT_CMP
    (<Quantity 925806.5486332772 m>, <Quantity 216168.1432314818 m>)

    >>> x.to(u.imperial.ft), y.to(u.imperial.ft)  # doctest: +FLOAT_CMP
    (<Quantity 3037416.9849743457 ft>, <Quantity 709211.6499186204 ft>)

    >>> transform = geo.transform_factory(3452, 4326)
    >>> transform(3037416.985 * u.imperial.ft, 709211.650 * u.imperial.ft)  # doctest: +FLOAT_CMP
    (<Quantity -92.10581908573734 deg>, <Quantity 30.447921938477027 deg>)

Unfortunately, there seems to be no way to ask `pyproj` about which
original unit is tied to an EPSG code (although it internally must
know it, otherwise, it wouldn't work correctly).

Geocentric systems
------------------
Some frames are Geocentric systems that work with `(x, y, z)` coordinates.
One important example is ITRF::

    >>> geo.wgs84_to_itrf2005(rt_lon, rt_lat, 300 * u.m)  # doctest: +FLOAT_CMP
    (<Quantity 4033874.035684814 m>,
     <Quantity 486981.61007349996 m>,
     <Quantity 4900340.557846253 m>)


See Also
========

- `Astropy Units and Quantities package <http://docs.astropy.org/en/stable/
  units/index.html>`_, which is used extensively in pycraf.
- `proj.4 <http://proj4.org/>`_
- `pyproj <https://pypi.python.org/pypi/pyproj>`_ package
- `UTM zones <http://www.dmap.co.uk/utmworld.htm>`_
- `EPSG and ESRI codes database 1 <http://spatialreference.org>`__
- `EPSG and ESRI codes database 2 <https://epsg.io/>`__

Reference/API
=============

.. automodapi:: pycraf.geospatial
    :no-inheritance-diagram:
