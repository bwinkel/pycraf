.. pycraf-geospatial:

*****************************************
Geographical frames (`pycraf.geospatial`)
*****************************************

.. currentmodule:: pycraf.geospatial

Introduction
============

GIS software often works with coordinate frames other than GPS/WGS84.
Therefore, the `~pycraf.geospatial` sub-package offers  routines
to convert between GPS/WGS84 <-> UTM and GPS/WGS84 <-> ETSR89.

.. note::

    Internally, the `pycraf.geospatial` module uses the `pyproj
    <https://pypi.python.org/pypi/pyproj>`_ package, which itself is a wrapper
    around the `proj.4 <http://proj4.org/>`_ software for the transformation.

Using `pycraf.geospatial`
=========================

The provided functions are just a thin wrapper around `pyproj`::

    >>> import astropy.units as u
    >>> from pycraf.geospatial import *

    >>> # coordinates of the 100-m telescope at Effelsberg/Germany
    >>> rt_lon, rt_lat = 6.88361 * u.deg, 50.52483 * u.deg

    >>> utm_lon, utm_lat = wgs84_to_utm(rt_lon, rt_lat, 32)
    >>> print(utm_lon, utm_lat)  # doctest: +FLOAT_CMP
    349988.58854241937 m 5599125.388981281 m

    >>> etrs_lon, etrs_lat = wgs84_to_etrs89(rt_lon, rt_lat)
    >>> print(etrs_lon, etrs_lat)  # doctest: +FLOAT_CMP
    4100074.484338885 m 3050584.470522307 m



See Also
========

- `Astropy Units and Quantities package <http://docs.astropy.org/en/stable/
  units/index.html>`_, which is used extensively in pycraf.
- `proj.4 <http://proj4.org/>`_
- `pyproj <https://pypi.python.org/pypi/pyproj>`_ package
- `UTM zones <http://www.dmap.co.uk/utmworld.htm>`_

Reference/API
=============

.. automodapi:: pycraf.geospatial
