.. _working_with_srtm:

**********************
Working with SRTM data
**********************

Introduction
============

To make full use of the path attenuation calculations provided by `pycraf`
we recommend to use NASA's
`Shuttle Radar Topography Mission (SRTM) <https://www2.jpl.nasa.gov/srtm/>`_
data for height-profile generation. `pycraf` can work with so-called *.hgt*
files, a very simple binary format. Each *.hgt* file, a so-called tile, just
contains 1201x1201 16-bit integers. From the file naming scheme, one can infer
the associated coordinates. Tiles contain one square-degree of data and
are named in a very simple manner, e.g., *N50E006.hgt*. The first
character is either "N" (North) or "S" (South), indicating whether the
following number is positive or negative latitude. Likewise, the 4th
character, "W" (West) or "E" (East), means negative and positive longitude.

Unfortunately, we cannot provide SRTM data as part of the package, due to the
large file sizes and legal reasons.

Obtaining SRTM data
===================

There are various versions of SRTM data. "Official" are the Versions 1
and 2.1 available on https://dds.cr.usgs.gov/srtm/. There is even a NASA
version 3, but we couldn't find a site that allows direct download. It may
work with an EarthData Account on
https://lpdaac.usgs.gov/data_access/data_pool.

Then, there is a version V4.1 by `CGIAR
<ftp://srtm.csi.cgiar.org/SRTM_V41/SRTM_Data_GeoTiff/>`_
and an unofficial version by `viewfinderpanoramas.org
<http://viewfinderpanoramas.org/>`_.

.. warning::

    Working with the SRTM data from the above sources may not be allowed for
    commercial use unless you get explicit permission to do so.
    For research or private use you may have to properly acknowledge
    the source of the data. Please check the web sites for more information!

Configuring pycraf to use SRTM data
===================================

When you use the functionality from the  `~pycraf.pathprof` sub-package,
`pycraf` will initally assume that *.hgt* files are in the local folder
(or a sub-directory). If you have many different projects and want to
store the tiles in a custom (system-wide) directory, there are two options,
explained in the following.

.. warning::

    Placing the *.hgt* files in the current work directory (where Python
    is started) and having `pycraf` look for them there seems handy, but it
    can be slow if many other sub-directories are present as `pycraf` will
    recursively search for *.hgt* files.

Using `SRTMDATA` environment variable
-------------------------------------

If you have downloaded some (or even all) tiles into a directory, you can let
`pycraf` know by defining an environment variable in the terminal where you
run your program.

On windows::

    set SRTMDATA=C:\[path-to-srtm]\

On Linux/MacOS (sh-like)::

    export SRTMDATA=[path-to-srtm]/

Of course, it is possible to make this permanent. On Linux/MacOS put it into
your shell config file (e.g., *.bashrc*). On Windows 7+, you could click
on the start menu, type "env" and open the dialog that allows to edit your
user environment settings.

Changing the SRTM path during a running session
-----------------------------------------------

It is also possible to change pycraf's behavior during runtime. In the
`~pycraf.pathprof` sub-package, the `~pycraf.pathprof.SrtmConf` class is
defined, which allows to change the SRTM directory, but also makes it possible
to download missing tiles during run-time::


    >>> from pycraf.pathprof import SrtmConf
    >>> SrtmConf.set(srtm_dir='/path/to/srtmdir')  # doctest: +IGNORE_OUTPUT

Alternatively, if only a temporary change of the config is desired,
one can use `~pycraf.pathprof.SrtmConf` as a context manager::

    >>> with SrtmConf.set(srtm_dir='/path/to/srtmdir'):
    ...     # do stuff
    ...     pass

Afterwards, the old settings will be re-established.

It is also possible to allow downloading of missing *.hgt* files::

    >>> SrtmConf.set(download='missing')  # doctest: +IGNORE_OUTPUT

The default download server will be `server='nasa_v2.1'`. One could
also use the (very old) data (`server='nasa_v1.0'`) or inofficial
tiles from `viewfinderpanoramas.org
<http://viewfinderpanoramas.org/>`_ (`server='viewpano'`)::

    >>> SrtmConf.set(server='viewpano')  # doctest: +IGNORE_OUTPUT

.. note::

  Note: As of Spring 2021, NASA decided to put all SRTM data products
  behind a log-in page, such that automatic download ceases to work.
  For the time being, the default server (and only server) will thus be
  `viewpano`.
  If you prefer to use NASA tiles (over viewpano), please use their
  services, e.g., the `Land Processes Distributed Active Archive Center
  <https://lpdaac.usgs.gov/>`.

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
spline degrees (`kx` and `ky`) and smoothing factor (`s`).
To change these use::

    >>> SrtmConf.set(interp='spline', spline_opts=(3, 0))  # doctest: +IGNORE_OUTPUT

We refer to `~scipy.interpolate.RectBivariateSpline` description for
further information.


Copernicus DEM (GLO-90 / GLO-30)
================================

Besides the SRTM *.hgt* tiles, `~pycraf.pathprof` can use the `Copernicus DEM
<https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model>`_
(GLO-90 and GLO-30) as a terrain source. Compared to SRTM this has a number of
advantages:

- It is hosted as `Cloud-Optimised GeoTIFFs
  <https://registry.opendata.aws/copernicus-dem/>`_ on the AWS Open Data
  buckets (no authentication needed) and thus - unlike the retired SRTM
  download servers - provides a reliably working automatic-download path.
- It is global (pole-to-pole), whereas SRTM only covers 60 deg N to 56 deg S.
- It is void-free over water (edited/flattened water bodies), avoiding the
  data-gap artifacts that SRTM tiles can contain.

To use it, select one of the two Copernicus servers::

    >>> from pycraf.pathprof import SrtmConf
    >>> SrtmConf.set(  # doctest: +SKIP
    ...     srtm_dir='/path/to/copernicus',
    ...     download='missing',
    ...     server='copernicus_glo90',  # or 'copernicus_glo30'
    ...     )

The public API (`~pycraf.pathprof.srtm_height_data`,
`~pycraf.pathprof.srtm_height_profile`, and the P.452 `~pycraf.pathprof.PathProp`
engine) is unchanged; internally the reader accounts for the Copernicus tiles'
pixel-centre (area) registration and their latitude-dependent longitude spacing
(the tiles are not square above :math:`|\mathrm{lat}| = 50` deg).

.. note::

  Reading the GeoTIFF tiles requires the optional `rasterio
  <https://rasterio.readthedocs.io/>`_ package. If a Copernicus server is
  selected without it, a clear error is raised.

.. note::

  The Copernicus DEM uses the EGM2008 geoid as its vertical datum (SRTM uses
  EGM96). The difference is at the metre level and is not converted, since
  propagation profiles only depend on *relative* terrain heights.

**Attribution.** When using or redistributing Copernicus DEM data, the
following statement must be reproduced:

  Produced using Copernicus WorldDEM-90 © DLR e.V. 2010-2014 and
  © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by
  the European Union and ESA; all rights reserved.


Handling missing tiles and voids
================================

Two `~pycraf.pathprof.SrtmConf` options control what happens when data is
missing or invalid. By default, if a tile that *should* exist on the chosen
server is not found on disk (and is not downloaded), its terrain is set to
zero and a `~pycraf.pathprof.TileNotAvailableOnDiskWarning` is emitted. To turn
this into a hard error instead (so that computations never silently run over
zero terrain), use::

    >>> SrtmConf.set(on_missing='raise')  # doctest: +IGNORE_OUTPUT

Void pixels (SRTM data gaps; the Copernicus DEM is void-free) are, by default,
replaced with zero. They can instead be kept as ``NaN`` (to be detected
downstream) or filled from the nearest valid pixels::

    >>> SrtmConf.set(void_fill='interp')  # doctest: +IGNORE_OUTPUT

.. note::

  Restore the historic defaults with
  ``SrtmConf.set(on_missing='zeros', void_fill='zero')``.


Download links
==============
- `NASA v2.1 <https://dds.cr.usgs.gov/srtm/version2_1/SRTM3/>`__
- `NASA v1.0 <https://dds.cr.usgs.gov/srtm/version1/>`__
- `viewfinderpanoramas.org <http://www.viewfinderpanoramas.org/Coverage%20map%20viewfinderpanoramas_org3.htm>`__
- `Copernicus GLO-90 (AWS Open Data) <https://copernicus-dem-90m.s3.amazonaws.com/readme.html>`__
- `Copernicus GLO-30 (AWS Open Data) <https://copernicus-dem-30m.s3.amazonaws.com/readme.html>`__


