1.1.0 (2021-07-07)
=======================

New Features
------------
pycraf.pathprof
^^^^^^^^^^^^^^^
- Add a `cache` option to `pathprof.height_map_data`, which is based on the
  `joblib` Python package. With this, one can easily re-use existing height
  data (`hprof_cache`) from a previous run to save computing time. [#b86c7d]
- Add a `generic_heights` option to the several functions and the `PathProp`
  class in the `pathprof` sub-package. With this, one case set terrain heights
  to zero more conveniently. [#a3e799]

Notes
-----
- As of Spring 2021, NASA decided to put all SRTM data products behind a
  log-in page, such that automatic download ceases to work. For the time
  being, the default server (and only server) in pycraf will thus be
  `viewpano`. If you prefer to use NASA tiles (over `viewpano`), please use
  their services, e.g., the Land Processes Distributed Active Archive Center.

Bugfixes
~~~~~~~~~~
- In some cases, but only when working with hgt tiles having a different
  angular resolution than 3", terrain height interpolation was still based
  on the 3" instead of the proper resolution. [#89e0b0]
- There was a bug in the `geometry` sub-package, which caused all rotation
  matrices be transposed (which is the same a applying a negative rotation
  angle). We used the opportunity to also fix the sign of the `etilt` angle
  in the `imt2020_composite_pattern`. [#47e772,#9fb8d6]



1.0.4 (2020-10-21)
=======================

Bugfixes
~~~~~~~~~~
- Fix (stable) HTML manual
- Make satellite sub-package work with `sgp4 2.0+`

1.0.3 (2020-05-21)
=======================

New Features
------------
pycraf.pathprof
^^^^^^^^^^^^^^^
- Add some helper functions to ease working with other GIS data sets. An
  example would be Landcover data (e.g., from Copernicus mission Corine data
  set). There is also a tutorial notebook, demonstrating the use. [#29]

1.0.2 (2020-04-16)
=======================

New Features
------------
pycraf.antenna, pycraf.geometry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Several functions in the `antenna` and `geometry` sub-packages were
  implemented in Cython and parallelized using OpenMP. This will greatly
  speed-up Monte-Carlo simulations (e.g., EPFD, which is planned for the next
  release), where lots of antenna-gain values and boresight angles need to be
  computed. [#22]


Bugfixes
--------
- `antenna.imt2020_composite_pattern` was using float32 precision, only, for
  the real and imaginary part (in the complex valued internal calculation).
  This has been fixed and now uses the `complex128` data type. [#26]
- `antenna.imt` was using wrong units (dimension-less instead of dB) for
  front-to-back ratios. Note, that all examples still have the same result,
  as the input parameters were also chosen wrongly. [#25]
- A more detailed error message is now raised, if `astropy.units` is already
  imported - which is not allowed with `astropy` v4 anymore. Please see Issue
  [#24]


1.0.1 (2020-01-08)
=======================

Other
-----
- Migrate CI/CD to Azure Pipelines. This also allows to provide
  (manylinux-based) binary wheels for Linux on PyPI. [#19]

1.0.0 (2020-01-08)
=======================

New Features
------------

pycraf-gui
^^^^^^^^^^
- A simple graphical user interface was added, which can be used to quickly
  analyze the path propogation losses along a transmitter-receiver sight-line
  and plot the path geometry. It is also possible to compute attenuation
  maps. At the moment, it doesn't come with an amazing amount of features,
  but it is foreseen to add more in the future. [#15]

pycraf.atm
^^^^^^^^^^^^^^
- Complete overhaul of the `pycraf.atm` sub-package [#11]. Now
  - it does ray-tracing through atmosphere correctly,
  - the computational speed for Annex 1 algorithms (such as ray-tracing) is
    greatly improved, as performance critical sections are implemented in
    Cython and physical properties of the atmospheric layers are cached
  - there are new functions, such as `pycraf.atm.find_elevation`, which finds
    optimal ray geometry (to reach a given target)

pycraf.pathprof
^^^^^^^^^^^^^^^
- The `pycraf.pathprof` sub-package was re-worked to improve parallel
  computations in various places. Most of these are "under the hood", but
  one new function is `losses_complete`. It can be used to do bulk
  calculations of path propagation loss for fixed Tx-Rx location. This
  makes computations of statistics much faster (see user manual). [#16]

Bugfixes
--------
- For `pycraf.pathprof` we used the BFSG loss (see Rec. ITU-R P.452-16) as
  proxy for line-of-sight loss. However, to bring it inline with the other
  quantities returned by the `pycraf.pathprof.loss_complete`, we now use
  L_b0p, which includes focussing effects [#13, #16]
- `pycraf.geospatial` produced an error with newer version of `pyproj`.
  Basically, the "+init=" part of the projection definition is now
  deprecated. We removed this from the `pycraf.geospatial` module, but
  as a consequence, `pyproj>=2.0` is now required. [#10]
- Make `pycraf` compatible with `astropy>=4.0`, which doesn't offer some `
  utils.compat` functions any more. [00556d]

0.25.8 (2019-02-23)
=======================

Bugfixes
--------
- `pycraf.protection.ra769_limits` now returns an `astropy.table.QTable`
  instead of a `astropy.table.Table`. This ensures that in all
  circumstances one retrieves proper `astropy.units.Quantity` objects from
  the table. Previously, logarithmic units would not fully support this
  (although this was a just bug in `astropy`, which is now fixed). [#8]

0.25.7 (2018-11-24)
=======================

pycraf.antenna
^^^^^^^^^^^^^^
- Add k-factor to single element IMT2020 pattern in `pycraf.antenna`
  module. [#ef1c]
- Add antenna pattern function for IMT advance (LTE) basestation (sectorized,
  peak-side lobe, see ITU-R Rec. F.1336) to `pycraf.antenna` module. [#a4e1]

pycraf.srtm
^^^^^^^^^^^
- Various smaller improvements and bugfixes to SRTM querying.
  [#dcc5, #1950, #0c55]

Bugfixes
--------
- The `pycraf.geospatial` Gauss-Kruger test function revealed a problem with
  inconsistent results between new proj4 version (5.2.0) and older versions.
  At the moment, it is not clear, what's going on. The test cases have marked
  "xfail" for now. [#24e9]
- A sign was wrong in `pycraf.antenna` IMT2020 composite pattern. [#4164]

0.25.6 (2018-05-09)
=======================

pycraf.conversions
^^^^^^^^^^^^^^^^^^
- Add a function `protection.ra769_calculate_entry` that allows to calculate
  RA.769 thresholds for non-standard values (e.g., to query limits for
  RAS bands that are not included in the RA.769 tables).

Other
-----
- Update to newest version (v3.0) of Astropy helpers.

Bugfixes
--------
- Fixing a serious bug in the B03 tutorial notebook (reflections at wind
  turbines). Previously, we accounted twice for the distance when calculating
  the reflected power. As a result, the total received power at the
  RT is now much larger (though still smaller over direct path).


0.25.5 (2017-12-02)
=======================

pycraf.conversions
^^^^^^^^^^^^^^^^^^
- Add further utility routines to `pycraf.conversions` module, to compute
  antenna temperatures and sensitivities. [#8419]

pycraf.atm
^^^^^^^^^^^^^^^
- Add two new functions `atm.elevation_from_airmass` and `
  atm.airmass_from_elevation`, which use a better formula for small
  elevations (compared to the 1/sin(El) behavior). Furthermore,
  the elevation parameter in `atm.opacity_from_atten` and
  `atm.atten_from_opacity` has been made optional. If given, the
  airmass is corrected for (i.e., one works with zenith opacities). [#61e0]

API Changes
-----------
- The functions `pathprof.atten_path_fast` and `pathprof.atten_map_fast`
  now return dictionaries. This makes it easier to add new return values
  in the future (without API breaking). Three new parameters are returned
  for now: the type of path (Line-of-sight or Trans-horizon) and the
  distance of the horizon w.r.t. Tx/Rx.

Bugfixes
--------
- The solution to last exercise in the conversions tutorial notebook was
  wrong. (Thanks to A. Jessner for spotting this.)
- The `phi = 0`-singularity if using `do_bessel=True` in `antenna.ras_pattern`
  was not properly handled. [#5acf]

Other
-----
- Added some notebooks with exercises and solutions, as well as tutorial 03e.

0.25.4 (2017-09-21)
====================

New Features
------------

pycraf.antenna
^^^^^^^^^^^^^^^
- Add correlation-level parameter to `imt2020_composite_pattern`
  function. [#c57b]

pycraf.geometry
^^^^^^^^^^^^^^^
- Add various convenience functions to create 3D rotation matrices from
  rotation axis or Euler angles (and vice versa). Streamline the geometry
  subpackage to allow proper numpy broadcasting. [#c970]

pycraf.pathprof
^^^^^^^^^^^^^^^
- Add a method, `geoid_area` to calculate surface area on the WGS84
  ellipsoid. Only rectangular limits (absolute coordinates) are supported.
  This can be used to determine the area of SRTM pixels (in km^2). [#678f]
- Add a method `atten_path_fast`, which can be used to calculate attenuations
  for full paths very quickly (compared to the manual approach); also see
  tutorial notebooks. To produce the necessary aux-data dictionary,
  two functions are available: `height_path_data` and
  `height_path_data_generic`. To avoid confusion and to streamline
  everything, the function `heigth_profile_data` was renamed to
  `height_map_data`. [#9d6a]

pycraf.protection
^^^^^^^^^^^^^^^
- Add the possibility to generate VLBI threshold values in `ra769_limits`
  function, as contained in Table 3 of RA.769. Furthermore, it is now
  possible to specify the integration time to be used for the
  thresholds. [#8a15]

Documentation
-------------

- Add a notebook about how to tilt or rotate IMT2020 antenna patterns.
- Various updates.

Bugfixes
--------
- In `atm.atten_slant_annex1` the `obs_alt` parameter was not properly
  accounted for. This led to significant errors for high-altitude
  observers. [#f616]
- The function `imt2020_composite_pattern` in the antenna subpackage
  now allows better broadcasting of input arrays. Speed was also
  improved. [#b8ac, #1219]

0.25.3 (2017-08-09)
====================

New Features
------------

pycraf.geospatial
^^^^^^^^^^^^^^^^^
- This sub-package was heavily re-factored. One can now work with EPSG
  or ESRI codes (see docs) and there is a factory to produce arbitrary
  transforms, which come with correct docstrings and quantity/range
  checker (i.e, proper unit handling). Also, we finally made ITRF
  work. [#6892]

Bugfixes
--------
- Increase minimal numpy version to 1.11 to avoid build conflicts with
  MacOS and Windows wheels. The wheels are now built with this minimal
  numpy version (1.11) rather than the latest version. Users of wheels
  have to have at least the same numpy version as the one with which
  pycraf wheels were built. [#2139]


0.25.2 (2017-07-30)
====================

New Features
------------

pycraf.pathprof
^^^^^^^^^^^^^^^
- Add option `interp` to `SrtmConf` to allow different interpolation schemes
  for SRTM data. Currently, 'nearest', 'linear' (default), and 'spline' are
  supported. [#8b43]
- SRTM query functions now support numpy broadcasting. [#af45]

Bugfixes
--------

- SRTM-related plots in documentation were not rendered. [#ea01]
- Clutter loss for "CLUTTER.UNKNOWN" was buggy (must always be zero). [#a8b5]

0.25.1 (2017-07-28)
====================

Bugfixes
--------

- Tests now don't run any SRTM-data related function, only if the option
  `remote-data=any` is given (on CLI, within Python it is invoked with
  `pycraf.test(remote_data='any')`). Therefore, one doesn't need to
  manually download SRTM tiles beforehand anymore. [#eeab]


0.25.0 (2017-07-27)
====================


New Features
------------

General
^^^^^^^
- Minimize dependencies: for a some of `pycraf` sub-packages, do imports
  only on demand and not during pycraf importing. Examples are the
  `satellite` and `geospatial` sub-packages, that depend on `pyproj` and
  `sgp4`. Of course, to use these sub-packages, one has to install the
  necessary dependencies. [#8051]

- Tests that need to download data from the internet are now decorated with
  astropy's `@remote_data` decorator. If you want to run these tests, use::

      import pycraf
      pycraf.test(remote_data='any')

  or

      .. code-block:: bash
      python setup.py test --remote-data='any'

  otherwise, these will be skipped over. [#1444]

- MacOS tests finally work. [#239d]

pycraf.pathprof
^^^^^^^^^^^^^^^^^

- Much better handling of SRTM data was implemented. It is now possible to
  define the SRTM directory during run-time. Furthermore, one can have
  pycraf download missing tiles. For this a new `SrtmConf` manager was
  introduced. [#2d30, #01ba, #208c, #2b5d]

