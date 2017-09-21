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

