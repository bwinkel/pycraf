0.25.3 (unreleased)
====================

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

