.. pycraf-satellite:

**************************************
Satellites (`pycraf.satellite`)
**************************************

.. currentmodule:: pycraf.satellite

Introduction
============

Often the compatibility of space-based radio communications with other
services has to be studied. Then, it may happen, that one has to determine the
position of one or more satellites with respect to an observer on Earth's
surface. The `~pycraf.satellite` sub-package uses the `sgp4 package
<https://pypi.python.org/pypi/sgp4/>`_ to calculate Geocentric cartesian
inertial (ECI) position for a given time and adds the transformations to the (
local) horizontal coordinates (azimuth, elevation, distance) of an observer.


Using `pycraf.satellite`
=========================

The first step to calculate a satellite's position is to obtain a so-called
`Two-line element set (TLE) <https://en.wikipedia.org/wiki/Two-line_element_set>`_.

.. note::

    The TLEs are usually published once a day, because the contained
    parameters quickly change; drag forces cause rapid changes in the orbits
    of almost all satellites.

On most websites that offer TLEs (e.g., `Celestrak <http://celestrak.com/>`_)
in fact three-line strings are used, where the first line just contains the name of the satellite:

.. code-block:: none

    ISS (ZARYA)
    1 25544U 98067A   13165.59097222  .00004759  00000-0  88814-4 0    47
    2 25544  51.6478 121.2152 0011003  68.5125 263.9959 15.50783143834295

The pycraf package follows this format. To calculate the horizontal
coordinates of the ISS at a given time, download the current TLE and do::

    >>> import datetime
    >>> import numpy as np
    >>> from astropy.coordinates import EarthLocation
    >>> from astropy import time
    >>> from pycraf import satellite

    >>> tle_string = '''ISS (ZARYA)
    ... 1 25544U 98067A   13165.59097222  .00004759  00000-0  88814-4 0    47
    ... 2 25544  51.6478 121.2152 0011003  68.5125 263.9959 15.50783143834295'''

    >>> # define observer location
    >>> location = EarthLocation(6.88375, 50.525, 366.)

    >>> # create a SatelliteObserver instance
    >>> sat_obs = satellite.SatelliteObserver(location)

    >>> dt = datetime.datetime(2010, 7, 22, 8, 38, 57)
    >>> obstime = time.Time(dt)

    >>> az, el, dist = sat_obs.azel_from_sat(tle_string, obstime)  # doctest: +REMOTE_DATA
    >>> print('az, el, dist: {:.1f}, {:.1f}, {:.0f}'.format(az, el, dist))  # doctest: +REMOTE_DATA
    az, el, dist: -165.5 deg, 7.4 deg, 1713 km

    >>> # can also use arrays of obstime
    >>> mjd = 55123. + np.array([0.123, 0.99, 50.3])
    >>> obstime2 = time.Time(mjd, format='mjd')
    >>> az, el, dist = sat_obs.azel_from_sat(tle_string, obstime2)  # doctest: +REMOTE_DATA
    >>> np.set_printoptions(precision=3)
    >>> az  # doctest: +SKIP
    <Quantity [  28.397, -143.170,  -6.974] deg>
    >>> el  # doctest: +SKIP
    <Quantity [-43.169, -21.468, -36.773] deg>
    >>> dist  # doctest: +SKIP
    <Quantity [ 9388.292, 5675.039, 8361.284] km>

The `~pycraf.satellite.SatelliteObserver.azel_from_sat` method also accepts
a `sgp4.io.Satellite` object. This could be created by using the
`sgp4 package <https://pypi.python.org/pypi/sgp4/>`_ directly. pycraf also offers a convenience routine, `~pycraf.satellite.get_sat`, which does the TLE
parsing for your and returns the satellite name and `Satellite` object. It also
uses caching to save some computing time, if many different satellites are to
be processed::

    >>> from pycraf import satellite

    >>> tle_string = '''ISS (ZARYA)
    ... 1 25544U 98067A   13165.59097222  .00004759  00000-0  88814-4 0    47
    ... 2 25544  51.6478 121.2152 0011003  68.5125 263.9959 15.50783143834295'''

    >>> satname, sat = satellite.get_sat(tle_string)
    >>> satname
    'ISS (ZARYA)'
    >>> sat.satnum
    25544

    # using sgp4 directly, to get position and velocity in ECI coordinates
    >>> from sgp4.api import jday

    >>> jd, fr = jday(2017, 6, 29, 12, 50, 19)
    >>> err_code, position, velocity = sat.sgp4(jd, fr)
    >>> position  # km  # doctest: +FLOAT_CMP
    (3289.302521216188, 3816.880925531413, 4443.175627001508)
    >>> velocity  # km/s  # doctest: +FLOAT_CMP
    (-2.958995807371371, 6.335950621185331, -3.241778555003016)


See Also
========

- `Astropy Units and Quantities package <http://docs.astropy.org/en/stable/
  units/index.html>`_, which is used extensively in pycraf.
- `sgp4 package <https://pypi.python.org/pypi/sgp4/>`_
- `Two-line element set (TLE)
  <https://en.wikipedia.org/wiki/Two-line_element_set>`_
- `Celestrak <http://celestrak.com/>`_

Reference/API
=============

.. automodapi:: pycraf.satellite
    :no-inheritance-diagram:
