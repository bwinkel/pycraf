#!/usr/bin/python
# -*- coding: utf-8 -*-

from functools import lru_cache
import numpy as np
from astropy.coordinates import EarthLocation
from astropy import time, units as apu

EARTH_EQUATORIAL_RADIUS = 6378.135  # equatorial radius
EARTH_FLATTENING_CONSTANT = 1 / 298.26
GEO_SYNC_RADIUS = 42164.57


__all__ = ['get_sat', 'SatelliteObserver']


@lru_cache(maxsize=128)
def get_sat(tle_string):
    '''
    Construct a satellite instance (`sgp4.io.Satellite`) from TLE string.

    Parameters
    ----------
    tle_string : str
        Two-line elements (TLE) as 3-line string

    Returns
    -------
    satname : str
        Name (identifier) of satellite
    satellite : `sgp4.io.Satellite` instance
        Satellite object filled from TLE

    Notes
    -----
    TLE string must be of the form:

    .. code-block:: none

        ISS (ZARYA)
        1 25544U 98067A   13165.59097222  .00004759  00000-0  88814-4 0    47
        2 25544  51.6478 121.2152 0011003  68.5125 263.9959 15.50783143834295
    '''

    try:
        from sgp4.api import Satrec, WGS72
    except ImportError:
        raise ImportError(
            'The "sgp4" package (version 2+) is necessary to use this '
            'function.'
            )

    tle_string_list = tle_string.split('\n')
    satname = tle_string_list[0]
    if satname[0:2] == '0 ':  # remove leading 0 if present
        satname = satname[2:]
    satellite = Satrec.twoline2rv(*tle_string_list[1:3], WGS72)

    return satname, satellite


def _propagate(sat, dt):
    '''
    True equator mean equinox (TEME) position from `sgp4` at given time.

    Parameters
    ----------
    sat : `sgp4.io.Satellite` instance
        Satellite object filled from TLE
    dt : `~datetime.datetime`
        Time

    Returns
    -------
    xs, ys, zs : float
        TEME (=True equator mean equinox) position of satellite [km]
    '''

    # pos [km], vel [km/s]

    try:
        from sgp4.api import jday, SGP4_ERRORS
    except ImportError:
        raise ImportError(
            'The "sgp4" package (version 2+) is necessary to use this '
            'function.'
            )

    jd, fr = jday(
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute, dt.second + dt.microsecond / 1e6
        )
    err_code, position, velocity = sat.sgp4(jd, fr)

    if err_code:
        raise ValueError(
            'Satellite propagation error', err_code,
            '({})'.format(SGP4_ERRORS[err_code])
            )

    return position


_vec_propagate = np.vectorize(
    _propagate, excluded=['sat'], otypes=[np.float64] * 3
    )


class SatelliteObserver(object):
    '''
    Calculate position of a satellite relative to an observer on Earth.

    Parameters
    ----------
    obs_location : `~astropy.coordinates.EarthLocation` or tuple
        Location of observer.

        This must be in a form that can be casted to
        `~astropy.coordinates.EarthLocation`.

    Returns
    -------
    sat_obs : SatelliteObserver instance
    '''

    def __init__(
            self, obs_location
            ):

        self._obs_location = EarthLocation(obs_location)

    def _eci_coords_observer(self, lmst_rad):
        '''
        Infer ECI coordinates of observer for a certain observation time.

        Parameters
        ----------
        lmst_rad : `~numpy.ndarray` or float
            Local mean sidereal time, LMST [rad]

        Returns
        -------
        x, y, z : `~numpy.ndarray` or float
            ECI (=Geocentric cartesian inertial) coordinates of observer [km]

        Notes
        -----
        The function accounts for Earth flattening.
        '''

        obs_lat_rad = self._obs_location.lat.rad
        obs_alt_km = self._obs_location.height.to(apu.km).value

        C = 1. / np.sqrt(
            1 + EARTH_FLATTENING_CONSTANT * (EARTH_FLATTENING_CONSTANT - 2) *
            np.sin(obs_lat_rad) ** 2
            )
        S = (1. - EARTH_FLATTENING_CONSTANT) ** 2 * C

        # make ECI coordinates corrected for earth flattening
        earth_rad = EARTH_EQUATORIAL_RADIUS + obs_alt_km
        x = earth_rad * C * np.cos(obs_lat_rad) * np.cos(lmst_rad)
        y = earth_rad * C * np.cos(obs_lat_rad) * np.sin(lmst_rad)
        z = earth_rad * S * np.sin(obs_lat_rad)

        return x, y, z

    def _eci_coords_satellite(self, satellite, obstime):
        '''
        Parameters
        ----------
        satellite : `sgp4.io.Satellite` instance
            Satellite object filled from TLE
        obstime : `~astropy.time.Time`
            Time of observation

        Returns
        -------
        xs, ys, zs : `~numpy.ndarray` or float
            ECI (=Geocentric cartesian inertial) coordinates of satellite [km]
        '''

        # dt = obstime.datetime

        # position, velocity = satellite.propagate(
        #     dt.year, dt.month, dt.day,
        #     dt.hour, dt.minute, dt.second + dt.microsecond / 1e6
        #     )

        # if position is None:
        #     raise ValueError('Satellite propagation error')

        # return position

        return _vec_propagate(satellite, obstime.datetime)

    def _lookangle(
            self,
            xs, ys, zs,
            xo, yo, zo,
            lmst_rad,
            ):
        '''
        Horizontal position of and distance to satellite w.r.t. observer.

        Parameters
        ----------
        xs, ys, zs : `~numpy.ndarray` or float
            ECI (=Geocentric cartesian inertial) coordinates of satellite [km]
        xo, yo, zo : `~numpy.ndarray` or float
            ECI (=Geocentric cartesian inertial) coordinates of observer [km]
        lmst_rad : `~numpy.ndarray` or float
            Local mean sidereal time, LMST [rad]

        Returns
        -------
        az, el : `~numpy.ndarray` or float
            Azimuth/elevation of satellite w.r.t. observer [deg]
        dist : `~numpy.ndarray` or float
            Distance to satellite [km]
        '''

        rx = xs - xo
        ry = ys - yo
        rz = zs - zo

        obs_lat_rad = self._obs_location.lat.rad
        s_obs_lat_rad = np.sin(obs_lat_rad)
        c_obs_lat_rad = np.cos(obs_lat_rad)
        s_lmst_rad = np.sin(lmst_rad)
        c_lmst_rad = np.cos(lmst_rad)

        r_s = (
            s_obs_lat_rad * c_lmst_rad * rx +
            s_obs_lat_rad * s_lmst_rad * ry -
            c_obs_lat_rad * rz
            )
        r_e = -s_lmst_rad * rx + c_lmst_rad * ry
        r_z = (
            c_obs_lat_rad * c_lmst_rad * rx +
            c_obs_lat_rad * s_lmst_rad * ry +
            s_obs_lat_rad * rz
            )

        dist = np.sqrt(r_s ** 2 + r_e ** 2 + r_z ** 2)
        az = np.degrees(np.arctan2(-r_e, r_s) + np.pi)
        az = (az + 180) % 360 - 180
        el = np.degrees(np.arcsin(r_z / dist))

        return az, el, dist

    def azel_from_sat(self, satellite_or_tle, obstime):
        '''
        Parameters
        ----------
        satellite_or_tle : `sgp4.io.Satellite` instance or TLE 3-line string
            Satellite object or TLE
        obstime : `~astropy.time.Time`
            Time of observation

        Returns
        -------
        az, el : `~numpy.ndarray` or float
            Azimuth/elevation of satellite w.r.t. observer [deg]
        dist : `~numpy.ndarray` or float
            Distance to satellite [km]

        Notes
        -----
        If satellite is provided as TLE string, it must be of the form:

        .. code-block:: none

            ISS (ZARYA)
            1 25544U 98067A   13165.59097222  .00004759  00000-0  88814-4 0    47
            2 25544  51.6478 121.2152 0011003  68.5125 263.9959 15.50783143834295
        '''

        assert isinstance(obstime, time.Time), (
            'obstime must be an astropy.time.Time object!'
            )

        try:
            from sgp4.api import Satrec
        except ImportError:
            raise ImportError(
                'The "sgp4" package is necessary to use this Class.'
                )

        if not isinstance(satellite_or_tle, Satrec):
            _, satellite = get_sat(satellite_or_tle)
        else:
            satellite = satellite_or_tle

        obs_lon_rad = self._obs_location.lon.rad
        lmst_rad = obstime.sidereal_time('mean', 'greenwich').rad + obs_lon_rad

        xo, yo, zo = self._eci_coords_observer(lmst_rad)
        xs, ys, zs = self._eci_coords_satellite(satellite, obstime)

        az, el, dist = self._lookangle(
            xs, ys, zs, xo, yo, zo,
            lmst_rad,
            )

        return az * apu.deg, el * apu.deg, dist * apu.km
