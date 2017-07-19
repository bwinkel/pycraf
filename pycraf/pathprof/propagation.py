#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

# from functools import partial, lru_cache
from astropy import units as apu
import numpy as np

from . import cyprop
from .. import conversions as cnv
from .. import utils
# import ipdb


__all__ = [
    'PathProp',
    'loss_freespace', 'loss_troposcatter', 'loss_ducting',
    'loss_diffraction', 'loss_complete',
    'clutter_correction', 'clutter_imt',
    'height_profile_data', 'atten_map_fast',
    ]

# Note, we have to curry the quantities here, because Cython produces
# "built-in" functions that don't provide a signature (such that
# ranged_quantity_input fails)


# This is a wrapper class to expose the ppstruct members as attributes
# (unfortunately, one cannot do dynamical attributes on cdef-classes)
class PathProp(cyprop._PathProp):
    '''
    Container class that holds all path profile properties.

    Parameters
    ----------
    freq : `~astropy.units.Quantity`
        Frequency of radiation [GHz]
    temperature : `~astropy.units.Quantity`
        Ambient temperature at path midpoint [K]
    pressure : `~astropy.units.Quantity`
        Ambient pressure at path midpoint  [hPa]
    lon_t, lat_t : `~astropy.units.Quantity`
        Geographic longitude/latitude of transmitter [deg]
    lon_r, lat_r : `~astropy.units.Quantity`
        Geographic longitude/latitude of receiver [deg]
    h_tg, h_rg : `~astropy.units.Quantity`
        Transmitter/receiver height over ground [m]
    hprof_step : `~astropy.units.Quantity`
        Distance resolution of height profile along path [m]
    timepercent : `~astropy.units.Quantity`
        Time percentage [%] (maximal 50%)
    omega : `~astropy.units.Quantity`, optional
        Fraction of the path over water [%] (see Table 3)
        (default: 0%)
    d_tm : `~astropy.units.Quantity`, optional
        longest continuous land (inland + coastal) section of the
        great-circle path [km]
        (default: distance between Tx and Rx)
    d_lm : `~astropy.units.Quantity`, optional
        longest continuous inland section of the great-circle path [km]
        (default: distance between Tx and Rx)
    d_ct, d_cr : `~astropy.units.Quantity`, optional
        Distance over land from transmitter/receiver antenna to the coast
        along great circle interference path [km]
        (default: 50000 km)
    zone_t, zone_r : CLUTTER enum, optional
        Clutter type for transmitter/receiver terminal.
        (default: CLUTTER.UNKNOWN)
    polarization : int, optional
        Polarization (default: 0)
        Allowed values are: 0 - horizontal, 1 - vertical
    version : int, optional
        ITU-R Rec. P.452 version. Allowed values are: 14, 16
    delta_N : `~astropy.units.Quantity`, optional
        Average radio-refractive index lapse-rate through the lowest 1 km of
        the atmosphere [N-units/km = 1/km]
        (default: query `~pycraf.pathprof.deltaN_N0_from_map`)
    N_0 : `~astropy.units.Quantity`, optional
        Sea-level surface refractivity [N-units = dimless]
        (default: query `~pycraf.pathprof.deltaN_N0_from_map`)
    hprof_dists : `~astropy.units.Quantity`, optional
        Distance vector associated with the height profile `hprof_heights`.
        (default: query `~pycraf.pathprof.srtm_height_profile`)
    hprof_heights : `~astropy.units.Quantity`, optional
        Terrain heights profile for the distances in `hprof_dists`.
        (default: query `~pycraf.pathprof.srtm_height_profile`)
    hprof_bearing : `~astropy.units.Quantity`, optional
        Start bearing of the height profile path.
        (default: query `~pycraf.pathprof.srtm_height_profile`)
    hprof_backbearing : `~astropy.units.Quantity`, optional
        Back-bearing of the height profile path.
        (default: query `~pycraf.pathprof.srtm_height_profile`)

    Returns
    -------
    pprop : PathProp instance

    Notes
    -----
    - The diffraction-loss algorithm was changed between ITU-R P.452
      version 14 and 15. The former used a Deygout method, the new one
      is based on a Bullington calculation with correction terms.

    - Set `d_ct` and `d_cr` to zero for a terminal on ship or on a sea
      platform; only relevant if less than 5 km.

    - Per default, the values for `delta_N` and `N_0` are queried from
      a radiometeorological map provided with `ITU-R Rec. P.452
      <https://www.itu.int/rec/R-REC-P.452-16-201507-I/en>`_
      using the function `~pycraf.pathprof.deltaN_N0_from_map`. If
      you want to use your own values, you have to provide both,
      `delta_N` and `N_0`.

    - Per default, the height-profile data are queried from
      `SRTM data <https://www2.jpl.nasa.gov/srtm/>`_ using the
      `~pycraf.pathprof.srtm_height_profile` function. If
      you want to use your own values, you have to provide all four
      parameters: `hprof_dists`, `hprof_heights`, `bearing`, and
      `back_bearing`.

      If you *don't do* the automatic query from SRTM data, make sure that
      the first element should is zero (transmitter location) and the last
      element is the distance between Tx and Rx. Also, the given
      `lon_t`, `lat_t` and `lon_r`, `lat_r` values should be consistent
      with the height profile. The bearings can be set to zero, if you
      don't need to calculate boresight angles.

      If you *do* the automatic query from
      `SRTM data <https://www2.jpl.nasa.gov/srtm/>`_ you need to downloaded
      them first. Also, an environment variable `SRTMDATA` has to be
      set to point to the directory containing the .hgt files; see
      :ref:`srtm_data`.
    '''

    @utils.ranged_quantity_input(
        freq=(0.1, 100, apu.GHz),
        temperature=(None, None, apu.K),
        pressure=(None, None, apu.hPa),
        lon_t=(-180, 180, apu.deg),
        lat_t=(-90, 90, apu.deg),
        lon_r=(-180, 180, apu.deg),
        lat_r=(-90, 90, apu.deg),
        h_tg=(None, None, apu.m),
        h_rg=(None, None, apu.m),
        hprof_step=(None, None, apu.m),
        timepercent=(0, 50, apu.percent),
        omega=(0, 100, apu.percent),
        d_tm=(None, None, apu.m),
        d_lm=(None, None, apu.m),
        d_ct=(None, None, apu.m),
        d_cr=(None, None, apu.m),
        delta_N=(None, None, cnv.dimless / apu.km),
        N0=(None, None, cnv.dimless),
        hprof_dists=(None, None, apu.km),
        hprof_heights=(None, None, apu.m),
        hprof_bearing=(None, None, apu.deg),
        hprof_backbearing=(None, None, apu.deg),
        strip_input_units=True, allow_none=True, output_unit=None
        )
    def __init__(
            self,
            freq,
            temperature,
            pressure,
            lon_t, lat_t,
            lon_r, lat_r,
            h_tg, h_rg,
            hprof_step,
            timepercent,
            omega=0 * apu.percent,
            d_tm=None, d_lm=None,
            d_ct=None, d_cr=None,
            zone_t=cyprop.CLUTTER.UNKNOWN, zone_r=cyprop.CLUTTER.UNKNOWN,
            polarization=0,
            version=16,
            # override if you don't want builtin method:
            delta_N=None, N0=None,
            # override if you don't want builtin method:
            hprof_dists=None, hprof_heights=None,
            hprof_bearing=None, hprof_backbearing=None,
            ):

        super().__init__(
            freq,
            temperature,
            pressure,
            lon_t, lat_t,
            lon_r, lat_r,
            h_tg, h_rg,
            hprof_step,
            timepercent,
            omega=omega,
            d_tm=d_tm, d_lm=d_lm,
            d_ct=d_ct, d_cr=d_cr,
            zone_t=zone_t, zone_r=zone_r,
            polarization=polarization,
            version=version,
            delta_N=delta_N, N0=N0,
            hprof_dists=hprof_dists,
            hprof_heights=hprof_heights,
            hprof_bearing=hprof_bearing,
            hprof_backbearing=hprof_backbearing,
            )

        self.__params = list(cyprop.PARAMETERS_BASIC)  # make a copy
        if self._pp['version'] == 14:
            self.__params += cyprop.PARAMETERS_V14
        elif self._pp['version'] == 16:
            self.__params += cyprop.PARAMETERS_V16

        # no need to set property, as readonly and immutable
        # can just copy to __dict__
        # for p in self.__params:
        #     setattr(
        #         PathProp,
        #         p[0],
        #         property(lambda self: getattr(self._pp, p[0]))
        #         )

        for p in self.__params:
            self.__dict__[p[0]] = self._pp[p[0]] * p[3]

    def __repr__(self):

        return 'PathProp<Freq: {:.3f}>'.format(self.freq)

    def __str__(self):

        return '\n'.join(
            '{}: {{:{}}} {}'.format(
                '{:15s}', p[1], '{:10s}'
                ).format(
                p[0], self._pp[p[0]], p[2]
                )
            for p in self.__params
            )


@utils.ranged_quantity_input(
    output_unit=(cnv.dB, cnv.dB, cnv.dB)
    )
def loss_freespace(pathprop):
    '''
    Calculate the free-space loss, L_bfsg, of a propagating radio wave
    according to ITU-R P.452-16 Eq (8-12).

    Parameters
    ----------
    pathprop : `~pycraf.pathprof.PathProp` instance
        This helper class works as a container to hold various properties
        of the path (e.g., geometry).

    Returns
    -------
    L_bfsg : `~astropy.units.Quantity`
        Free-space loss [dB]
    E_sp : `~astropy.units.Quantity`
        Focussing/multipath correction for p% [dB]
    E_sbeta : `~astropy.units.Quantity`
        Focussing/multipath correction for beta0% [dB]


    Notes
    -----
    - This function is similar to the `~pycraf.conversions.free_space_loss`
      function but additionally accounts for the atmospheric absorption and
      corrects for focusing and multipath effects.
    - With the return values one can also form::

          L_b0p = L_bfsg + E_sp
          L_b0beta = L_bfsg + E_sbeta

      which is necessary for some steps in the ITU-R P.452 algorithms.
    '''

    return cyprop.free_space_loss_bfsg_cython(pathprop)


@utils.ranged_quantity_input(
    G_t=(None, None, cnv.dBi),
    G_r=(None, None, cnv.dBi),
    strip_input_units=True, output_unit=cnv.dB
    )
def loss_troposcatter(
        pathprop, G_t=0. * cnv.dBi, G_r=0. * cnv.dBi,
        ):
    '''
    Calculate the tropospheric scatter loss, L_bs, of a propagating radio wave
    according to ITU-R P.452-16 Eq (45).

    Parameters
    ----------
    pathprop : `~pycraf.pathprof.PathProp` instance
        This helper class works as a container to hold various properties
        of the path (e.g., geometry).
    G_t, G_r  : `~astropy.units.Quantity`
        Antenna gain (transmitter, receiver) in the direction of the
        horizon(!) along the great-circle interference path [dBi]

    Returns
    -------
    L_bs : `~astropy.units.Quantity`
        Tropospheric scatter loss [dB]
    '''

    return cyprop.tropospheric_scatter_loss_bs_cython(pathprop, G_t, G_r)


@utils.ranged_quantity_input(output_unit=cnv.dB)
def loss_ducting(pathprop):
    '''
    Calculate the ducting/layer reflection loss, L_ba, of a propagating radio
    wave according to ITU-R P.452-16 Eq (46-56).

    Parameters
    ----------
    pathprop : `~pycraf.pathprof.PathProp` instance
        This helper class works as a container to hold various properties
        of the path (e.g., geometry).

    Returns
    -------
    L_ba : `~astropy.units.Quantity`
        Ducting/layer reflection loss [dB]
    '''

    return cyprop.ducting_loss_ba_cython(pathprop)


@utils.ranged_quantity_input(
    output_unit=(cnv.dB, cnv.dB, cnv.dB, cnv.dB, cnv.dB)
    )
def loss_diffraction(pathprop):
    '''
    Calculate the Diffraction loss of a propagating radio
    wave according to ITU-R P.452-16 Eq (14-44).

    Parameters
    ----------
    pathprop : `~pycraf.pathprof.PathProp` instance
        This helper class works as a container to hold various properties
        of the path (e.g., geometry).

    Returns
    -------
    L_d_50 : `~astropy.units.Quantity`
        Median diffraction loss [dB]
    L_dp : `~astropy.units.Quantity`
        Diffraction loss not exceeded for p% time, [dB]
    L_bd_50 : `~astropy.units.Quantity`
        Median basic transmission loss associated with
        diffraction [dB]::

            L_bd_50 = L_bfsg + L_d50

    L_bd : `~astropy.units.Quantity`
        Basic transmission loss associated with diffraction not
        exceeded for p% time [dB]::

            L_bd = L_b0p + L_dp

    L_min_b0p : `~astropy.units.Quantity`
        Notional minimum basic transmission loss associated with
        LoS propagation and over-sea sub-path diffraction

    Notes
    -----
    - L_d_50 and L_dp are just intermediary values; the complete
      diffraction loss is L_bd_50 or L_bd, respectively (taking into
      account a free-space loss component for the diffraction path)
    '''

    return cyprop.diffraction_loss_complete_cython(pathprop)


@utils.ranged_quantity_input(
    G_t=(None, None, cnv.dBi),
    G_r=(None, None, cnv.dBi),
    strip_input_units=True,
    output_unit=(cnv.dB, cnv.dB, cnv.dB, cnv.dB, cnv.dB, cnv.dB, cnv.dB)
    )
def loss_complete(
        pathprop, G_t=0. * cnv.dBi, G_r=0. * cnv.dBi,
        ):
    '''
    Calculate the total loss of a propagating radio
    wave according to ITU-R P.452-16 Eq (58-64).

    Parameters
    ----------
    pathprop : `~pycraf.pathprof.PathProp` instance
        This helper class works as a container to hold various properties
        of the path (e.g., geometry).
    G_t, G_r  : `~astropy.units.Quantity`
        Antenna gain (transmitter, receiver) in the direction of the
        horizon(!) along the great-circle interference path [dBi]

    Returns
    -------
    L_bfsg : `~astropy.units.Quantity`
        Free-space loss [dB]
    L_bd : `~astropy.units.Quantity`
        Basic transmission loss associated with diffraction not
        exceeded for p% time [dB]::

            L_bd = L_b0p + L_dp

    L_bs : `~astropy.units.Quantity`
        Tropospheric scatter loss [dB]
    L_ba : `~astropy.units.Quantity`
        Ducting/layer reflection loss [dB]
    L_b : `~astropy.units.Quantity`
        Complete path propagation loss [dB]
    L_b_corr : `~astropy.units.Quantity`
        As L_b but with clutter correction [dB]
    L : `~astropy.units.Quantity`
        As L_b_corr but with gain and clutter correction [dB]
    '''

    return cyprop.path_attenuation_complete_cython(pathprop, G_t, G_r)


@utils.ranged_quantity_input(
    h_g=(None, None, apu.m),
    freq=(None, None, apu.GHz),
    strip_input_units=True,
    output_unit=cnv.dB
    )
def clutter_correction(
        h_g, zone, freq
        ):
    '''
    Calculate the Clutter loss of a propagating radio
    wave according to ITU-R P.452-16 Eq (57).

    Parameters
    ----------
    h_g : `~astropy.units.Quantity`
        Height over ground [m]
    zone : CLUTTER enum, optional
        Clutter category for terminal.
    freq : `~astropy.units.Quantity`
        Frequency of radiation [GHz]

    Returns
    -------
    A_h : `~astropy.units.Quantity`
        Clutter correction to path attenuation [dB]
    '''

    return cyprop.clutter_correction_cython(h_g, zone, freq)


def _clutter_imt(
        freq,
        dist,
        location_percent,
        num_end_points=1,
        ):

    assert num_end_points in [1, 2]

    def Qinv(x):
        # Note, this is *not* identical to cyprop._I_helper
        # only good between 1.e-6 and 0.5
        # See R-Rec P.1546

        x = np.atleast_1d(x).copy()
        mask = x > 0.5
        x[mask] = 1 - x[mask]

        T = np.sqrt(-2 * np.log(x))
        Z = (
            (
                ((0.010328 * T + 0.802853) * T) + 2.515516698
                ) /
            (
                ((0.001308 * T + 0.189269) * T + 1.432788) * T + 1.
                )
            )

        Q = T - Z
        Q[mask] *= -1
        return Q

    # def Qinv(x):
    #     # larger x range than the approximation given in P.1546?
    #     # definitely much slower

    #     from scipy.stats import norm as qnorm

    #     x = np.atleast_1d(x).copy()

    #     mask = x > 0.5
    #     x[mask] = 1 - x[mask]

    #     Q = -qnorm.ppf(x, 0)
    #     Q[mask] *= -1

    #     return Q

    L_l = 23.5 + 9.6 * np.log10(freq)
    L_s = 32.98 + 23.9 * np.log10(dist) + 3.0 * np.log10(freq)

    L_clutter = -5 * np.log10(
        np.power(10, -0.2 * L_l) + np.power(10, -0.2 * L_s)
        ) - 6 * Qinv(location_percent / 100.)

    if num_end_points == 2:
        L_clutter *= 2

    return L_clutter


@utils.ranged_quantity_input(
    freq=(2, 67, apu.GHz),
    dist=(0.25, None, apu.km),
    location_percent=(0, 100, apu.percent),
    strip_input_units=True, output_unit=cnv.dB
    )
def clutter_imt(
        freq,
        dist,
        location_percent,
        num_end_points=1,
        ):
    '''
    Calculate the Clutter loss according to IMT.CLUTTER document (method 2).

    Parameters
    ----------
    freq : `~astropy.units.Quantity`
        Frequency of radiation [GHz]
    dist : `~astropy.units.Quantity`
        Distance between Tx/Rx antennas [km]

        Minimal distance must be 0.25 km (single endpoint clutter) or 1 km
        (if both endpoints are to be corrected for clutter)
    location_percent : `~astropy.units.Quantity`
        Percentage of locations for which the clutter loss `L_clutter`
        (calculated with this function) will not be exceeded [%]
    num_end_points : int, optional
        number of endpoints affected by clutter, allowed values: 1, 2

    Returns
    -------
    L_clutter : `~astropy.units.Quantity`
        Clutter loss [dB]

    Notes
    -----
    - The algorithm is independent of effective antenna height (w.r.t.
      clutter height), i.e., it doesn't distinguish between terminals which
      are close to the ground and those closer to the top of the building.
      However, the model is only appropriate if the terminal is "in the
      clutter", below the rooftops.
    - The result of this function is to be understood as a cumulative
      value. For example, if `location_percent = 2%`, it means that for
      2% of all possible locations, the clutter loss will not exceed the
      returned `L_clutter` value, for the remaining 98% of locations it
      will therefore be lower than `L_clutter`. The smaller `location_percent`,
      the smaller the returned `L_clutter`, i.e., low clutter attenuations
      are more unlikely.
    - This model was proposed by ITU study group SG 3 to replace
      `~pycraf.pathprof.clutter_correction` for IMT 5G studies (especially,
      at higher frequencies, where multipath effects play a role in
      urban and suburban areas).
    '''

    return _clutter_imt(
        freq,
        dist,
        location_percent,
        num_end_points=num_end_points,
        )


# TODO: do we want to convert output dictionary arrays to quantities?
@utils.ranged_quantity_input(
    lon_t=(-180, 180, apu.deg),
    lat_t=(-90, 90, apu.deg),
    map_size_lon=(0.002, 90, apu.deg),
    map_size_lat=(0.002, 90, apu.deg),
    map_resolution=(0.0001, 0.1, apu.deg),
    d_tm=(None, None, apu.m),
    d_lm=(None, None, apu.m),
    d_ct=(None, None, apu.m),
    d_cr=(None, None, apu.m),
    strip_input_units=True, allow_none=True, output_unit=None
    )
def height_profile_data(
        lon_t, lat_t,
        map_size_lon, map_size_lat,
        map_resolution=3. * apu.arcsec,
        do_cos_delta=True,
        zone_t=cyprop.CLUTTER.UNKNOWN, zone_r=cyprop.CLUTTER.UNKNOWN,
        d_tm=None, d_lm=None,
        d_ct=None, d_cr=None,
        ):

    '''
    Calculate height profiles and auxillary maps needed for
    `~pycraf.pathprof.atten_map_fast`.

    This can be used to cache height-profile data. Since it is independent
    of frequency, timepercent, Tx and Rx heights, etc., one can re-use
    it to save computing time when doing batch jobs.

    The basic idea is to only calculate the height profiles between each of
    the pixels on the map edges (super-sampled with a factor of ~3) and the
    map center. The `me` height profiles have different lengths, because
    the distance to the pixels on the edges varies (e.g., corners have the
    largest distance from the center). For shorter profiles, the arrays are
    zero-padded such that all `me` profiles have the same length `mh`. They
    are stored in a 2D array, called `height_profs`. Associated with this,
    a 1D array, `dist_prof`, contains the associated path distances. It has
    length `mh`. For convenience, also a zero-valued array, `zheight_prof`
    is stored, having length `mh`. This is used for the flat-Earth
    calculations in P.452.

    Now, for each other pixel in the map, we have to find out, which of the
    above height profile paths comes closest (with one of its elements) to
    the pixel in question. The identified path ID is stored in `path_idx_map`,
    and the index of the distance element in that path is kept in
    `dist_end_idx_map`. For debugging reasons, the angular distance of
    the "best" path is provided in `pix_dist_map`, but this is not used
    in `~pycraf.pathprof.atten_map_fast`.

    Parameters
    ----------
    lon_t, lat_t : `~astropy.units.Quantity`
        Geographic longitude/latitude of transmitter [deg]
    map_size_lon, map_size_lat : `~astropy.units.Quantity`
        Map size in longitude/latitude[deg]
    map_resolution : `~astropy.units.Quantity`, optional
        Pixel resolution of map [deg] (default: 3 arcsec)
    do_cos_delta : bool, optional
        If True, divide `map_size_lon` by `cos(lat_t)` to produce a more
        square-like map. (default: True)
    zone_t, zone_r : CLUTTER enum, optional
        Clutter type for transmitter/receiver terminal.
        (default: CLUTTER.UNKNOWN)
    d_tm : `~astropy.units.Quantity`, optional
        longest continuous land (inland + coastal) section of the
        great-circle path [km]
        (default: distance between Tx and Rx)
    d_lm : `~astropy.units.Quantity`, optional
        longest continuous inland section of the great-circle path [km]
        (default: distance between Tx and Rx)
    d_ct, d_cr : `~astropy.units.Quantity`, optional
        Distance over land from transmitter/receiver antenna to the coast
        along great circle interference path [km]
        (default: 50000 km)

    Returns
    -------
    hprof_data : dict
        Dictionary with height profiles and auxillary data as
        calculated with `~pycraf.pathprof.height_profile_data`.

        The dictionary contains the following entities (the map dimension
        is mx * my):

        - "lon_t", "lat_t" : float

          Map center coordinates.

        - "map_size_lon", "map_size_lat" : float

          Map size.

        - "hprof_step" : float

          Distance resolution of height profile.

        - "do_cos_delta" : int

          Whether cos-delta correction was applied for map size creation.

        - "xcoords", "ycoords" : `~numpy.ndarray` 1D (float; (mx, ); (my, ))

          Longitude and latitude coordinates of first row and first column
          in the map

        - "lon_mid_map", "lat_mid_map" : `~numpy.ndarray` 2D (float; (mx, my))

          Longitude and latitude path center coordinates for each pixel
          w.r.t. map center.

        - "dist_map" : `~numpy.ndarray` 2D (float; (mx, my))

          Distances to map center for each pixel.

        - "d_ct_map", "d_cr_map" : `~numpy.ndarray` 2D (float; (mx, my))

          The `d_ct` and `d_cr` values for each pixel in the map.

        - "d_lm_map", "d_tm_map" : `~numpy.ndarray` 2D (float; (mx, my))

          The `d_lm` and `d_tm` values for each pixel in the map.

        - "zone_t_map", "zone_r_map" : `~numpy.ndarray` 2D (CLUTTER enum; (mx, my))

          The clutter zone types `zone_t` and `zone_r` for each pixel in the
          map.

        - "bearing_map", "back_bearing_map" : `~numpy.ndarray` 2D (float; (mx, my))

          The `bearing` and `backbearing` values for each pixel in the map.

        - "N0_map", "delta_N_map", "beta0_map" : `~numpy.ndarray` 2D (float; (mx, my))

          The `N0`, `delta_N`, and `beta0` values for each pixel in the map.

        - "path_idx_map" : `~numpy.ndarray` 2D (int; (mx, my))

          Path IDs for each pixel in the map. With this index, one can query
          the associated height profile from `height_profs`.

        - "pix_dist_map" : `~numpy.ndarray` 2D (float; (mx, my))

          Angular distance of the closest path to each of the map pixels.

        - "dist_end_idx_map" : `~numpy.ndarray` 2D (int; (mx, my))

          Index of the last element in the dist/height profiles to be used
          when querying the profiles from `dist_prof` and
          `height_profs`.

        - "dist_prof" : `~numpy.ndarray` 1D (float, (mh, ))

          Distance values for each of the paths stored in `height_profs`.

        - "height_profs" : `~numpy.ndarray` 2D (float, (me, mh))

          Height profiles to each of the pixels on the map edge, zero padded.

        - "zheight_prof" : `~numpy.ndarray` 1D (float, (mh, ))

          Zero-valued array of the same length as `height_profs` for
          convenience.

    Notes
    -----
    - Path attenuation is completely symmetric, i.e., it doesn't matter if
      the transmitter or the receiver is situated in the map center.
    - `SRTM data <https://www2.jpl.nasa.gov/srtm/>`_ need to be downloaded
      manually by the user. An environment variable `SRTMDATA` has to be
      set to point to the directory containing the .hgt files; see
      :ref:`srtm_data`.
    '''

    return cyprop.height_profile_data_cython(
        lon_t, lat_t,
        map_size_lon, map_size_lat,
        map_resolution=map_resolution,
        do_cos_delta=1 if do_cos_delta else 0,
        zone_t=zone_t, zone_r=zone_r,
        d_tm=d_tm, d_lm=d_lm,
        d_ct=d_ct, d_cr=d_cr,
        )


@utils.ranged_quantity_input(
    freq=(0.1, 100, apu.GHz),
    temperature=(None, None, apu.K),
    pressure=(None, None, apu.hPa),
    h_tg=(None, None, apu.m),
    h_rg=(None, None, apu.m),
    timepercent=(0, 50, apu.percent),
    omega_percent=(0, 100, apu.percent),
    strip_input_units=True,
    output_unit=(cnv.dB, apu.deg, apu.deg)
    )
def atten_map_fast(
        freq,
        temperature,
        pressure,
        h_tg, h_rg,
        timepercent,
        hprof_data,  # dict_like
        omega=0 * apu.percent,
        polarization=0,
        version=16,
        ):
    '''
    Calculate attenuation maps using a fast method.

    Parameters
    ----------
    freq : `~astropy.units.Quantity`
        Frequency of radiation [GHz]
    temperature : `~astropy.units.Quantity`
        Temperature (K)
    pressure : `~astropy.units.Quantity`
        Pressure (hPa)
    h_tg, h_rg : `~astropy.units.Quantity`
        Transmitter/receiver heights over ground [m]
    timepercent : `~astropy.units.Quantity`
        Time percentage [%] (maximal 50%)
    hprof_data : dict, dict-like
        Dictionary with height profiles and auxillary maps as
        calculated with `~pycraf.pathprof.height_profile_data`.
    omega : `~astropy.units.Quantity`, optional
        Fraction of the path over water [%] (see Table 3)
        (default: 0%)
    polarization : int, optional
        Polarization (default: 0)
        Allowed values are: 0 - horizontal, 1 - vertical
    version : int, optional
        ITU-R Rec. P.452 version. Allowed values are: 14, 16

    Returns
    -------
    atten_map : 3D `~numpy.ndarray`
        Attenuation maps. First dimension has length 6, which refers to:

        0) L_bfsg - Free-space loss [dB]
        1) L_bd - Basic transmission loss associated with diffraction
           not exceeded for p% time [dB]; L_bd = L_b0p + L_dp
        2) L_bs - Tropospheric scatter loss [dB]
        3) L_ba - Ducting/layer reflection loss [dB]
        4) L_b - Complete path propagation loss [dB]
        5) L_b_corr - As L_b but with clutter correction [dB]

        (i.e., the output of path_attenuation_complete without
        gain-corrected values)
    eps_pt_map : 2D `~numpy.ndarray`
        Elevation angle of paths w.r.t. Tx [deg]
    eps_pr_map : 2D `~numpy.ndarray`
        Elevation angle of paths w.r.t. Rx [deg]

    Notes
    -----
    - The diffraction-loss algorithm was changed between ITU-R P.452
      version 14 and 15. The former used a Deygout method, the new one
      is based on a Bullington calculation with correction terms.
    '''

    return cyprop.atten_map_fast_cython(
        freq,
        temperature,
        pressure,
        h_tg, h_rg,
        timepercent,
        hprof_data,  # dict_like
        omega=omega,
        polarization=polarization,
        version=version,
        )


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
