#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

# from functools import partial, lru_cache
from astropy import units as apu
import numpy as np

from . import cyprop
from . import heightprofile
from . import helper
from . import srtm
from .. import conversions as cnv
from .. import utils
# import ipdb


__all__ = [
    'PathProp',
    'loss_freespace', 'loss_troposcatter', 'loss_ducting',
    'loss_diffraction', 'loss_complete',
    'clutter_correction',
    'height_map_data', 'atten_map_fast',
    'height_path_data', 'height_path_data_generic', 'atten_path_fast',
    'losses_complete',
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
    generic_heights : bool
        If `generic_heights` is set to True, heights will be set to zero.
        This can be useful for generic (aka flat-Earth) computations.
        The option is only meaningful, if the hprof_xxx parameters are set
        to `None` (which means automatic querying of the profiles).
        (Default: False)

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

      If you *don't do* the automatic query from SRTM data, make sure
      that the first element in `hprof_dists` is zero (transmitter
      location) and the last element is the distance between Tx and Rx.
      Also, the given `lon_t`, `lat_t` and `lon_r`, `lat_r` values
      should be consistent with the height profile. The bearings can
      be set to zero, if you don't need to calculate boresight angles.

      `SRTM <https://www2.jpl.nasa.gov/srtm/>`_ data tiles (`*.hgt`) need
      to be accessible by `pycraf`.  It is assumed that these are either
      present in the current working directory or in the path defined by the
      `SRTMDATA` environment variable (sub-directories are also parsed).
      Alternatively, use the `~pycraf.pathprof.SrtmConf` manager to
      change the directory, where `pycraf` looks for SRTM data, during
      run-time. The `~pycraf.pathprof.SrtmConf` manager also offers
      additional features such as automatic downloading of missing
      tiles or applying different interpolation methods (e.g., splines).
      For details see :ref:`working_with_srtm`.
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
            generic_heights=False
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
            generic_heights=generic_heights,
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
    L_b0p : `~astropy.units.Quantity`
        Free-space loss including focussing effects (for p% of time) [dB]
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
    omega_percent=(0, 100, apu.percent),
    strip_input_units=True, allow_none=True, output_unit=None
    )
def height_map_data(
        lon_t, lat_t,
        map_size_lon, map_size_lat,
        map_resolution=3. * apu.arcsec,
        do_cos_delta=True,
        zone_t=cyprop.CLUTTER.UNKNOWN, zone_r=cyprop.CLUTTER.UNKNOWN,
        d_tm=None, d_lm=None,
        d_ct=None, d_cr=None,
        omega_percent=0 * apu.percent,
        cache_path=None, clobber=False,
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
    omega_percent : `~astropy.units.Quantity`, optional
        Fraction of the path over water [%] (see Table 3)
        (default: 0%)
    cache_path : str, optional
        If set, the `joblib package
        <https://joblib.readthedocs.io/en/latest/>`_ is used to cache
        results provided by this function on disk, such that future queries
        are executed much faster. If set to `None`, no caching is performed.
        (default: None)
    clobber : bool, optional
        If set to `True` and caching is active re-compute the result even
        if an older result is found in cache. This is useful, when something
        has change with the underlying terrain data, e.g., new tiles were
        downloaded. (default: `False`)

    Returns
    -------
    hprof_data : dict
        Dictionary with height profiles and auxillary data as
        calculated with `~pycraf.pathprof.height_map_data`.

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

          This is returned for information, only. It is not
          needed by `~pycraf.atten_map_fast`!

        - "dist_map" : `~numpy.ndarray` 2D (float; (mx, my))

          Distances to map center for each pixel.

        - "d_ct_map", "d_cr_map" : `~numpy.ndarray` 2D (float; (mx, my))

          The `d_ct` and `d_cr` values for each pixel in the map.

        - "d_lm_map", "d_tm_map" : `~numpy.ndarray` 2D (float; (mx, my))

          The `d_lm` and `d_tm` values for each pixel in the map.

        - "omega_map" : `~numpy.ndarray` 2D (float; (mx, my))

          The `omega` values for each pixel in the map.

        - "zone_t_map", "zone_r_map" : `~numpy.ndarray` 2D (CLUTTER enum; (mx, my))

          The clutter zone types `zone_t` and `zone_r` for each pixel in the
          map.

        - "bearing_map", "back_bearing_map" : `~numpy.ndarray` 2D (float; (mx, my))

          The `bearing` and `backbearing` values for each pixel in the map.

          This is returned for information, only. It is not
          needed by `~pycraf.atten_map_fast`!

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
    - `SRTM <https://www2.jpl.nasa.gov/srtm/>`_ data tiles (`*.hgt`) need
      to be accessible by `pycraf`.  It is assumed that these are either
      present in the current working directory or in the path defined by the
      `SRTMDATA` environment variable (sub-directories are also parsed).
      Alternatively, use the `~pycraf.pathprof.SrtmConf` manager to
      change the directory, where `pycraf` looks for SRTM data, during
      run-time. The `~pycraf.pathprof.SrtmConf` manager also offers
      additional features such as automatic downloading of missing
      tiles or applying different interpolation methods (e.g., splines).
      For details see :ref:`working_with_srtm`.
    '''

    args = lon_t, lat_t, map_size_lon, map_size_lat
    kwargs = dict(
        map_resolution=map_resolution,
        do_cos_delta=1 if do_cos_delta else 0,
        zone_t=int(zone_t), zone_r=int(zone_r),  # needed for joblib
        d_tm=d_tm, d_lm=d_lm,
        d_ct=d_ct, d_cr=d_cr,
        omega=omega_percent,
        )

    joblib_available = True
    try:
        import joblib
        memory = joblib.Memory(cache_path, compress=5, verbose=False)
    except ImportError:
        joblib_available = False

    if cache_path and joblib_available:

        def wrapped(*args, **kwargs):
            @memory.cache
            def f(tile_path, interp, spline_opts, *args, **kwargs):
                return cyprop.height_map_data_cython(*args, **kwargs)

            if clobber:
                result = f.call_and_shelve(
                    srtm.SrtmConf.srtm_dir,
                    srtm.SrtmConf.interp,
                    srtm.SrtmConf.spline_opts,
                    *args, **kwargs
                    )
                try:
                    result.clear()
                except KeyError:
                    pass

            return f(
                srtm.SrtmConf.srtm_dir,
                srtm.SrtmConf.interp,
                srtm.SrtmConf.spline_opts,
                *args, **kwargs
                )

    elif cache_path and not joblib_available:
        raise ImportError('"joblib" package must be installed for caching')
    else:
        wrapped = cyprop.height_map_data_cython

    return wrapped(*args, **kwargs)


@utils.ranged_quantity_input(
    freq=(0.1, 100, apu.GHz),
    temperature=(None, None, apu.K),
    pressure=(None, None, apu.hPa),
    h_tg=(None, None, apu.m),
    h_rg=(None, None, apu.m),
    timepercent=(0, 50, apu.percent),
    strip_input_units=True,
    )
def atten_map_fast(
        freq,
        temperature,
        pressure,
        h_tg, h_rg,
        timepercent,
        hprof_data,  # dict_like
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
        Dictionary with height profiles and auxillary maps
        of dimension `(my, mx)` as calculated with
        `~pycraf.pathprof.height_map_data`.
    polarization : int, optional
        Polarization (default: 0)
        Allowed values are: 0 - horizontal, 1 - vertical
    version : int, optional
        ITU-R Rec. P.452 version. Allowed values are: 14, 16

    Returns
    -------
    results : dict
        Results of the path attenuation calculation. Each entry
        in the dictionary is a 2D `~numpy.ndarray` containing
        the associated value for the map of dimension `(my, mx)`.
        The following entries are contained:

        - `L_b0p` - Free-space loss including focussing effects
            (for p% of time) [dB]

        - `L_bd` - Basic transmission loss associated with diffraction
            not exceeded for p% time [dB]; L_bd = L_b0p + L_dp

        - `L_bs` - Tropospheric scatter loss [dB]

        - `L_ba` - Ducting/layer reflection loss [dB]

        - `L_b` - Complete path propagation loss [dB]

        - `L_b_corr` - As L_b but with clutter correction [dB]

        - `eps_pt` - Elevation angle of paths w.r.t. Tx [deg]

        - `eps_pr` - Elevation angle of paths w.r.t. Rx [deg]

        - `d_lt` - Distance to horizon w.r.t. Tx [km]

        - `d_lr` - Distance to horizon w.r.t. Rx [km]

        - `path_type` - Path type (0 - LoS, 1 - Trans-horizon)

    Notes
    -----
    - The diffraction-loss algorithm was changed between ITU-R P.452
      version 14 and 15. The former used a Deygout method, the new one
      is based on a Bullington calculation with correction terms.
    - In future versions, more entries may be added to the results
      dictionary.
    '''

    float_res, int_res = cyprop.atten_map_fast_cython(
        freq,
        temperature,
        pressure,
        h_tg, h_rg,
        timepercent,
        hprof_data,  # dict_like
        polarization=polarization,
        version=version,
        )

    return {
        'L_b0p': float_res[0] * cnv.dB,
        'L_bd': float_res[1] * cnv.dB,
        'L_bs': float_res[2] * cnv.dB,
        'L_ba': float_res[3] * cnv.dB,
        'L_b': float_res[4] * cnv.dB,
        'L_b_corr': float_res[5] * cnv.dB,
        'eps_pt': float_res[6] * apu.deg,
        'eps_pr': float_res[7] * apu.deg,
        'd_lt': float_res[8] * apu.km,
        'd_lr': float_res[9] * apu.km,
        'path_type': int_res[0],
        }


@utils.ranged_quantity_input(
    lon_t=(-180, 180, apu.deg),
    lat_t=(-90, 90, apu.deg),
    lon_r=(-180, 180, apu.deg),
    lat_r=(-90, 90, apu.deg),
    step=(1., 1.e5, apu.m),
    strip_input_units=True,
    )
def height_path_data(
        lon_t, lat_t,
        lon_r, lat_r,
        step,
        zone_t=cyprop.CLUTTER.UNKNOWN, zone_r=cyprop.CLUTTER.UNKNOWN,
        ):

    '''
    Calculate height profile auxillary data needed for
    `~pycraf.pathprof.atten_path_fast`.

    This can be used to cache height-profile data. Since it is independent
    of frequency, timepercent, Tx and Rx heights, etc., one can re-use
    it to save computing time when doing batch jobs. It is assumed that
    the Tx is at a fixed position, while the Rx "moves" accross the
    specified paths (since the attenuation is completely symmetric,
    this is without loss of generality).

    Parameters
    ----------
    lon_t, lat_t : double
        Geographic longitude/latitude of start point (transmitter) [deg]
    lon_r, lat_r : double
        Geographic longitude/latitude of end point (receiver) [deg]
    step : double
        Distance resolution of height profile along path [m]
    zone_t, zone_r : CLUTTER enum, optional
        Clutter type for transmitter/receiver terminal.
        (default: CLUTTER.UNKNOWN)

    Returns
    -------
    hprof_data : dict
        Dictionary with height profile auxillary data.

        The dictionary contains the following entities (the path length
        is m):

        - "lons", "lats" : `~numpy.ndarray` 1D (float; (m,))

          Longitudes and latitudes of path.

          This is returned for information, only. It is not
          needed by `~pycraf.pathprof.atten_path_fast`!

        - "lon_mids", "lat_mids" : `~numpy.ndarray` 1D (float; (m,))

          Path center longitudes and latitudes of path. The values
          at index `i` in these arrays are the midpoints of the
          paths from `lons[0]` to `lons[i]` (likewise for `lats`).

          This is returned for information, only. It is not
          needed by `~pycraf.pathprof.atten_path_fast`!

        - "distances" : `~numpy.ndarray` 1D (float; (m,))

          Distances of each point in the path from Tx position.

        - "heights" : `~numpy.ndarray` 1D (float; (m,))

          Height profile.

        - "bearing" : float

          Start bearing of path.

          This is returned for information, only. It is not
          needed by `~pycraf.pathprof.atten_path_fast`!

        - "backbearings" : `~numpy.ndarray` 1D (float; (m,))

          Back-bearing of each point in the path.

          This is returned for information, only. It is not
          needed by `~pycraf.pathprof.atten_path_fast`!

        - "omega" : `~numpy.ndarray` 1D (float; (m,))

          The `omega` values for each point in the path.

        - "d_tm", "d_lm" : `~numpy.ndarray` 1D (float; (m,))

          The `d_tm` and `d_lm` values for each point in the path.

        - "d_ct", "d_cr" : `~numpy.ndarray` 1D (float; (m,))

          The `d_ct` and `d_cr` values for each point in the path.

        - "zone_t" : int

          Clutter type at Tx.

        - "zone_r" : `~numpy.ndarray` 1D (int; (m,))

          Clutter type at Rx. (Currently only a single type is used.)

        - "N0", "delta_N", "beta0" : `~numpy.ndarray` 1D (float; (m,))

          The `N0`, `delta_N`, and `beta0` values for each point in the path.

    Notes
    -----
    Currently, no sea or lake bodies are accounted for. Also the clutter
    type of the receiver is fixed, while one could also think of using
    different clutter types (in the array). You can modify the returned
    arrays in `hprof_data`, of course, before feeding into
    `~pycraf.pathprof.atten_path_fast`.
    '''

    (
        lons, lats, distance, distances, heights,
        bearing, back_bearing, backbearings
        ) = heightprofile._srtm_height_profile(
            lon_t, lat_t, lon_r, lat_r, step
            )

    # TODO: query the following programmatically
    # for now assume land-paths, only
    omega = np.zeros_like(heights)
    d_tm = distances.copy()
    d_lm = distances.copy()
    d_ct = np.full_like(heights, 50000)
    d_cr = np.full_like(heights, 50000)

    # radiomet data:
    # get path centers for each pair of pixels (0 - i)
    mid_idx = [i // 2 for i in range(len(distances))]
    lon_mids = lons[mid_idx]
    lat_mids = lats[mid_idx]

    delta_N, beta0, N0 = helper._radiomet_data_for_pathcenter(
        lon_mids, lat_mids, d_tm, d_lm
        )

    hprof_data = {}
    hprof_data['lons'] = lons  # <-- ignored
    hprof_data['lats'] = lats  # <-- ignored
    hprof_data['lon_mids'] = lon_mids  # <-- ignored
    hprof_data['lat_mids'] = lat_mids  # <-- ignored
    hprof_data['distances'] = distances
    hprof_data['heights'] = heights
    # hprof_data['zheights'] = np.zeros_like(heights)
    hprof_data['bearing'] = bearing  # <-- scalar
    hprof_data['backbearings'] = backbearings
    hprof_data['omega'] = omega
    hprof_data['d_tm'] = d_tm
    hprof_data['d_lm'] = d_lm
    hprof_data['d_ct'] = d_ct
    hprof_data['d_cr'] = d_cr
    hprof_data['zone_t'] = zone_t  # <-- scalar
    hprof_data['zone_r'] = np.full(heights.shape, zone_r, dtype=np.int32)
    hprof_data['delta_N'] = delta_N
    hprof_data['N0'] = N0
    hprof_data['beta0'] = beta0

    return hprof_data


@utils.ranged_quantity_input(
    distance=(0, None, apu.km),
    step=(1., 1.e5, apu.m),
    lon_mid=(-180, 360, apu.deg),
    lat_mid=(-90, 90, apu.deg),
    strip_input_units=True,
    )
def height_path_data_generic(
        distance, step,
        lon_mid, lat_mid,
        zone_t=cyprop.CLUTTER.UNKNOWN, zone_r=cyprop.CLUTTER.UNKNOWN,
        ):

    '''
    Calculate height profile auxillary data needed for
    `~pycraf.pathprof.atten_path_fast`.

    This can be used to cache height-profile data. Since it is independent
    of frequency, timepercent, Tx and Rx heights, etc., one can re-use
    it to save computing time when doing batch jobs. It is assumed that
    the Tx is at a fixed position, while the Rx "moves" accross the
    specified paths (since the attenuation is completely symmetric,
    this is without loss of generality).

    Parameters
    ----------
    distance : double
        Maximal path length [km]
    step : double
        Distance resolution of height profile along path [m]
    lon_mid, lat_mid : double
        Geographic longitude/latitude of path's mid point [deg]

        This is needed to query radiometerological values.
    zone_t, zone_r : CLUTTER enum, optional
        Clutter type for transmitter/receiver terminal.
        (default: CLUTTER.UNKNOWN)

    Returns
    -------
    hprof_data : dict
        Dictionary with height profile auxillary data.

        The dictionary contains the following entities (the path length
        is m):

        - "lons", "lats" : `~numpy.ndarray` 1D (float; (m,))

          Longitudes and latitudes of path.

          This is returned for information, only. It is not
          needed by `~pycraf.pathprof.atten_path_fast`!

        - "lon_mids", "lat_mids" : `~numpy.ndarray` 1D (float; (m,))

          Path center longitudes and latitudes of path. The values
          at index `i` in these arrays are the midpoints of the
          paths from `lons[0]` to `lons[i]` (likewise for `lats`).

          This is returned for information, only. It is not
          needed by `~pycraf.pathprof.atten_path_fast`!

        - "distances" : `~numpy.ndarray` 1D (float; (m,))

          Distances of each point in the path from Tx position.

        - "heights" : `~numpy.ndarray` 1D (float; (m,))

          Height profile.

        - "bearing" : float

          Start bearing of path.

          This is returned for information, only. It is not
          needed by `~pycraf.pathprof.atten_path_fast`!

        - "backbearings" : `~numpy.ndarray` 1D (float; (m,))

          Back-bearing of each point in the path.

          This is returned for information, only. It is not
          needed by `~pycraf.pathprof.atten_path_fast`!

        - "omega" : `~numpy.ndarray` 1D (float; (m,))

          The `omega` values for each point in the path.

        - "d_tm", "d_lm" : `~numpy.ndarray` 1D (float; (m,))

          The `d_tm` and `d_lm` values for each point in the path.

        - "d_ct", "d_cr" : `~numpy.ndarray` 1D (float; (m,))

          The `d_ct` and `d_cr` values for each point in the path.

        - "zone_t" : int

          Clutter type at Tx.

        - "zone_r" : `~numpy.ndarray` 1D (int; (m,))

          Clutter type at Rx. (Currently only a single type is used.)

        - "N0", "delta_N", "beta0" : `~numpy.ndarray` 1D (float; (m,))

          The `N0`, `delta_N`, and `beta0` values for each point in the path.

    Notes
    -----
    Currently, no sea or lake bodies are accounted for. Also the clutter
    type of the receiver is fixed, while one could also think of using
    different clutter types (in the array). You can modify the returned
    arrays in `hprof_data`, of course, before feeding into
    `~pycraf.pathprof.atten_path_fast`.
    '''

    step /= 1000.
    distances = np.arange(0, distance + step, step)
    heights = np.zeros_like(distances)

    # TODO: query the following programmatically
    # for now assume land-paths, only
    omega = heights.copy()
    d_tm = distances.copy()
    d_lm = distances.copy()
    d_ct = np.full_like(heights, 50000)
    d_cr = np.full_like(heights, 50000)

    # radiomet data:
    # assume all points on path have equal radiomet values
    lon_mids = np.full_like(heights, lon_mid)
    lat_mids = np.full_like(heights, lat_mid)

    delta_N, beta0, N0 = helper._radiomet_data_for_pathcenter(
        lon_mids, lat_mids, d_tm, d_lm
        )

    hprof_data = {}
    hprof_data['distances'] = distances
    hprof_data['heights'] = heights
    hprof_data['omega'] = omega
    hprof_data['d_tm'] = d_tm
    hprof_data['d_lm'] = d_lm
    hprof_data['d_ct'] = d_ct
    hprof_data['d_cr'] = d_cr
    hprof_data['zone_t'] = zone_t  # <-- scalar
    hprof_data['zone_r'] = np.full(heights.shape, zone_r, dtype=np.int32)
    hprof_data['delta_N'] = delta_N
    hprof_data['N0'] = N0
    hprof_data['beta0'] = beta0

    return hprof_data


@utils.ranged_quantity_input(
    freq=(0.1, 100, apu.GHz),
    temperature=(None, None, apu.K),
    pressure=(None, None, apu.hPa),
    h_tg=(None, None, apu.m),
    h_rg=(None, None, apu.m),
    timepercent=(0, 50, apu.percent),
    strip_input_units=True,
    )
def atten_path_fast(
        freq,
        temperature,
        pressure,
        h_tg, h_rg,
        timepercent,
        hprof_data,  # dict_like
        polarization=0,
        version=16,
        ):
    '''
    Calculate attenuation along a path using a parallelized method.

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
        Dictionary with height profile auxillary data as
        calculated with `~pycraf.pathprof.height_path_data` or
        `~pycraf.pathprof.height_path_data_generic`.
    polarization : int, optional
        Polarization (default: 0)
        Allowed values are: 0 - horizontal, 1 - vertical
    version : int, optional
        ITU-R Rec. P.452 version. Allowed values are: 14, 16

    Returns
    -------
    results : dict
        Results of the path attenuation calculation. Each entry
        in the dictionary is a 1D `~numpy.ndarray` containing
        the associated value for the path.
        The following entries are contained:

        - `L_b0p` - Free-space loss including focussing effects
           (for p% of time) [dB]

        - `L_bd` - Basic transmission loss associated with diffraction
            not exceeded for p% time [dB]; L_bd = L_b0p + L_dp

        - `L_bs` - Tropospheric scatter loss [dB]

        - `L_ba` - Ducting/layer reflection loss [dB]

        - `L_b` - Complete path propagation loss [dB]

        - `L_b_corr` - As L_b but with clutter correction [dB]

        - `eps_pt` - Elevation angle of paths w.r.t. Tx [deg]

        - `eps_pr` - Elevation angle of paths w.r.t. Rx [deg]

        - `d_lt` - Distance to horizon w.r.t. Tx [km]

        - `d_lr` - Distance to horizon w.r.t. Rx [km]

        - `path_type` - Path type (0 - LoS, 1 - Trans-horizon)

    Notes
    -----
    - The diffraction-loss algorithm was changed between ITU-R P.452
      version 14 and 15. The former used a Deygout method, the new one
      is based on a Bullington calculation with correction terms.
    - In future versions, more entries may be added to the results
      dictionary.

    Examples
    --------

    A typical usage would be::

        import numpy as np
        import matplotlib.pyplot as plt
        from astropy import units as u
        from pycraf import pathprof


        lon_t, lat_t = 6.8836 * u.deg, 50.525 * u.deg
        lon_r, lat_r = 7.3334 * u.deg, 50.635 * u.deg
        hprof_step = 100 * u.m

        hprof_data = pathprof.height_path_data(
            lon_t, lat_t, lon_r, lat_r, hprof_step,
            zone_t=pathprof.CLUTTER.URBAN, zone_r=pathprof.CLUTTER.SUBURBAN,
            )

        plt.plot(hprof_data['distances'], hprof_data['heights'], 'k-')
        plt.grid()
        plt.show()

        freq = 1. * u.GHz
        temperature = 290. * u.K
        pressure = 1013. * u.hPa
        h_tg, h_rg = 5. * u.m, 50. * u.m
        time_percent = 2. * u.percent

        results = pathprof.atten_path_fast(
            freq, temperature, pressure,
            h_tg, h_rg, time_percent,
            hprof_data,
            )

        for k in ['L_bfsg', 'L_bd', 'L_bs', 'L_ba', 'L_b', 'L_b_corr']:
            plt.plot(hprof_data['distances'], results[k], '-')

        plt.ylim((50, 275))
        plt.grid()
        plt.legend(
            ['LOS', 'Diffraction', 'Troposcatter', 'Ducting',
            'Total', 'Total w. clutter'
            ], fontsize=10, loc='lower right')
        plt.show()

        plt.plot(hprof_data['distances'], eps_pt_path, 'b-')
        plt.plot(hprof_data['distances'], eps_pr_path, 'r-')
        plt.grid()
        plt.show()
    '''

    float_res, int_res = cyprop.atten_path_fast_cython(
        freq,
        temperature,
        pressure,
        h_tg, h_rg,
        timepercent,
        hprof_data,  # dict_like
        polarization=polarization,
        version=version,
        )

    return {
        'L_b0p': float_res[0] * cnv.dB,
        'L_bd': float_res[1] * cnv.dB,
        'L_bs': float_res[2] * cnv.dB,
        'L_ba': float_res[3] * cnv.dB,
        'L_b': float_res[4] * cnv.dB,
        'L_b_corr': float_res[5] * cnv.dB,
        'eps_pt': float_res[6] * apu.deg,
        'eps_pr': float_res[7] * apu.deg,
        'd_lt': float_res[8] * apu.km,
        'd_lr': float_res[9] * apu.km,
        'path_type': int_res[0],
        }


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
    G_t=(None, None, cnv.dBi),
    G_r=(None, None, cnv.dBi),
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
def losses_complete(
        freq,
        temperature,
        pressure,
        lon_t, lat_t,
        lon_r, lat_r,
        h_tg, h_rg,
        hprof_step,
        timepercent,
        G_t=0. * cnv.dBi, G_r=0. * cnv.dBi,
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
        generic_heights=False,
        ):
    '''
    Calculate propagation losses for a fixed path using a parallelized method.

    The difference to the usual `~pycraf.pathprof.PathProp` +
    `~pycraf.pathprof.loss_complete` approach is, that `losses_complete`
    supports full `numpy broad-casting
    <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_. This
    allows perform many calculations at once, e.g., if one interested in
    a statistics plot of `L` vs. `time_percent`, without querying the
    height profile over and over.

    Parameters
    ----------
    freq : `~astropy.units.Quantity`
        Frequency of radiation [GHz]
    temperature : `~astropy.units.Quantity`
        Ambient temperature at path midpoint [K]
    pressure : `~astropy.units.Quantity`
        Ambient pressure at path midpoint  [hPa]
    lon_t, lat_t : `~astropy.units.Quantity`, scalar
        Geographic longitude/latitude of transmitter [deg]
    lon_r, lat_r : `~astropy.units.Quantity`, scalar
        Geographic longitude/latitude of receiver [deg]
    h_tg, h_rg : `~astropy.units.Quantity`
        Transmitter/receiver height over ground [m]
    hprof_step : `~astropy.units.Quantity`, scalar
        Distance resolution of height profile along path [m]
    timepercent : `~astropy.units.Quantity`
        Time percentage [%] (maximal 50%)
    G_t, G_r  : `~astropy.units.Quantity`, optional
        Antenna gain (transmitter, receiver) in the direction of the
        horizon(!) along the great-circle interference path [dBi]
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
    zone_t, zone_r : `~numpy.ndarray` of int (aka CLUTTER enum), optional
        Clutter type for transmitter/receiver terminal.
        (default: CLUTTER.UNKNOWN)
    polarization : `~numpy.ndarray` of int, optional
        Polarization (default: 0)
        Allowed values are: 0 - horizontal, 1 - vertical
    version : `~numpy.ndarray` of int, optional
        ITU-R Rec. P.452 version. Allowed values are: 14, 16
    delta_N : `~astropy.units.Quantity`, scalar, optional
        Average radio-refractive index lapse-rate through the lowest 1 km of
        the atmosphere [N-units/km = 1/km]
        (default: query `~pycraf.pathprof.deltaN_N0_from_map`)
    N_0 : `~astropy.units.Quantity`, scalar, optional
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
    generic_heights : bool
        If `generic_heights` is set to True, heights will be set to zero.
        This can be useful for generic (aka flat-Earth) computations.
        The option is only meaningful, if the hprof_xxx parameters are set
        to `None` (which means automatic querying of the profiles).
        (Default: False)

    Returns
    -------
    results : dict
        Results of the path attenuation calculation. Each entry
        in the dictionary is a nD `~astropy.units.Quantity` containing
        the associated values for the path.
        The following entries are contained:

        - `L_b0p` - Free-space loss including focussing effects
          (for p% of time) [dB]

        - `L_bd` - Basic transmission loss associated with diffraction
            not exceeded for p% time [dB]; L_bd = L_b0p + L_dp

        - `L_bs` - Tropospheric scatter loss [dB]

        - `L_ba` - Ducting/layer reflection loss [dB]

        - `L_b` - Complete path propagation loss [dB]

        - `L_b_corr` - As L_b but with clutter correction [dB]

        - `eps_pt` - Elevation angle of paths w.r.t. Tx [deg]

        - `eps_pr` - Elevation angle of paths w.r.t. Rx [deg]

        - `d_lt` - Distance to horizon w.r.t. Tx [km]

        - `d_lr` - Distance to horizon w.r.t. Rx [km]

        - `path_type` - Path type (0 - LoS, 1 - Trans-horizon)

    Examples
    --------

    A typical usage would be::

        import numpy as np
        import matplotlib.pyplot as plt
        from pycraf import pathprof
        from astropy import units as u

        frequency = np.logspace(-1, 2, 200) * u.GHz
        temperature = 290. * u.K
        pressure = 980 * u.hPa
        lon_t, lat_t = 6.8836 * u.deg, 50.525 * u.deg
        lon_r, lat_r = 7.3334 * u.deg, 50.635 * u.deg
        h_tg, h_rg = 20 * u.m, 30 * u.m
        hprof_step = 100 * u.m
        time_percent = np.logspace(-3, np.log10(50), 100) * u.percent
        zone_t, zone_r = pathprof.CLUTTER.URBAN, pathprof.CLUTTER.SUBURBAN

        # as frequency and time_percent are arrays, we need to add
        # new axes to allow proper broadcasting
        results = pathprof.losses_complete(
            frequency[:, np.newaxis],
            temperature,
            pressure,
            lon_t, lat_t,
            lon_r, lat_r,
            h_tg, h_rg,
            hprof_step,
            time_percent[np.newaxis],
            zone_t=zone_t, zone_r=zone_r,
            )

        # 2D plot of L_b vs frequency and time_percent
        # (proper axes labels and units omitted!)
        plt.imshow(results['L_b'].value)
        plt.show()

    Notes
    -----
    - It is extremely important how the broadcasting axes are chosen! There
      are six entities - `freq`, `h_tg`, `h_rg`, `version`, `zone_t`, `zone_r`
      - that have influence on the propagation path geometry. In the
      broadcasted arrays, the associated axes should vary as slow as
      possible. The internal Cython routine will trigger a re-computation of
      the path geometry if one of these parameters changes. Therefore, if the
      axes for `frequency` and `time_percent` would have been chosen in the
      opposite manner, the function would run about an order of magnitude
      slower!
    - The diffraction-loss algorithm was changed between ITU-R P.452
      version 14 and 15. The former used a Deygout method, the new one
      is based on a Bullington calculation with correction terms.
    - In future versions, more entries may be added to the results
      dictionary.

    '''

    res = cyprop.losses_complete_cython(
        freq,
        temperature,
        pressure,
        lon_t, lat_t,
        lon_r, lat_r,
        h_tg, h_rg,
        hprof_step,
        timepercent,
        G_t=G_t,
        G_r=G_r,
        omega=omega,
        d_tm=d_tm,
        d_lm=d_lm,
        d_ct=d_ct,
        d_cr=d_cr,
        zone_t=zone_t, zone_r=zone_r,
        polarization=polarization,
        version=version,
        delta_N=delta_N, N0=N0,
        hprof_dists=hprof_dists,
        hprof_heights=hprof_heights,
        hprof_bearing=hprof_bearing,
        hprof_backbearing=hprof_backbearing,
        generic_heights=generic_heights,
        )
    return {
        'L_b0p': res[0] * cnv.dB,
        'L_bd': res[1] * cnv.dB,
        'L_bs': res[2] * cnv.dB,
        'L_ba': res[3] * cnv.dB,
        'L_b': res[4] * cnv.dB,
        'L_b_corr': res[5] * cnv.dB,
        'eps_pt': res[6] * apu.deg,
        'eps_pr': res[7] * apu.deg,
        'd_lt': res[8] * apu.km,
        'd_lr': res[9] * apu.km,
        'path_type': res[10],
        }


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
