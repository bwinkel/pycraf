#!python
# -*- coding: utf-8 -*-
# cython: language_level=3
# cython: cdivision=True, boundscheck=False, wraparound=False
# cython: embedsignature=True

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

cimport cython
from cython.parallel import prange, parallel
from libc.stdlib cimport abort, malloc, free
cimport numpy as np
cimport openmp
from libc.math cimport (
    exp, log, log10, sqrt, fabs, M_PI, floor, pow as cpower,
    sin, cos, tan, asin, acos, atan, atan2, tanh
    )
from .p838 import _rain_coefficients_p838
import numpy as np
from astropy import units as apu
from . import heightprofile
from . import srtm
from . import geodesics
from . import cygeodesics
from . import helper
from .. import conversions as cnv
from .. import utils

np.import_array()


# Implementation of P.2001-3 (based on a copy-paste version of P.452.pyx)
# - clutter is not part of the recommendation, but we provide the P.452
#   procedures, for convenience


__all__ = [
    'CLUTTER', 'CLUTTER_NAMES', 'CLUTTER_DATA',
    'PARAMETERS',
    'set_num_threads',
    ]


cdef double NAN = np.nan
cdef double DEG2RAD = M_PI / 180
cdef double RAD2DEG = 180 / M_PI
cdef double M_PI2 = 2 * M_PI
cdef double M_PI_2 = 0.5 * M_PI
cdef double M_PI_4 = 0.25 * M_PI


cdef double c_light = 2.998e8  # m/s, Speed of propagation
cdef double R_E = 6371.  # km, Average Earth radius
cdef double eps_r_land = 22.  # Relative permittivity for land
cdef double eps_r_sea = 80.  # Relative permittivity for sea
cdef double sigma_land = 0.003  # S/m, Conductivity for land
cdef double sigma_sea = 5.  # S/m, Conductivity for sea


# cpdef enum CLUTTER:
#     UNKNOWN = -1
#     SPARSE = 0
#     VILLAGE = 1
#     DECIDIOUS_TREES = 2
#     CONIFEROUS_TREES = 3
#     TROPICAL_FOREST = 4
#     SUBURBAN = 5
#     DENSE_SUBURBAN = 6
#     URBAN = 7
#     DENSE_URBAN = 8
#     HIGH_URBAN = 9
#     INDUSTRIAL_ZONE = 10

# CLUTTER_NAMES = [
#     'UNKNOWN',
#     'SPARSE',
#     'VILLAGE',
#     'DECIDIOUS_TREES',
#     'CONIFEROUS_TREES',
#     'TROPICAL_FOREST',
#     'SUBURBAN',
#     'DENSE_SUBURBAN',
#     'URBAN',
#     'DENSE_URBAN',
#     'HIGH_URBAN',
#     'INDUSTRIAL_ZONE',
#     ]

# CLUTTER_DATA = np.array(
#     [
#         [4., 0.1],
#         [5., 0.07],
#         [15., 0.05],
#         [20., 0.05],
#         [20., 0.03],
#         [9., 0.025],
#         [12., 0.02],
#         [20., 0.02],
#         [25., 0.02],
#         [35., 0.02],
#         [20., 0.05],
#     ],
#     dtype=np.float64)

# cdef double[:, ::1] CLUTTER_DATA_V = CLUTTER_DATA

RAIN_PROB_DATA_IDX = helper._rain_probs['index']
RAIN_PROB_DATA_REL_HEIGHT = helper._rain_probs['rel_height']
RAIN_PROB_DATA_PROB = helper._rain_probs['prob']

cdef int[:] RAIN_PROB_DATA_IDX_V = RAIN_PROB_DATA_IDX
cdef double[:] RAIN_PROB_DATA_REL_HEIGHT_V = RAIN_PROB_DATA_REL_HEIGHT
cdef double[:] RAIN_PROB_DATA_PROB_V = RAIN_PROB_DATA_PROB


PARAMETERS_BASIC = [
    # version is fixed
    ('version', '12d', '(P.2001 version; 3)', cnv.dimless),

    # input parameters
    ('freq', '12.6f', 'GHz', apu.GHz),
    ('wavelen', '12.6f', 'm', apu.m),
    ('polarization', '12d', '(0 - horizontal, 1 - vertical)', cnv.dimless),
    ('lon_t', '12.6f', 'deg', apu.deg),  # aka phi_te
    ('lat_t', '12.6f', 'deg', apu.deg),  # aka phi_tn
    ('lon_r', '12.6f', 'deg', apu.deg),  # aka phi_re
    ('lat_r', '12.6f', 'deg', apu.deg),  # aka phi_rn
    ('h_tg', '12.6f', 'm', apu.m),  # height above ground
    ('h_rg', '12.6f', 'm', apu.m),
    # Percentage of average year for which the predicted basic transmission
    # loss is not exceeded
    ('T_pc', '12.6f', 'percent', apu.percent),
    # Percentage of average year for which predicted basic transmission loss
    # is not exceeded, limited to range 0.00001% ≤ p ≤ 99.99999%
    ('p', '12.6f', 'percent', apu.percent),  # aka time_percent from P.452
    ('q', '12.6f', 'percent', apu.percent),  # 100% - p
    ('G_t', '12.6f', 'dBi', cnv.dBi),  # Tx gain
    ('G_r', '12.6f', 'dBi', cnv.dBi),  # Rx gain

    # principle parameters
    ('a_e', '12.6f', 'km', apu.km),  # Median effective Earth radius
    # Gaseous attenuation, and the water vapour attenuations with and without
    # rain, for a surface path
    ('A_gsur', '12.6f', 'dB/km', cnv.dB / apu.km),
    ('A_wrsur', '12.6f', 'dB/km', cnv.dB / apu.km),
    ('A_wsur', '12.6f', 'dB/km', cnv.dB / apu.km),
    ('a_p', '12.6f', 'km', apu.km),  # Effective Earth radius exceeded p% time
    ('c_p', '12.6f', '1/km', 1 / apu.km),  # Effective Earth curvature
    ('distance', '12.6f', 'km', apu.km),
    # Terminal to horizon distances
    ('d_lt', '12.6f', 'km', apu.km),
    ('d_lr', '12.6f', 'km', apu.km),
    # Terminal to troposcatter common volume distances
    ('d_tcv', '12.6f', 'km', apu.km),
    ('d_rcv', '12.6f', 'km', apu.km),
    ('h_cv', '12.6f', 'm', apu.m),  # Height of troposcatter common volume
    ('h_hi', '12.6f', 'm', apu.m),  # Higher antenna height
    ('h_lo', '12.6f', 'm', apu.m),  # Lower antenna height
    ('eps_p', '12.6f', 'mrad', apu.mrad)  # positive value of path inclination
    ('h_m', '12.6f', 'm', apu.m),  # Path roughness parameter
    ('h_mid', '12.6f', 'm', apu.m),  # Ground height at mid point of path
    # Effective transmitter, receiver, heights above smooth surface for
    # anomalous (ducting/layer-reflection) model
    ('h_tea', '12.6f', 'm', apu.m),
    ('h_rea', '12.6f', 'm', apu.m),
    # Effective transmitter, receiver, heights above smooth surface for
    # diffraction model
    ('h_tep', '12.6f', 'm', apu.m),
    ('h_rep', '12.6f', 'm', apu.m),
    # Transmitter, receiver, height above mean sea level
    ('h_ts', '12.6f', 'm', apu.m),
    ('h_rs', '12.6f', 'm', apu.m),
    # Profile indices of transmitter, receiver, horizons
    ('i_lt', '12d', '', cnv.dimless),
    ('i_lr', '12d', '', cnv.dimless),

    # Free-space basic transmission loss for the path length and frequency
    ('L_bfs', '12.6f', 'dBi', cnv.dB),
    # Basic transmission loss associated with sub-model 1, diffraction,
    # clear-air and precipitation fading
    ('L_bm1', '12.6f', 'dBi', cnv.dB),
    # Basic transmission loss associated with sub-model 2, anomalous
    # propagation
    ('L_bm2', '12.6f', 'dBi', cnv.dB),
    # Basic transmission loss associated with sub-model 3, troposcatter
    # propagation and precipitation fading
    ('L_bm3', '12.6f', 'dBi', cnv.dB),
    # Basic transmission loss associated with sub-model 4, sporadic-E
    # propagation
    ('L_bm4', '12.6f', 'dBi', cnv.dB),
    # Diffraction loss not exceeded p% time
    ('L_d', '12.6f', 'dBi', cnv.dB),

    # Median value of average refractivity gradient in the lowest 1 km of the
    # atmosphere. Numerically equal to Delta_N as defined in ITU-R P.452 but
    # with opposite sign
    ('N_d1km50', '12.6f', 'dimless', cnv.dimless),
    # Average refractivity gradient in the lowest 1 km of the atmosphere
    # exceeded for p% of an average year. Normally negative but can be zero
    # or positive
    ('N_d1kmp', '12.6f', 'dimless', cnv.dimless),
    # Refractivity gradient in the lowest 65 m of the atmosphere exceeded for
    # 1% of an average year
    ('N_d65m1', '12.6f', 'dimless', cnv.dimless),

    # Troposcatter common volume longitude, latitude
    ('lon_cv', '12.6f', 'deg', apu.deg),  # longitude, aka phi_cve
    ('lat_cv', '12.6f', 'deg', apu.deg),  # latitude, aka phi_cvn
    # Longitude, latitude, of mid-point of path segment from transmitter to
    # the troposcatter common volume
    ('lon_tcv', '12.6f', 'deg', apu.deg),  # longitude, aka phi_tcve
    ('lat_tcv', '12.6f', 'deg', apu.deg),  # latitude, aka phi_tcvn
    # Longitude, latitude, of mid-point of path segment from receiver to the
    # troposcatter common volume
    ('lon_rcv', '12.6f', 'deg', apu.deg),  # longitude, aka phi_rcve
    ('lat_rcv', '12.6f', 'deg', apu.deg),  # latitude, aka phi_rcvn
    # Path mid-point longitude, latitude
    ('lon_m', '12.6f', 'deg', apu.deg),  # longitude, aka phi_me
    ('lat_m', '12.6f', 'deg', apu.deg),  # latitude, aka phi_mn

    # Angle subtended by d km at centre of spherical Earth
    ('theta_e', '12.6f', 'rad', apu.rad),
    # Horizon elevation angles relative to the local horizontal as viewed from
    # transmitter and receiver
    ('theta_t', '12.6f', 'mrad', apu.mrad),
    ('theta_r', '12.6f', 'mrad', apu.mrad),
    # Horizon elevation angles relative to the local horizontal limited to be
    # positive (not less than zero)
    ('theta_tpos', '12.6f', 'mrad', apu.mrad),
    ('theta_rpos', '12.6f', 'mrad', apu.mrad),

    # Sea-level specific attenuation due to oxygen
    ('gamma_0', '12.6f', 'dB/km', cnv.dB / apu.km),
    # Fraction of the path over sea
    ('omega', '12.6f', 'percent', apu.percent),
    # Surface water vapour density at mid point, tx, rx
    ('rho_m', '12.6f', 'g/m3', apu.g / apu.m ** 3),
    ('rho_t', '12.6f', 'g/m3', apu.g / apu.m ** 3),
    ('rho_r', '12.6f', 'g/m3', apu.g / apu.m ** 3),

    ]


cdef struct ppstruct:
    int version  # P.2001 version (3)
    double freq  # GHz
    double wavelen  # m
    int polarization  # 0 - horizontal, 1 - vertical
    double lon_t  # deg
    double lat_t  # deg
    double lon_r  # deg
    double lat_r  # deg
    double h_tg  # m
    double h_rg  # m
    double T_pc  # %
    double p  # %
    double q  # %
    double G_t  # dBi
    double G_r  # dBi
    double a_e  # km
    double A_gsur  # dB / km
    double A_osur  # dB / km
    double A_wsur  # dB / km
    double A_wrsur  # dB / km
    double a_p  # km
    double c_p  # 1 / km
    double distance  # km
    double d_lt  # km
    double d_lr  # km
    double d_tcv  # km
    double d_rcv  # km
    double h_cv  # m
    double h_hi  # m
    double h_lo  # m
    double h_m  # m
    double h_mid  # m
    double h_tea  # m
    double h_rea  # m
    double h_tep  # m
    double h_rep  # m
    double h_ts  # m
    double h_rs  # m
    double eps_p  # mrad
    int i_lt  # 1
    int i_lr  # 1
    double L_bfs  # dBi
    double L_bm1  # dBi
    double L_bm2  # dBi
    double L_bm3  # dBi
    double L_bm4  # dBi
    double L_d  # dBi
    double N_d1km50  # 1
    double N_d1kmp  # 1
    double N_d65m1  # 1
    double lon_cv  # deg, aka phi_cve
    double lat_cv  # deg, aka phi_cve
    double lon_tcv  # deg, aka phi_tcve
    double lat_tcv  # deg, aka phi_tcvn
    double lon_rcv  # deg, aka phi_rcve
    double lat_rcv  # deg, aka phi_rcvn
    double lon_m  # deg, aka phi_me
    double lat_m  # deg, aka phi_mn
    double theta_e  # rad
    double theta_t  # mrad
    double theta_r  # mrad
    double theta_tpos  # mrad
    double theta_rpos  # mrad
    double gamma_0  # dB / km
    double gamma_w  # dB / km
    double omega  # %
    double rho_mid  # g / m3
    double rho_t  # g / m3
    double rho_r  # g / m3

    # ##################
    # derived quantities
    # ##################

    # path geometry
    int path_type  # 0 - LOS, 1 - transhoriz
    double theta_tim  # mrad
    double theta_rim  # mrad
    double theta_tr  # mrad

    # diffraction
    double nu_max  # m ???
    int i_m  # index for nu_max
    double d_bp  # distance to bullington point (BP)
    doulbe h_bp  # height of BP
    doulbe h_eff  # eff. height of BP?
    double S_tim
    double S_rim
    double S_tr

    # diffraction for zero-terrain heights
    double nu_max0  # m ???
    int i_m0  # index for nu_max
    double d_bp0  # distance to bullington point (BP)
    doulbe h_bp0  # height of BP
    doulbe h_eff0  # eff. height of BP?
    double S_tim0
    double S_rim0
    double S_tr0

    double h_stip  # m
    double h_srip  # m
    double lon_25p  # deg, longitude at 25% of distance
    double lat_25p  # deg, latitude at 25% of distance
    double lon_75p  # deg, longitude at 75% of distance
    double lat_75p  # deg, latitude at 75% of distance

    # freqs for sporadic-E basic transmission loss not exceeded for p%
    double f_oes_m  # MHz, mid-point
    double f_oes_2h  # MHz, lower of 25% and 75% distance

    # P.2001 is special in the sense, that multiple time_percents
    # ought to be used for proper MC treatment;
    double p_sporadic

    double d_fs  # km, helper pythagorean distance between Tx and Rx

    double Q_0ca  # notional zero-fade annual percentage time

    # double P_r6_m  # Pr6 value for mid-point
    # double M_T_m  # M_T value for mid-point
    # double beta_rain_m  # beta_rain value for mid-point
    # double h0_m  # mean zero-degree isotherm rain height for mid-point

    double[50] G_m  # helper array for rain fade
    double[50] P_m  # helper array for rain fade
    double M  # max index for above helper arrays

def set_num_threads(int nthreads):
    '''
    Change maximum number of threads to use.

    Parameters
    ----------
    nthreads - int
        Number of threads to use.

    Notes
    -----
    - This can also be controlled by setting the environment variable
      `OMP_NUM_THREADS`.
    '''

    openmp.omp_set_num_threads(nthreads)


cdef inline double f_max(double a, double b) nogil:

    return a if a >= b else b

cdef inline double f_min(double a, double b) nogil:

    return a if a <= b else b

cdef inline int i_max(int a, int b) nogil:

    return a if a >= b else b

cdef inline int i_min(int a, int b) nogil:

    return a if a <= b else b

cdef inline double _J_edgeknife(double nu) nogil:

    if nu <= -0.78:
        return 0.
    else:
        return (
            6.9 + 20 * log10(
                sqrt((nu - 0.1) ** 2 + 1) + nu - 0.1
                )
            )


cdef class _PathProp(object):

    cdef:
        readonly ppstruct _pp

    def __init__(
            self,
            double freq,
            double lon_t, double lat_t,
            double lon_r, double lat_r,
            double h_tg, double h_rg,
            double hprof_step,
            double time_percent,  # T_pc to be precise
            double omega,
            d_tm, d_lm,
            d_ct, d_cr,
            int zone_t, int zone_r,
            int polarization,
            # override if you don't want builtin method:
            N_d1km50, N_d1kmp, N_d65m1,
            # override if you don't want builtin method:
            hprof_dists, hprof_heights,
            hprof_bearing, hprof_backbearing,
            # set terrain heights to zero if desired
            # (only if hprof_xxx set to None aka automatic)
            bint generic_heights=False,
            # P.2001 is special in the sense, that multiple time_percents
            # ought to be used for proper MC treatment;
            # below, these are provided - if set to NaN, use the
            # "time_percent" as default, however, this should be avoided!
            # (is OK for single samples, i.e., when no MC is performed)
            double p_sporadic=np.nan,
            ):

        assert time_percent >= 0. and time_percent <= 100.
        assert version == 14 or version == 16

        assert zone_t >= -1 and zone_t <= 11
        assert zone_r >= -1 and zone_r <= 11

        assert (N_d1km50 is None) == (N_d1kmp is None) == (N_d65m1 is None), (
            'N_d1km50, N_d1kmp, N_d65m1 must all be None or all be provided'
            )

        assert (
            (hprof_dists is None) == (hprof_heights is None) ==
            (hprof_bearing is None) == (hprof_backbearing is None)
            ), (
                'hprof_dists, hprof_heights, bearing, and back_bearing '
                'must all be None or all be provided'
                )

        self._pp.version = version
        self._pp.freq = freq
        self._pp.wavelen = c_light / freq
        # self._pp.temperature = temperature
        # self._pp.pressure = pressure
        self._pp.lon_t = lon_t
        self._pp.lat_t = lat_t
        self._pp.lon_r = lon_r
        self._pp.lat_r = lat_r
        self._pp.zone_t = zone_t
        self._pp.zone_r = zone_r
        self._pp.h_tg_in = h_tg
        self._pp.h_rg_in = h_rg
        if zone_t == CLUTTER.UNKNOWN:
            self._pp.h_tg = h_tg
        else:
            self._pp.h_tg = f_max(CLUTTER_DATA_V[zone_t, 0], h_tg)
        if zone_r == CLUTTER.UNKNOWN:
            self._pp.h_rg = h_rg
        else:
            self._pp.h_rg = f_max(CLUTTER_DATA_V[zone_r, 0], h_rg)

        self._pp.hprof_step = hprof_step
        self._pp.T_pc = time_percent
        self._pp.p = time_percent + 0.00001 * (50 - time_percent) / 50
        self._pp.q = 100 - self._pp.p
        self._pp.polarization = polarization

        if hprof_dists is None:
            (
                lons,
                lats,
                distance,
                distances,
                heights,
                bearing,
                back_bearing,
                back_bearings,
                ) = heightprofile._srtm_height_profile(
                    lon_t, lat_t,
                    lon_r, lat_r,
                    hprof_step,
                    generic_heights=generic_heights
                    )
        else:
            distances = hprof_dists.astype(np.float64, order='C', copy=False)
            heights = hprof_heights.astype(np.float64, order='C', copy=False)
            hsize = distances.size
            distance = distances[hsize - 1]
            bearing = hprof_bearing
            back_bearing = hprof_backbearing

        if len(distances) < 5:
            raise ValueError('Height profile must have at least 5 steps.')

        zheights = np.zeros_like(heights)

        self._pp.distance = distance
        self._pp.bearing = bearing
        self._pp.back_bearing = back_bearing
        self._pp.alpha_tr = bearing
        self._pp.alpha_rt = back_bearing

        # if d_tm is None:
        #     d_tm = distance
        # if d_lm is None:
        #     d_lm = distance
        # if d_ct is None:
        #     d_ct = 50000.
        # if d_cr is None:
        #     d_cr = 50000.

        # TODO: add functionality to produce the following
        # five parameters programmatically (using some kind of Geo-Data)
        self._pp.omega = omega
        # self._pp.d_tm = d_tm
        # self._pp.d_lm = d_lm
        # self._pp.d_ct = d_ct
        # self._pp.d_cr = d_cr

        hsize = distances.size
        mid_idx = hsize // 2
        # TODO: properly treat even/odd mid_idx
        self._pp.h_mid = heights[mid_idx]


        # determine mid-point, as well as d_25 and d_75 lon/lats
        # if hprof_dists is None:
        #     self._pp.lon_m = lons[mid_idx]
        #     self._pp.lat_m = lats[mid_idx]
        # else:

        cdef double dist, bearing1_rad, bearing2_rad
        dist, bearing1_rad, bearing2_rad = cygeodesics._inverse(
            lon_t * DEG2RAD, lat_t * DEG2RAD,
            lon_r * DEG2RAD, lat_r * DEG2RAD,
            1.e-12, 50,
            )
        self._pp.lon_m, self._pp.lat_m, bearing2_rad = _direct(
            lon_t * DEG2RAD, lat_t * DEG2RAD,
            bearing1_rad, 0.5 * dist,
            1.e-12, 50, 1
            )
        self._pp.lon_25p, self._pp.lat_25p, bearing2_rad = _direct(
            lon_t * DEG2RAD, lat_t * DEG2RAD,
            bearing1_rad, 0.25 * dist,
            1.e-12, 50, 1
            )
        self._pp.lon_75p, self._pp.lat_75p, bearing2_rad = _direct(
            lon_t * DEG2RAD, lat_t * DEG2RAD,
            bearing1_rad, 0.75 * dist,
            1.e-12, 50, 1
            )
        self._pp.lon_m *= RAD2DEG
        self._pp.lat_m *= RAD2DEG
        self._pp.lon_25p *= RAD2DEG
        self._pp.lat_25p *= RAD2DEG
        self._pp.lon_75p *= RAD2DEG
        self._pp.lat_75p *= RAD2DEG


        # TODO: cythonize _radiomet_data_for_pathcenter
        if delta_N is None:
            _dn_med, _dn_sup, _dn_sub, _dn_dz = helper._DN_P2001_from_map(
                self._pp.lon_mid, self._pp.lat_mid
                )

        self._pp.N_d1km50 = -_dn_med
        if p < 50:
            self._pp.N_d1kmp = self._pp.N_d1km50 + _dn_sup * log10(0.02 * p)
        else:
            self._pp.N_d1kmp = self._pp.N_d1km50 - _dn_sub * log10(0.02 * p)
        self._pp.N_d65m1 = _dn_dz

        # beta0 = _beta_from_DN_N0(
        #     self._pp.lat_mid,
        #     self._pp.delta_N, self._pp.N0,
        #     self._pp.d_tm, self._pp.d_lm
        #     )

        # self._pp.beta0 = beta0

        # read surface water vapour density for atmospheric loss calculations
        # need this for three different positions, Tx, Rx, and mid-point
        self._pp.rho_mid = helper._surfwv_50_interpolator(
            self._pp.lon_mid % 360, self._pp.lat_mid
            )
        self._pp.rho_t = helper._surfwv_50_interpolator(
            self._pp.lon_t % 360, self._pp.lat_t
            )
        self._pp.rho_r = helper._surfwv_50_interpolator(
            self._pp.lon_r % 360, self._pp.lat_r
            )

        # read Sporadic E values, for a number of positions (TODO)
        if np.isnan(p_sporadic):
            p_sporadic = time_percent

        self._pp.f_oes_m = helper._sporadic_E_P2001_from_map(
            self._pp.lon_m, self._pp.lat_m, p_sporadic
            )

        cdef double f_oes_25p, f_oes_75p
        f_oes_25p = helper._sporadic_E_P2001_from_map(
            self._pp.lon_25p, self._pp.lat_25p, p_sporadic
            )
        f_oes_75p = helper._sporadic_E_P2001_from_map(
            self._pp.lon_75p, self._pp.lat_75p, p_sporadic
            )
        self._pp.f_oes_2h = f_min(f_oes_25p, f_oes_75p)

        _process_path(
            &self._pp,
            # lons,
            # lats,
            distances,
            heights,
            zheights,
            )


# cdef double _beta_from_DN_N0(
#         double lat_mid, double DN, double N0, double d_tm, double d_lm
#         ) nogil:

#     cdef:
#         double tau, a, b, mu1, log_mu1, mu4, beta0

#     tau = 1. - exp(-4.12e-4 * cpower(d_lm, 2.41))
#     lat_mid = fabs(lat_mid)

#     a = cpower(10, -d_tm / (16. - 6.6 * tau))
#     b = cpower(10, -5 * (0.496 + 0.354 * tau))
#     mu1 = cpower(a + b, 0.2)
#     if mu1 > 1.:
#         mu1 = 1.
#     log_mu1 = log10(mu1)

#     if lat_mid <= 70.:
#         mu4 = cpower(10, (-0.935 + 0.0176 * lat_mid) * log_mu1)
#     else:
#         mu4 = cpower(10, 0.3 * log_mu1)

#     if lat_mid <= 70.:
#         beta_0 = cpower(10, -0.015 * lat_mid + 1.67) * mu1 * mu4
#     else:
#         beta_0 = 4.17 * mu1 * mu4

#     return beta_0


# def beta_from_DN_N0(
#         double lat_mid, double DN, double N0, double d_tm, double d_lm
#         ):
#     '''
#     Calculate radiometeorological data, beta0.

#     Parameters
#     ----------
#     lat - path center coordinates [deg]
#     delta_N - average radio-refractive index lapse-rate through the
#             lowest 1 km of the atmosphere [N-units/km]
#     N_0 - sea-level surface refractivity [N-units]
#     d_tm - longest continuous land (inland + coastal) section of the
#         great-circle path [km]
#     d_lm - longest continuous inland section of the great-circle path [km]

#     Returns
#     -------
#     beta_0 - the time percentage for which refractive index lapse-rates
#         exceeding 100 N-units/km can be expected in the first 100 m
#         of the lower atmosphere [%]

#     Notes
#     -----
#     - ΔN and N_0 can be derived from digitized maps (shipped with P.452).
#     - Radio-climaticzones can be queried from ITU Digitized World Map (IDWM).
#       For many applications, it is probably the case, that only inland
#       zones are present along the path of length d.
#       In this case, set d_tm = d_lm = d.
#     '''

#     return _beta_from_DN_N0(
#         lat_mid, DN, N0, d_tm, d_lm
#         )


cdef void _process_path(
        ppstruct *pp,
        # double[::1] lons_view,
        # double[::1] lats_view,
        double[::1] distances_view,
        double[::1] heights_view,
        double[::1] zheights_view,
        # double bearing,
        # double back_bearing,
        # double distance,
        ) nogil:

    # TODO: write down, which entries "pp" MUST have already

    cdef:

        int diff_edge_idx
        int hsize = distances_view.shape[0]

    # import time
    # _time = time.time()

    pp.h0 = heights_view[0]
    pp.hn = heights_view[hsize - 1]

    pp.h_ts = pp.h0 + pp.h_tg
    pp.h_rs = pp.hn + pp.h_rg

    pp.h_lo = f_min(pp.h_ts, pp.h_rs)
    pp.h_hi = f_max(pp.h_ts, pp.h_rs)

    # path inclination
    pp.eps_p = (pp.h_hi - pp.h_lo) / pp.distance

    # # smooth-earth height profile
    # pp.h_st, pp.h_sr = _smooth_earth_heights(
    #     pp.distance, distances_view, heights_view,
    #     )

    # print('_smooth_earth_heights', time.time() - _time)
    # _time = time.time()

    # # effective antenna heights for diffraction model
    # pp.h_std, pp.h_srd = _effective_antenna_heights(
    #     pp.distance,
    #     distances_view, heights_view,
    #     pp.h_ts, pp.h_rs,
    #     pp.h_st, pp.h_sr
    #     )

    # print('_effective_antenna_heights', time.time() - _time)
    # _time = time.time()

    # # parameters for ducting/layer-reflection model
    # # (use these only for ducting or also for smooth-earth?)
    # pp.h_st = min(pp.h_st, pp.h0)
    # pp.h_sr = min(pp.h_sr, pp.hn)

    # pp.duct_slope = (pp.h_sr - pp.h_st) / pp.distance

    # pp.h_te = pp.h_tg + pp.h0 - pp.h_st
    # pp.h_re = pp.h_rg + pp.hn - pp.h_sr

    pp.a_e = R_E * 157. / (157. + pp.N_d1km50)
    pp.c_p = (157. + pp.N_d1kmp) / R_E / 157.
    if pp.c_p > 1e-6:
        pp.a_p = 1 / pp.c_p
    else:
        pp.a_p = 1e6

    pp.theta_e = pp.distance / pp.a_e
    # pp.a_e_b0 = R_E * 3.

    (
        pp.path_type,
        pp.theta_t, pp.theta_r, pp.theta_tim, pp.theta_rim, pp.theta_tr,
        pp.i_lt, pp.i_lr, pp.d_lt, pp.d_lr, pp.nu_max, pp.i_m
        ) = _path_geometry_helper(
        pp.wavelen, pp.a_e_50, pp.distance,
        distances_view, heights_view,
        pp.h_ts, pp.h_rs,
        )

    pp.theta_tpos = f_max(pp.theta_t, 0)
    pp.theta_rpos = f_max(pp.theta_r, 0)

    (
        pp.h_stip, pp.h_srip, pp.h_tea, pp.h_rea, pp.h_m
        ) = _smooth_earth_heights_and_terrain_roughness(
        pp.distance,
        distances_view, heights_view,
        pp.h_ts, pp.h_rs,
        pp.i_lt, pp.i_lr
        )

    (pp.h_tep, pp.h_rep) = _effective_antenna_heights(
        pp.distance,
        distances_view, heights_view,
        pp.h_ts, pp.h_rs,
        pp.h_stip, pp.h_srip,
        )

    (
        pp.d_tcv, pp.d_rcv, pp.h_cv,
        pp.lon_cv, pp.lat_cv, pp.lon_tcv, pp.lat_tcv, pp.lon_rcv, pp.lat_rcv
        ) = _troposcatter_path_segments(
        pp.distance, pp.a_e,
        pp.theta_rpos, pp.theta_tpos,
        pp.theta_e,
        pp.h_ts, pp.h_rs,
        pp.lon_t, pp.lat_t, pp.lon_r, pp.lat_r,
        )

    # TODO: the following function should go into its own routine
    (
        pp.A_osur, pp.A_wsur, pp.A_wrsur,
        pp.gamma_0, pp.gamma_w
        ) = _gaseous_absorption_surface_path(
        pp.distance, pp.h_mid, pp.freq, pp.rho_sur,
        pp.h_ts, pp.h_rs,
        )
    pp.A_gsur = pp.A_osur + pp.A_wsur

    # helper distance for L_bfs
    pp.d_fs = sqrt(pp.distance ** 2 + (1e-3 * (pp.h_ts - pp.h_rs)) ** 2)

    (
        pp.path_type, pp.d_bp, pp.h_bp, pp.h_eff,
        pp.nu_max, pp.i_m,
        pp.S_tim, pp.S_rim, pp.S_tr
        ) = _diffraction_bullington(
        pp.a_e, pp.distance,
        distances_view, heights_view,
        pp.h_ts, pp.h_rs,
        pp.wavelen,
        )

    # now for smooth Earth (zero terrain heights)
    (
        pp.path_type0, pp.d_bp0, pp.h_bp0, pp.h_eff0,
        pp.nu_max0, pp.i_m0,
        pp.S_tim0, pp.S_rim0, pp.S_tr0
        ) = _diffraction_bullington(
        pp.a_e, pp.distance,
        distances_view, zheights_view,
        pp.h_tep, pp.h_rep,
        pp.wavelen,
        )


    # TODO: move the lines below to an appropriate func
    cdef double L_dbka = _J_edgeknife(pp.nu_max)
    cdef double L_dba = L_dbka + (1 - exp(-L_dbka / 6)) * (10 + 0.02 * pp.distance)
    cdef double L_dbks = _J_edgeknife(pp.nu_max0)
    cdef double L_dbs = L_dbks + (1 - exp(-L_dbks / 6)) * (10 + 0.02 * pp.distance)

    # Calculate the notional zero-fade annual percentage time, Q0ca
    pp.Q_0ca = _multi_path_activity(
        pp.freq, pp.lat_m,
        pp.N_d65m1,
        pp.path_type,
        pp.distance,
        pp.eps_p,
        pp.h_lo,
        pp.d_lt, pp.d_lr,
        pp.theta_t, pp.theta_r,
        pp.i_lt, pp.i_lr,
        pp.h_ts, pp.h_rs,
        heights_view,
        )


    # (
    #     pp.path_type_50, pp.d_bp_50, pp.h_bp_50, pp.h_eff_50,
    #     pp.nu_bull_50, pp.nu_bull_idx_50,
    #     pp.S_tim_50, pp.S_rim_50, pp.S_tr_50
    #     ) = _diffraction_helper_v16(
    #     pp.a_e_50, pp.distance,
    #     distances_view, heights_view,
    #     pp.h_ts, pp.h_rs,
    #     pp.wavelen,
    #     )

    # (
    #     pp.path_type_b0, pp.d_bp_b0, pp.h_bp_b0, pp.h_eff_b0,
    #     pp.nu_bull_b0, pp.nu_bull_idx_b0,
    #     pp.S_tim_b0, pp.S_rim_b0, pp.S_tr_b0
    #     ) = _diffraction_helper_v16(
    #     pp.a_e_b0, pp.distance,
    #     distances_view, heights_view,
    #     pp.h_ts, pp.h_rs,
    #     pp.wavelen,
    #     )

    # # similarly, we have to repeat the game with heights set to zero

    # (
    #     pp.path_type_zh_50, pp.d_bp_zh_50, pp.h_bp_zh_50, pp.h_eff_zh_50,
    #     pp.nu_bull_zh_50, pp.nu_bull_idx_zh_50,
    #     pp.S_tim_zh_50, pp.S_rim_zh_50, pp.S_tr_zh_50
    #     ) = _diffraction_helper_v16(
    #     pp.a_e_50, pp.distance,
    #     distances_view, zheights_view,
    #     pp.h_ts - pp.h_std, pp.h_rs - pp.h_srd,
    #     pp.wavelen,
    #     )

    # (
    #     pp.path_type_zh_b0, pp.d_bp_zh_b0, pp.h_bp_zh_b0, pp.h_eff_zh_b0,
    #     pp.nu_bull_zh_b0, pp.nu_bull_idx_zh_b0,
    #     pp.S_tim_zh_b0, pp.S_rim_zh_b0, pp.S_tr_zh_b0
    #     ) = _diffraction_helper_v16(
    #     pp.a_e_b0, pp.distance,
    #     distances_view, zheights_view,
    #     pp.h_ts - pp.h_std, pp.h_rs - pp.h_srd,
    #     pp.wavelen,
    #     )

    # # print('_diffraction_helpers', time.time() - _time)
    # # _time = time.time()

    # # finally, determine remaining path geometry properties
    # # note, this can depend on the bullington point (index) derived in
    # # _diffraction_helper for 50%

    # diff_edge_idx = pp.nu_bull_idx_50

    # (
    #     pp.path_type, pp.theta_t, pp.theta_r, pp.eps_pt, pp.eps_pr,
    #     pp.theta,
    #     pp.d_lt, pp.d_lr, pp.h_m
    #     ) = _path_geometry_helper(
    #     pp.a_e_50, pp.distance,
    #     distances_view, heights_view,
    #     pp.h_ts, pp.h_rs, pp.h_st,
    #     diff_edge_idx, pp.duct_slope,
    #     )

    # print('_path_geometry_helper', time.time() - _time)
    # _time = time.time()

    return


cdef (int, double, double, double, double, double, int, int, double, double, double, int) _path_geometry_helper(
        double wavelen,
        double a_e,
        double distance,
        double[::1] d_v,  # cython view on distance array, aka d_i
        double[::1] h_v,  # cython view on heights array, aka h_i
        double h_ts,
        double h_rs,
        ) nogil:

    cdef:
        int i, dsize
        double d = distance
        int path_type
        double theta_i, nu_i
        double theta_tim = -1.e31, theta_rim = -1.e31
        double theta_t, theta_r, theta_tr
        int i_lt, i_lr
        double d_lt, d_lr
        double nu_max = -1.e31
        int i_m_t, i_m_r, i_m

    dsize = d_v.shape[0]

    # Find the highest elevation angle to an intermediate profile point,
    # relative to the horizontal at the Tx
    for i in range(1, dsize - 1):

        theta_i = (
            (h_v[i] - h_ts) / d_v[i] - 500 * d_v[i] / a_e
            )
        if theta_i > theta_tim:
            theta_tim = theta_i
            i_m_t = i

    theta_tr = (
        (h_rs - h_ts) / d - 500 * d / a_e
        )

    if theta_tim < theta_tr:
        path_type = 0
    else:
        path_type = 1

    if path_type == 1:
        # transhorizon

        # Find the highest elevation angle to an intermediate profile point,
        # relative to the horizontal at the Tx
        for i in range(1, dsize - 1):

            theta_i = (
                (h_v[i] - h_rs) / (d - d_v[i]) - 500 * (d - d_v[i]) / a_e
                )
            if theta_i > theta_rim:
                theta_rim = theta_i
                i_m_r = i

        d_lt = d_v[i_m_t]
        d_lr = d - d_v[i_m_r]
        i_lt = i_m_t
        i_lr = i_m_r
        theta_t = theta_tim
        theta_r = theta_rim

    else:
        # LOS

        # Find the profile point with the highest diffraction parameter
        # as seen from Tx
        for i in range(1, dsize - 1):

            nu_i = (
                h_v[i] +
                500 * d_v[i] * (d - d_v[i]) / a_e -
                h_ts * (d - d_v[i]) + h_rs * d_v[i] / d
                ) * sqrt(0.002 * d / wavelen / d_v[i] / (d - d_v[i]))
            if nu_i > nu_max:
                nu_max = nu_i
                i_m = i

        d_lt = d_v[i_m]
        d_lr = d - d_lt
        i_lt = i_m
        i_lr = i_m

        theta_t = theta_tr
        theta_r = -theta_tr - 1000 * d / a_e

    return (
        path_type,
        theta_t, theta_r, theta_tim, theta_rim, theta_tr,
        i_lt, i_lr, d_lt, d_lr, nu_max, i_m
        )


cdef (
    double, double, double, double, double
    ) _smooth_earth_heights_and_terrain_roughness(
        double distance,
        double[::1] d_v,
        double[::1] h_v,
        double h_ts,
        double h_rs,
        int i_lt,
        int i_lr,
        ) nogil:

    cdef:
        int i, dsize
        double d = distance, nu_1, nu_2
        double h_stip, h_srip, h_stipa, h_sripa
        double m_ses, h_tea, h_rea, _h, h_m = -1.e31
        double H_i


    dsize = d_v.shape[0]

    nu_1 = 0.
    nu_2 = 0.
    for i in range(1, dsize):

        nu_1 += (d_v[i] - d_v[i - 1]) * (h_v[i] + h_v[i - 1])
        nu_2 += (d_v[i] - d_v[i - 1]) * (
            h_v[i] * (2 * d_v[i] + d_v[i - 1]) +
            h_v[i - 1] * (d_v[i] + 2 * d_v[i - 1])
            )

    h_stip = (2 * nu_1 * d - nu_2) / d ** 2
    h_srip = (nu_2 - nu_1 * d) / d ** 2

    h_stipa = f_min(h_stip, h_v[0])
    h_sripa = f_min(h_srip, h_v[dsize - 1])

    m_ses = (h_sripa - h_stipa) / d
    h_tea = h_ts - h_stipa  # effective height of Tx above smooth Earth surf
    h_rea = h_rs - h_sripa  # effective height of Rx above smooth Earth surf

    for i in range(i_lt, i_lr + 1):

        _h = h_v[i] - h_stipa - m_ses * d_v[i]
        if _h > h_m:
            h_m = _h  # terrain roughness

    return (h_stip, h_srip, h_tea, h_rea, h_m)


cdef (double, double) _effective_antenna_heights(
        double distance,
        double[::1] d_v,
        double[::1] h_v,
        double h_ts, double h_rs,
        double h_stip, double h_srip,
        ) nogil:

    cdef:
        int i, dsize
        double d = distance, h0, hn

        double H_i, h_obs = -1.e31, alpha_obt = -1.e31, alpha_obr = -1.e31
        double tmp_alpha_obt, tmp_alpha_obr

        double h_st, h_sr, g_t, g_r
        double h_tep, h_rep

    dsize = d_v.shape[0]
    h0 = h_v[0]
    hn = h_v[dsize - 1]

    for i in range(1, dsize - 1):

        H_i = h_v[i] - (h_ts * (d - d_v[i]) + h_rs * d_v[i]) / d
        tmp_alpha_obt = H_i / d_v[i]
        tmp_alpha_obr = H_i / (d - d_v[i])

        if H_i > h_obs:
            h_obs = H_i

        if tmp_alpha_obt > alpha_obt:
            alpha_obt = tmp_alpha_obt

        if tmp_alpha_obr > alpha_obr:
            alpha_obr = tmp_alpha_obr

    if h_obs < 0.:
        h_st = h_stip
        h_sr = h_srip
    else:
        g_t = alpha_obt / (alpha_obt + alpha_obr)
        g_r = alpha_obr / (alpha_obt + alpha_obr)

        h_st = h_stip - h_obs * g_t
        h_sr = h_srip - h_obs * g_r

    if h_st > h0:
        h_st = h0

    if h_sr > hn:
        h_sr = hn

        h_tep = h_ts - h_st
        h_rep = h_rs - h_sr

    return (h_tep, h_rep)


cdef (
        double, double, double,
        double, double, double, double, double, double
        ) _troposcatter_path_segments(
        double distance, double a_e,
        double theta_rpos, double theta_tpos,
        double theta_e,
        double h_ts, double h_rs,
        double lon_t, double lat_t, double lon_r, double lat_r,
        ) nogil:

    cdef:
        double d = distance, d_tcv, d_rcv, h_cv
        double _d, _b1, _b2
        double lon_cv, lat_cv, lon_tcv, lat_tcv, lon_rcv, lat_rcv

    lon_t *= DEG2RAD
    lat_t *= DEG2RAD
    lon_r *= DEG2RAD
    lat_r *= DEG2RAD

    d_tcv = (
        d * tan(0.001 * theta_rpos + 0.5 * theta_e) - 0.001 * (h_ts - h_rs) /
        (
            tan(0.001 * theta_tpos + 0.5 * theta_e) +
            tan(0.001 * theta_rpos + 0.5 * theta_e)
            )
        )
    if d_tcv < 0.:
        d_tcv = 0.
    if d_tcv > d:
        d_tcv = d

    d_rcv = d - d_tcv

    h_cv = (
        h_ts + 1000 * d_tcv * tan(0.001 * theta_tpos) +
        1000 * d_tcv * d_tcv / 2 / a_e
        )

    _d, _b1, _b2 = cygeodesics._inverse(
        lon_t, lat_t, lon_r, lat_r, 1.e-12, 50
        )
    lon_cv, lat_cv, _b2 = _direct(
        lon_t, lat_t, _b1, d_tcv, 1.e-12, 50, 1
        )
    lon_tcv, lat_tcv, _b2 = _direct(
        lon_t, lat_t, _b1, 0.5 * d_tcv, 1.e-12, 50, 1
        )
    lon_rcv, lat_rcv, _b2 = _direct(
        lon_t, lat_t, _b1, d - 0.5 * d_rcv, 1.e-12, 50, 1
        )

    lon_cv *= RAD2DEG
    lat_cv *= RAD2DEG
    lon_tcv *= RAD2DEG
    lat_tcv *= RAD2DEG
    lon_rcv *= RAD2DEG
    lat_rcv *= RAD2DEG

    return (
        d_tcv, d_rcv, h_cv,
        lon_cv, lat_cv, lon_tcv, lat_tcv, lon_rcv, lat_rcv
        )


cdef (
        double, double, double,
        double, double,
        ) _gaseous_absorption_surface_path(
        double distance, double h_mid, double freq, double rho_sur,
        double h_ts, double h_rs,
        ) nogil:

    cdef:

        double h_sur = h_mid, h_rho
        double rho_surr, rho_sea

        double A_gsur, A_osur, A_wsur, A_wrsur
        double gamma_0, gamma_w

    rho_surr = (
        rho_sur + 0.4 + 0.0003 * h_sur
        if h_sur <= 2600 else
        rho_sur + 5 * exp(-h_sur / 1800)
        )

    # the following is only good up to 54 GHz!
    # freq in GHz
    gamma_0 = freq * freq * 1.e-3 * (
        7.2 / (freq * freq + 0.34) +
        0.62 / ((54 - freq) ** 1.16 + 0.83)
        )

    # non-rain conditions
    rho_sea = rho_sur * exp(h_sur / 2000)
    eta = 0.955 + 0.006 * rho_sea
    gamma_w = freq * freq * rho_sea * 1.e-4 * (
        0.046 + 0.0019 * rho_sea +
        3.98 * eta / ((f -22.235) ** 2 + 9.42 * eta ** 2) * (
            1 + ((freq - 22) / (freq + 22)) ** 2
            )
        )
    # rain conditions
    rho_sea = rho_surr * exp(h_sur / 2000)
    eta = 0.955 + 0.006 * rho_sea
    gamma_wr = freq * freq * rho_sea * 1.e-4 * (
        0.046 + 0.0019 * rho_sea +
        3.98 * eta / ((f -22.235) ** 2 + 9.42 * eta ** 2) * (
            1 + ((freq - 22) / (freq + 22)) ** 2
            )
        )

    h_rho = 0.5 * (h_ts + h_rs)

    A_osur = gamma_0 * distance * exp(-h_rho / 5000)
    A_wsur = gamma_w * distance * exp(-h_rho / 2000)
    A_wrsur = gamma_wr * distance * exp(-h_rho / 2000)

    return A_osur, A_wsur, A_wrsur, gamma_0, gamma_w



cdef double _spherical_earth_diffraction_first_term_helper(
        double a_dft,
        double distance
        double freq,
        double eps_r,
        double sigma,
        int polarization,  # 0 - horizontal, 1 - vertical
        double h_te, double h_re,
        ) nogil:

    cdef:
        double third = 0.3333333333333333
        double K, K2, K4, beta

        double X, Y_t, Y_r, F_X, B, G_t, G_r
        double L_dft

    # horizontal pol:
    K = 0.036 * (a_dft * freq) ** -third * (
        (eps_r - 1) ** 2 + (18 * sigma / freq) ** 2
        ) ** -0.25

    if polarization == 1:
        K = K * sqrt(eps_r ** 2 + (18 * sigma / freq) ** 2)

    K2 = K * K
    K4 = K2 * K2
    beta = (1 + 1.6 * K2 + 0.67 * K4) / (1 + 4.5 * K2 + 1.53 * K4)

    X = 21.88 * beta * (freq ** 2 / a_dft) ** third * distance
    Y_t = 0.9575 * beta * (freq ** 2 / a_dft) ** third * h_te
    Y_r = 0.9575 * beta * (freq ** 2 / a_dft) ** third * h_re

    F_X = (
        11 + 10 * log10(X) - 17.6 * X
        if X >= 1.6 else
        -20 * log10(X) - 5.6488 * X ** 1.425
        )

    B = beta * Y_t
    G_t = (
        17.6 * sqrt(B - 1.1) - 5 * log10(B - 1.1) - 8
        if B > 2 else
        20 * log10(B + 0.1 * B ** 3)
        )
    G_t = f_max(G_t, 2 + 20 * log10(K))

    B = beta * Y_r
    G_r = (
        17.6 * sqrt(B - 1.1) - 5 * log10(B - 1.1) - 8
        if B > 2 else
        20 * log10(B + 0.1 * B ** 3)
        )
    G_r = f_max(G_r, 2 + 20 * log10(K))

    L_dft = -F_X - G_g - G_r

    return L_dft


cdef double _spherical_earth_diffraction_first_term(
        double a_dft,
        double distance
        double freq,
        int polarization,  # 0 - horizontal, 1 - vertical
        double h_te, double h_re,
        double eps_r_land, double eps_r_sea,
        double sigma_land, double sigma_sea,
        double omega
        ) nogil:

    cdef:
        double L_dft_land, L_dft_sea, omega, L_dft

    L_dft_land = _spherical_earth_diffraction_first_term_helper(
        a_dft, distance freq, eps_r_land, sigma_land, polarization, h_te, h_re,
        )
    L_dft_sea = _spherical_earth_diffraction_first_term_helper(
        a_dft, distance freq, eps_r_sea, sigma_sea, polarization, h_te, h_re,
        )

    L_dft = omega * L_dft_sea + (1 - omega) * L_dft_land

    return L_dft


cdef double _spherical_earth_diffraction(
        double a_p,
        double distance,
        double h_tep, double h_rep,
        double wavelen,
        int polarization,  # 0 - horizontal, 1 - vertical
        double h_te, double h_re,
        double eps_r_land, double eps_r_sea,
        double sigma_land, double sigma_sea,
        double omega
        ) nogil:

    cdef:
        double d = distance, d_los, d_1, d_2
        double b_sph, h_sph, c_sph, m_sph, h_hreq
        double lam = wavelen, C_e500 = 500. / a_p

        double a_dft, L_dsph, L_dft

    d_los = sqrt(2 * a_p) * (sqrt(0.001 * h_tep) + sqrt(0.001) * h_rep)

    if d >= d_los:
        a_dft = a_p
        L_dsph = _spherical_earth_diffraction_first_term(
            a_dft, distance freq,
            polarization,
            h_te, h_re,
            eps_r_land, eps_r_sea,
            sigma_land, sigma_sea,
            omega
            )  #  == L_dft
    else:
        c_sph = (h_tep - h_rep) / (h_tep + h_rep)
        m_sph = 250 * d * d / a_p / (h_tep + h_rep)

        b_sph = 2 * sqrt((m_sph + 1) / 3 / m_sph) * cos(
            M_PI / 3 +
            1 / 3 * acos(3 * c_sph / 2 * sqrt(3 * m_sph / (m_sph + 1) ** 3))
            )

        d_1 = 0.5 * d * (1 + b_sph)
        d_2 = d - d_1

        h_sph = (
            (h_tep - C_e500 * d_1 ** 2) * d_2 +
            (h_rep - C_e500 * d_2 ** 2) * d_1
            ) / d

        h_req = 17.456 * sqrt(d_1 * d_2 * lam / d)

        if h_sph > h_req:
            L_dsph = 0.
        else:
            a_dft = 500 * (d / (sqrt(h_tep) + sqrt(h_rep))) ** 2  # == a_em
            L_dft = L_dsph = _spherical_earth_diffraction_first_term(
                a_dft, distance freq,
                polarization,
                h_te, h_re,
                eps_r_land, eps_r_sea,
                sigma_land, sigma_sea,
                omega
                )
            if L_dft < 0.:
                L_dsph = 0.
            else:
                L_dsph = (1 - h_sph / h_req) * L_dft

    return L_dsph


cdef (
    int, double, double, double, double, int, double, double, double, double
    ) _diffraction_bullington(
        double a_p,
        double distance,
        double[::1] d_v,
        double[::1] h_v,
        double h_ts, double h_rs,
        double wavelen,
        ) nogil:

    cdef:
        int i, dsize
        double d = distance, lam = wavelen, C_e500 = 500. / a_p
        int path_type

        double slope_i, slope_j, S_tim = -1.e31, S_tr, S_rim = -1.e31

        # nu_i == nu_a; nu_bull = nu_a_max (naming different in P452 vs P2001)
        # for zero terrain heights, nu_i == nu_s; nu_bull = nu_s_max
        int nu_bull_idx
        double nu_bull = -1.e31, nu_i
        double h_eff, h_eff_i
        double h_bp, d_bp
        double x, y  # temporary vars

    dsize = d_v.shape[0]

    for i in range(1, dsize - 1):

        slope_i = (
            h_v[i] + C_e500 * d_v[i] * (d - d_v[i]) - h_ts
            ) / d_v[i]

        if slope_i > S_tim:
            S_tim = slope_i

    S_tr = (h_rs - h_ts) / d

    if S_tim < S_tr:
        path_type = 0
    else:
        path_type = 1

    if path_type == 1:
        # transhorizon
        # find Bullington point, etc.
        for i in range(1, dsize - 1):
            slope_j = (
                h_v[i] + C_e500 * d_v[i] * (d - d_v[i]) - h_rs
                ) / (d - d_v[i])

            if slope_j > S_rim:
                S_rim = slope_j

        d_bp = x = (h_rs - h_ts + S_rim * d) / (S_tim + S_rim)
        y = a_p + h_ts / 1000 + d_bp * (S_tim / 1000 - d / 2 / a_p)
        h_bp = 1000 * (sqrt(x ** 2 + y ** 2) - a_p)

        h_eff = (
            h_ts + S_tim * d_bp -
            (
                h_ts * (d - d_bp) + h_rs * d_bp
                ) / d
            )

        nu_bull = h_eff * sqrt(
            0.002 * d / lam / d_bp / (d - d_bp)
            )  # == nu_b in Eq. 20
        nu_bull_idx = -1  # dummy value

    else:
        # LOS

        # find Bullington point, etc.

        S_rim = NAN

        # diffraction parameter
        for i in range(1, dsize - 1):
            h_eff_i = (
                h_v[i] +
                C_e500 * d_v[i] * (d - d_v[i]) -
                (h_ts * (d - d_v[i]) + h_rs * d_v[i]) / d
                )
            nu_i = h_eff_i * sqrt(
                0.002 * d / lam / d_v[i] / (d - d_v[i])
                )
            if nu_i > nu_bull:
                nu_bull = nu_i
                nu_bull_idx = i
                h_eff = h_eff_i

        d_bp = x = d_v[nu_bull_idx]
        y = a_p + h_ts / 1000 + d_bp * (S_tr / 1000 - d / 2 / a_p)
        h_bp = 1000 * (sqrt(x ** 2 + y ** 2) - a_p)

    return (
        path_type, d_bp, h_bp, h_eff, nu_bull, nu_bull_idx, S_tim, S_rim, S_tr
        )


cdef double _notional_zero_face_time_percentage(
        double K, double freq, double lat_m,
        double d_ca, double eps_ca, double h_ca,
        )

    cdef:

        double Q
        double q_w, C_g

    q_w = (
        K * d_ca ** 3.1 * (1 + eps_ca) ** -1.29 * f ** 0.8 *
        10 ** (-0.00089 * h_ca)
        )  # %

    if abs(lat_m) <= 45:
        C_g = (
            10.5 - 5.6 * log10(1.1 + abs(cos(2 * DEG2RAD * lat_m)) ** 0.7) -
            2.8 * log10(d_ca) + 1.7 * log10(1 + eps_ca)
            )
    else:
        C_g = (
            10.5 - 5.6 * log10(1.1 - abs(cos(2 * DEG2RAD * lat_m)) ** 0.7) -
            2.8 * log10(d_ca) + 1.7 * log10(1 + eps_ca)
            )

    if C_g > 10.8:
        C_g = 10.8

    Q = 10 ** (-0.1 * C_g) * q_w  # %

    return Q


cdef double _multi_path_activity(
        double freq, double lat_m,
        double N_d65m1,
        int path_type,
        double distance,
        double eps_p,
        double h_lo,
        double d_lt, double d_lr,
        double theta_t, double theta_r,
        int i_lt, int i_lr,
        double h_ts, double h_rs,
        double[::1] h_v,
        ) nogil:

    cdef:
        double K
        double Q_0ca, Q_0cat, Q_0car

    K = 10 ** (-4.6 - 0.0027 * N_d65m1)

    if path_type == 0:
        # LoS
        Q_0ca = _notional_zero_face_time_percentage(
            K, freq, lat_m, distance, eps_p, h_lo
            )

    elif path_type == 1:
        # NLoS

        Q_0cat = _notional_zero_face_time_percentage(
            K, freq, lat_m, d_lt, abs(theta_t), f_min(h_ts, h_v[i_lt])
            )
        Q_0car = _notional_zero_face_time_percentage(
            K, freq, lat_m, d_lr, abs(theta_r), f_min(h_rs, h_v[i_lr])
            )

        Q_0ca = f_max(Q_0cat, Q_0car)

    return Q_0ca

cdef inline double _Gamma_melting_layer(double delta_h):

    if delta_h > 0.:
        return 0.
    elif delta_h < -1200:
        return 1.
    else:
        return (
            4. * (1. - exp(delta_h / 70.)) ** 2 /
            (
                1 +
                (1. - exp((-delta_h / 600.) ** 2)) ** 2 * (
                    4. * (1. - exp(delta_h / 70.)) ** 2 - 1.
                    )
                )
            )


cdef double _path_avg_multiplier(double h_T, double h_lo, double h_hi):

    cdef:

        size_t s_lo = size_t(0.01 * (h_T - h_lo))
        size_t s_hi = size_t(0.01 * (h_T - h_hi))
        size_t s_i
        double G, double delta_h, double Q

    if s_lo < 0:
        G = 0.
    elif s_hi >= 12:
        G = 1.
    elif s_lo == s_hi:
        G = _Gamma_melting_layer(0.5 * (h_lo + h_hi) - h_T)
    else:
        G = 0.
        for s_i in range(i_max(s_hi, 0), i_min(s_lo, 12)):
            # note in P.2001 the s index is 1 based!

            if s_hi < s_i and s_i < s_lo:
                delta_h = 100 * (0.5 - (s_i + 1))
                Q = 100. / (h_hi - h_lo)
            elif s_i = s_lo:
                delta_h = 0.5 * (h_lo - h_T - 100 * s_i)
                Q = (h_T - 100 * s_i - h_lo) / (h_hi - h_lo)
            elif s_i = s_hi:
                delta_h = 0.5 * (h_hi - h_T - 100 * (s_i + 1))
                Q = (h_hi - (h_T - 100 * (s_i + 1))) / (h_hi - h_lo)

            G += Q * _Gamma_melting_layer(delta_h)

        if s_lo >= 12:
            Q = (h_T - 1200 - h_lo) / (h_hi - h_lo)
            G += Q

    return G


cdef (double, size_t) _accumulate_melting_layer(
        double h_lo, h_hi,
        double h_R, double h_rain_lo, double h_rain_hi,
        double *G_m, double *P_m,
        ) nogil:

    cdef:

        int[:] idx_v = RAIN_PROB_DATA_IDX_V
        double[:] rel_h_v = RAIN_PROB_DATA_REL_HEIGHT_V
        double[:] prob_v = RAIN_PROB_DATA_PROB_V

        size_t m = 0, n
        double h_T
        double P = 0., G = 1.
        double accum_P_G = 0.

    for n in range(49):
        P_m[n] = 0.

    G_m[0] = 1.

    for n in range(49):

        h_T = h_R + RAIN_PROB_DATA_REL_HEIGHT_V[n]

        if h_T <= h_rain_lo:
            continue

        if h_T - 1200 < h_rain_hi:

            G_m[m] = G = _path_avg_multiplier(h_T, h_lo, h_hi)
            P_m[m] = P = prov_v[n]
            if n < 48:
                m += 1
                accum_P_G += G * P
                G_m[m] = G = 1.
                P_m[m] = P = 0.

        else:

            G_m[m] = G = 1.
            P += prov_v[n]
            P_m[m] = P

    accum_P_G += G * P

    return accum_P_G, m


cdef (double, double, int, size_t) _precipitation_fading(
        double lon, double lat,
        double h_lo, double h_hi,
        double freq, int polarization,
        double q,  # TODO: what is q??? a time percentage?
        double P_r6, double M_T, double beta_rain, double h0,
        double h_rain_lo, double h_rain_hi, double d_rain,
        double *G_m, double *P_m,
        ) nogil:

    cdef:
        double h_R, h_Rtop
        double M_c, M_s
        double a1, b1, c1

        int path_class  # 0: non-rain path; 1: rain path

        double tau_rad, theta_rad
        double k_H, k_V, k, a_H, a_V, a
        double k_mod, a_mod, d_r, d_rmin

        double Q_0ra, F_wfv, Q_tran, R_wvr

        size_t M = 0

    h_R = 360 + 1000 * h0  # masl
    h_Rtop = h_R + 2400

    path_class = (0 if (P_r6 <= 1e-10 or h_rain_lo >= h_Rtop) else 1)

    if path_class == 0:

        Q_0ra = 0.
        F_wfv = 0.

    else:

        M_c = beta_rain * M_T
        M_s = (1 - beta_rain) * M_T

        Q_0ra = P_r6 * (1 - exp(-0.0079 * M_s / P_r6))

        a1 = 1.09
        b1 = (M_c + M_s) / 21797. / Q_0ra
        c1 = 26.02 * b1

        Q_tran = Q_0ra * exp(a1 * (2 * b1 - c1) / c1 ** 2)

        tau_rad = (0 if polarization == 0 else M_PI_2)
        theta_rad = 0.001 * (h_rain_hi - h_rain_lo) / d_rain  # eps_rain

        if freq < 1:
            k_H, k_V, k, a_H, a_V, a = _rain_coefficients_p838(
                0., theta_rad, tau_rad
                )
            K *= freq
        else:
            k_H, k_V, k, a_H, a_V, a = _rain_coefficients_p838(
                log10(freq), theta_rad, tau_rad
                )

        d_r = f_min(d_rain, 300)
        d_rmin = f_max(d_r, 1)

        k_mod = 1.763 ** a * k * (
            0.6546 * exp(-0.009546 * d_rmin) +
            0.3499 * exp(-0.001182 * d_rmin)
            )
        a_mod = (
            (0.753 + 0.197 / d_rmin) * a +
            0.1572 * exp(-0.02268 * d_rmin) -
            0.1594 * exp(-0.0003617 * d_rmin)
            )

        R_wvr = 6 * log10(Q_0ra / q) / log10(Q_0ra / Q_tran) - 3.
        accum_P_G, M = _accumulate_melting_layer(
            h_lo, h_hi, h_R, h_rain_lo, h_rain_hi, G_m, P_m,
            )
        F_wfv = 0.5 * (1 + tanh(R_wvr)) * accum_P_G

    return (path_class, Q_0ra, F_wfv, M)


cdef _time_percentage_prec_fade_level_exceeded(
        double A, int path_class,
        double *G_m, double *P_m, double M,
        ):

    cdef:

        double Q_rain
        size_t m

    if A < 0.:
        Q_rain = 100.  # %
    else:
        if path_class == 0:
            Q_rain = 0.
        else:
            Q_rain = 0.
            for m in range(0, M):
                Q_rain += P_m[m] * exp(-) # TODO


cdef inline double _free_space_loss_bfs(ppstruct pp) nogil:

    return 92.4 + 20 * log10(pp.freq) + 20 * log10(pp.d_fs)


cdef double _Q_caf(double A, double Q_0ca) nogil:

    cdef:
        # percentage of the non-rain time a given fade
        # in dB below the median signal level is exceeded
        double Q_caf
        double q_a, q_t, q_e, q_s

    if A >= 0:
        # signal fading
        q_t = 3.576 - 1.955 * log10(Q_0ca)
        q_a = (
            2 + (1 + 0.3 * 10 ** (-0.05 * A)) * 10 ** (-0.016 * A) * (
                q_t + 4.3 * (10 ** (-0.05 * A) + A / 800)
                )
            )
        Q_caf = 100 * (1 - exp(-10 ** (-0.05 * q_a * A) * log(2)))  # %
    else:
        # signal enhancement
        q_s = -4.05 - 2.35 * log10(Q_0ca)
        q_e = (
            8 + (1 + 0.3 * 10 ** (0.05 * A)) * 10 ** (0.035 * A) * (
                q_s + 12 * (10 ** (0.05 * A) - A / 800)
                )
            )
        Q_caf = 100 * (1 - exp(-10 ** (0.05 * q_e * A) * log(2)))  # %

    return Q_caf


cdef inline double _Q_caf_tropo(double A) nogil:

    cdef:
        # percentage of the non-rain time a given
        # fade in dB below the median signal level is exceeded
        # double Q_caf_tropo

    return (100 if A < 0 else 0)  # %


# @utils.ranged_quantity_input(
#     strip_input_units=True, output_unit=(cnv.dB, cnv.dB, cnv.dB)
#     )
def free_space_loss_bfs_cython(
        _PathProp pathprop
        ):
    '''
    Calculate the free space loss, L_bfs, of a propagating radio wave
    according to ITU-R P.2001-4 Eq. (41-42).

    Parameters
    ----------
    pathprop

    Returns
    -------
    L_bfs - double
        L_bfs - Free-space loss [dB]
    '''

    return _free_space_loss_bfs(pathprop._pp)

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
cdef (
    int, double, double, double, double, int, double, double, double
    ) _diffraction_helper_v16(
        double a_p,
        double distance,
        double[::1] d_v,
        double[::1] h_v,
        double h_ts, double h_rs,
        double wavelen,
        ) nogil:

    cdef:
        int i, dsize
        double d = distance, lam = wavelen, C_e500 = 500. / a_p
        int path_type

        double slope_i, slope_j, S_tim = -1.e31, S_tr, S_rim = -1.e31

        int nu_bull_idx
        double d_bp, nu_bull = -1.e31, nu_i
        double h_bp, h_eff, h_eff_i
        double x, y  # temporary vars

    dsize = d_v.shape[0]

    for i in range(1, dsize - 1):

        slope_i = (
            h_v[i] + C_e500 * d_v[i] * (d - d_v[i]) - h_ts
            ) / d_v[i]

        if slope_i > S_tim:
            S_tim = slope_i

    S_tr = (h_rs - h_ts) / d

    if S_tim < S_tr:
        path_type = 0
    else:
        path_type = 1

    if path_type == 1:
        # transhorizon
        # find Bullington point, etc.
        for i in range(1, dsize - 1):
            slope_j = (
                h_v[i] + C_e500 * d_v[i] * (d - d_v[i]) - h_rs
                ) / (d - d_v[i])

            if slope_j > S_rim:
                S_rim = slope_j

        d_bp = x = (h_rs - h_ts + S_rim * d) / (S_tim + S_rim)
        y = a_p + h_ts / 1000 + d_bp * (S_tim / 1000 - d / 2 / a_p)
        h_bp = 1000 * (sqrt(x ** 2 + y ** 2) - a_p)

        h_eff = (
            h_ts + S_tim * d_bp -
            (
                h_ts * (d - d_bp) + h_rs * d_bp
                ) / d
            )

        nu_bull = h_eff * sqrt(
            0.002 * d / lam / d_bp / (d - d_bp)
            )  # == nu_b in Eq. 20
        nu_bull_idx = -1  # dummy value

    else:
        # LOS

        # find Bullington point, etc.

        S_rim = NAN

        # diffraction parameter
        for i in range(1, dsize - 1):
            h_eff_i = (
                h_v[i] +
                C_e500 * d_v[i] * (d - d_v[i]) -
                (h_ts * (d - d_v[i]) + h_rs * d_v[i]) / d
                )
            nu_i = h_eff_i * sqrt(
                0.002 * d / lam / d_v[i] / (d - d_v[i])
                )
            if nu_i > nu_bull:
                nu_bull = nu_i
                nu_bull_idx = i
                h_eff = h_eff_i

        d_bp = x = d_v[nu_bull_idx]
        y = a_p + h_ts / 1000 + d_bp * (S_tr / 1000 - d / 2 / a_p)
        h_bp = 1000 * (sqrt(x ** 2 + y ** 2) - a_p)

    return (
        path_type, d_bp, h_bp, h_eff, nu_bull, nu_bull_idx, S_tim, S_rim, S_tr
        )


cdef (
        double, int, double, double, double, double, double,
        double, int, double, double, double, double, double,
        double, int, double, double, double, double, double,
        ) _diffraction_helper_v14(
        double a_e_50, double a_e_beta,
        double distance,
        double[::1] d_v,
        double[::1] h_v,
        double h_ts, double h_rs,
        double wavelen,
        ) nogil:

    cdef:
        int i, dsize
        double d = distance, lam = wavelen
        double C_e500 = 500. / a_e_50
        double C_b500 = 500. / a_e_beta

        double H_i, nu_i

        double zeta_m = NAN, zeta_t = NAN, zeta_r = NAN
        int i_m50 = -1, i_t50 = -1, i_r50 = -1
        # put default nu values to -1 (which leads to J(-1) == 0)
        double nu_m50 = -1.
        double nu_mbeta = -1.
        double nu_t50 = -1.
        double nu_tbeta = -1.
        double nu_r50 = -1.
        double nu_rbeta = -1.

        # height profile locations of the various diffraction edges
        double h50 = 0.
        double d50 = 0.
        double ht50 = 0.
        double dt50 = 0.
        double hr50 = 0.
        double dr50 = 0.

        # clearance (or knife edge) heights of the various diffraction edges
        double H50 = 0.
        double Ht50 = 0.
        double Hr50 = 0.

    dsize = d_v.shape[0]

    # Eq 14-15
    zeta_m = cos(atan(1.e-3 * (h_rs - h_ts) / d))

    nu_m50 = -1.e31
    for i in range(1, dsize - 1):

        H_i = (
            h_v[i] + C_e500 * d_v[i] * (d - d_v[i]) -
            (h_ts * (d - d_v[i]) + h_rs * d_v[i]) / d
            )
        nu_i = zeta_m * H_i * sqrt(
            0.002 * d / lam / d_v[i] / (d - d_v[i])
            )
        if nu_i > nu_m50:
            H50 = H_i
            nu_m50 = nu_i
            i_m50 = i

    h50 = h_v[i_m50]
    d50 = d_v[i_m50]

    if nu_m50 < -0.78:

        # every L will be zero
        return (
            zeta_m, i_m50, d50, h50, H50, nu_m50, nu_mbeta,
            zeta_t, i_t50, dt50, ht50, Ht50, nu_t50, nu_tbeta,
            zeta_r, i_r50, dr50, hr50, Hr50, nu_r50, nu_rbeta,
            )

    # calculate principle edge for beta
    H_i = (
        h50 + C_b500 * d50 * (d - d50) -
        (h_ts * (d - d50) + h_rs * d50) / d
        )
    nu_mbeta = zeta_m * H_i * sqrt(
        0.002 * d / lam / d50 / (d - d50)
        )

    if i_m50 > 1:

        # calculate transmitter-side secondary edge

        # Eq 17-18
        zeta_t = cos(atan(1.e-3 * (h50 - h_ts) / d50))
        nu_t50 = -1.e31
        for i in range(1, i_m50):

            H_i = (
                h_v[i] + C_e500 * d_v[i] * (d50 - d_v[i]) -
                (h_ts * (d50 - d_v[i]) + h50 * d_v[i]) / d50
                )
            nu_i = zeta_t * H_i * sqrt(
                0.002 * d50 / lam / d_v[i] / (d50 - d_v[i])
                )
            if nu_i > nu_t50:
                Ht50 = H_i
                nu_t50 = nu_i
                i_t50 = i

        ht50 = h_v[i_t50]
        dt50 = d_v[i_t50]

        # calculate beta
        H_i = (
            ht50 + C_b500 * dt50 * (d50 - dt50) -
            (h_ts * (d50 - dt50) + h50 * dt50) / d50
            )
        nu_tbeta = zeta_t * H_i * sqrt(
            0.002 * d50 / lam / dt50 / (d50 - dt50)
            )

        # sanity:
        if nu_t50 < -0.78:
            nu_tbeta = -1.

    if i_m50 + 1 < dsize - 1:

        # calculate receiver-side secondary edge

        # Eq 20-21
        zeta_r = cos(atan(1.e-3 * (h_rs - h50) / (d - d50)))
        nu_r50 = -1.e31
        for i in range(i_m50 + 1, dsize - 1):

            H_i = (
                h_v[i] + C_e500 * (d_v[i] - d50) * (d - d_v[i]) -
                (h50 * (d - d_v[i]) + h_rs * (d_v[i] - d50)) / (d - d50)
                )
            nu_i = zeta_r * H_i * sqrt(
                0.002 * (d - d50) / lam / (d_v[i] - d50) / (d - d_v[i])
                )
            if nu_i > nu_r50:
                Hr50 = H_i
                nu_r50 = nu_i
                i_r50 = i

        hr50 = h_v[i_r50]
        dr50 = d_v[i_r50]

        # calculate beta
        H_i = (
            hr50 + C_e500 * (dr50 - d50) * (d - dr50) -
            (h50 * (d - dr50) + h_rs * (dr50 - d50)) / (d - d50)
            )
        nu_rbeta = zeta_r * H_i * sqrt(
            0.002 * (d - d50) / lam / (dr50 - d50) / (d - dr50)
            )

        # sanity:
        if nu_r50 < -0.78:
            nu_rbeta = -1.

        return (
            zeta_m, i_m50, d50, h50, H50, nu_m50, nu_mbeta,
            zeta_t, i_t50, dr50, hr50, Hr50, nu_t50, nu_tbeta,
            zeta_r, i_r50, dt50, ht50, Ht50, nu_r50, nu_rbeta,
            )


# cdef (int, double, double, double, double, double, double, double, double) _path_geometry_helper(
#         double a_e,
#         double distance,
#         double[::1] d_v,
#         double[::1] h_v,
#         double h_ts, double h_rs, double h_st,
#         int nu_bull_idx, double duct_slope,
#         ) nogil:

#     cdef:
#         int i, dsize
#         double d = distance, m = duct_slope
#         int path_type

#         double theta_i, theta_j, theta_t, theta_r, theta
#         double theta_i_max = -1.e31, theta_j_max = -1.e31, theta_td
#         double eps_pt, eps_pr

#         int lt_idx, lr_idx
#         double d_lt, d_lr

#         double h_m_i, h_m = -1.e31

#     dsize = d_v.shape[0]

#     for i in range(1, dsize - 1):

#         theta_i = 1000. * atan(
#             (h_v[i] - h_ts) / 1.e3 / d_v[i] - d_v[i] / 2. / a_e
#             )
#         if theta_i > theta_i_max:
#             theta_i_max = theta_i
#             lt_idx = i

#     theta_td = 1000. * atan(
#         (h_rs - h_ts) / 1.e3 / d - d / 2. / a_e
#         )

#     if theta_i_max > theta_td:
#         path_type = 1
#     else:
#         path_type = 0

#     if path_type == 1:
#         # transhorizon

#         theta_t = theta_i_max
#         d_lt = d_v[lt_idx]

#         for i in range(1, dsize - 1):
#             theta_j = 1000. * atan(
#                 (h_v[i] - h_rs) / 1.e3 / (d - d_v[i]) -
#                 (d - d_v[i]) / 2. / a_e
#                 )
#             if theta_j > theta_j_max:
#                 theta_j_max = theta_j
#                 lr_idx = i

#         theta_r = theta_j_max
#         d_lr = d - d_v[lr_idx]

#         theta = 1.e3 * d / a_e + theta_t + theta_r

#         # calculate elevation angles of path
#         eps_pt = theta_t * 1.e-3 * 180. / M_PI
#         eps_pr = theta_r * 1.e-3 * 180. / M_PI

#         # calc h_m
#         for i in range(lt_idx, lr_idx + 1):
#             h_m_i = h_v[i] - (h_st + m * d_v[i])

#             if h_m_i > h_m:
#                 h_m = h_m_i

#     else:
#         # LOS

#         theta_t = theta_td

#         theta_r = 1000. * atan(
#             (h_ts - h_rs) / 1.e3 / d - d / 2. / a_e
#             )

#         theta = 1.e3 * d / a_e + theta_t + theta_r  # is this correct?

#         # calculate elevation angles of path
#         eps_pt = (
#             (h_rs - h_ts) * 1.e-3 / d - d / 2. / a_e
#             ) * 180. / M_PI
#         eps_pr = (
#             (h_ts - h_rs) * 1.e-3 / d - d / 2. / a_e
#             ) * 180. / M_PI

#         # horizon distance for LOS paths has to be set to distance to
#         # Bullington point in diffraction method
#         # assert (nu_bull_idx >= 0) and (nu_bull_idx < dsize)  # TODO
#         d_lt = d_v[nu_bull_idx]
#         d_lr = d - d_v[nu_bull_idx]

#         # calc h_m
#         # it seems, that h_m is calculated just from the profile height
#         # at the Bullington point???
#         h_m = h_v[nu_bull_idx] - (h_st + m * d_v[nu_bull_idx])

#     return (
#         path_type, theta_t, theta_r, eps_pt, eps_pr, theta, d_lt, d_lr, h_m
#         )


cdef (double, double, double) _free_space_loss_bfsg(
        ppstruct pp,
        ) nogil:
    # Better make this a member function?

    cdef:
        double rho_water
        (double, double) atten_dB

        double A_g, L_bfsg, E_sp, E_sbeta

    rho_water = 7.5 + 2.5 * pp.omega / 100.
    atten_dB = _specific_attenuation_annex2(
        pp.freq, pp.pressure, rho_water, pp.temperature
        )
    A_g = (atten_dB[0] + atten_dB[1]) * pp.distance

    L_bfsg = 92.5 + 20 * log10(pp.freq) + 20 * log10(pp.distance)
    L_bfsg += A_g

    E_sp = 2.6 * (
        1. - exp(-0.1 * (pp.d_lt + pp.d_lr))
        ) * log10(pp.time_percent / 50.)
    E_sbeta = 2.6 * (
        1. - exp(-0.1 * (pp.d_lt + pp.d_lr))
        ) * log10(pp.beta0 / 50.)

    return L_bfsg, E_sp, E_sbeta


# @utils.ranged_quantity_input(
#     strip_input_units=True, output_unit=(cnv.dB, cnv.dB, cnv.dB)
#     )
def free_space_loss_bfsg_cython(
        _PathProp pathprop
        ):
    '''
    Calculate the free space loss, L_bfsg, of a propagating radio wave
    according to ITU-R P.452-16 Eq. (8-12).

    Parameters
    ----------
    pathprop

    Returns
    -------
    (L_bfsg, E_sp, E_sβ) - tuple
        L_bfsg - Free-space loss [dB]
        E_sp - focussing/multipath correction for p% [dB]
        E_sbeta - focussing/multipath correction for beta0% [dB]

        with these, one can form
        L_b0p = L_bfsg + E_sp [dB]
        L_b0beta = L_bfsg + E_sβ [dB]

    Notes
    -----
    - Path profile parameters (PathProps object) can be derived using the
        [TODO].
    - This is similar to conversions.free_space_loss function but additionally
      accounts for athmospheric absorption and corrects for focusing and
      multipath effects.
    '''

    return _free_space_loss_bfsg(pathprop._pp)


cdef double _tropospheric_scatter_loss_bs(
        ppstruct pp, double G_t, double G_r
        ) nogil:
    # Better make this a member function?

    cdef:

        (double, double) atten_dB

        double A_g, L_f, L_c, L_bs

    atten_dB = _specific_attenuation_annex2(
        pp.freq, pp.pressure, 3., pp.temperature
        )

    A_g = (atten_dB[0] + atten_dB[1]) * pp.distance
    L_f = 25 * log10(pp.freq) - 2.5 * log10(0.5 * pp.freq) ** 2

    # TODO: why is toposcatter depending on gains towards horizon???
    L_c = 0.051 * exp(0.055 * (G_t + G_r))

    L_bs = (
        190. + L_f + 20 * log10(pp.distance) +
        0.573 * pp.theta - 0.15 * pp.N0 + L_c + A_g -
        10.1 * (-log10(pp.time_percent / 50.)) ** 0.7
        )

    return L_bs


def tropospheric_scatter_loss_bs_cython(
        _PathProp pathprop, double G_t=0., double G_r=0.
        ):
    '''
    Calculate the tropospheric scatter loss, L_bs, of a propagating radio wave
    according to ITU-R P.452-16 Eq. (45).

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    pathprop -
    G_t, G_r - Antenna gain (transmitter, receiver) in the direction of the
        horizon(!) along the great-circle interference path [dBi]

    Returns
    -------
    L_bs - Tropospheric scatter loss [dB]

    Notes
    -----
    - Path profile parameters (PathProps object) can be derived using the
        [TODO].
    '''

    return _tropospheric_scatter_loss_bs(pathprop._pp, G_t, G_r)


cdef double _ducting_loss_ba(
        ppstruct pp
        ) nogil:
    # Better make this a member function?

    cdef:

        double rho_water
        (double, double) atten_dB

        double A_g, A_lf, A_st, A_sr, A_ct, A_cr, A_p, A_d, L_ba

        double theta_t_prime, theta_r_prime, theta_t_prime2, theta_r_prime2
        double theta_prime

        double gamma_d, tau, eps, alpha, beta, mu_2, mu_3, d_I, Gamma

    rho_water = 7.5 + 2.5 * pp.omega / 100.
    atten_dB = _specific_attenuation_annex2(
        pp.freq, pp.pressure, rho_water, pp.temperature
        )

    A_g = (atten_dB[0] + atten_dB[1]) * pp.distance

    if pp.theta_t <= 0.1 * pp.d_lt:
        theta_t_prime = pp.theta_t
    else:
        theta_t_prime = 0.1 * pp.d_lt

    if pp.theta_r <= 0.1 * pp.d_lr:
        theta_r_prime = pp.theta_r
    else:
        theta_r_prime = 0.1 * pp.d_lr

    theta_t_prime2 = pp.theta_t - 0.1 * pp.d_lt
    theta_r_prime2 = pp.theta_r - 0.1 * pp.d_lr

    theta_prime = (
        1e3 * pp.distance / pp.a_e_50 + theta_t_prime + theta_r_prime
        )

    gamma_d = 5.e-5 * pp.a_e_50 * cpower(pp.freq, 1. / 3.)

    tau = 1. - exp(-4.12e-4 * cpower(pp.d_lm, 2.41))
    eps = 3.5
    alpha = -0.6 - eps * 1.e-9 * cpower(pp.distance, 3.1) * tau
    if alpha < -3.4:
        alpha = -3.4

    mu_2 = cpower(
        500. * pp.distance ** 2 / pp.a_e_50 /
        (sqrt(pp.h_te) + sqrt(pp.h_re)) ** 2,
        alpha
        )
    if mu_2 > 1.:
        mu_2 = 1.

    d_I = pp.distance - pp.d_lt - pp.d_lr
    if d_I > 40.:
        d_I = 40.

    if pp.h_m <= 10.:
        mu_3 = 1.
    else:
        mu_3 = exp(-4.6e-5 * (pp.h_m - 10.) * (43. + 6. * d_I))

    beta = pp.beta0 * mu_2 * mu_3

    Gamma = 1.076 / cpower(2.0058 - log10(beta), 1.012) * exp(
        -(9.51 - 4.8 * log10(beta) + 0.198 * log10(beta) ** 2) *
        1.e-6 * cpower(pp.distance, 1.13)
        )

    if pp.freq < 0.5:
        A_lf = 45.375 - 137. * pp.freq + 92.5 * pp.freq ** 2
    else:
        A_lf = 0.

    if theta_t_prime2 > 0.:
        A_st = (
            20 * log10(1 + 0.361 * theta_t_prime2 * sqrt(pp.freq * pp.d_lt)) +
            0.264 * theta_t_prime2 * cpower(pp.freq, 1. / 3.)
            )
    else:
        A_st = 0.

    if theta_r_prime2 > 0.:
        A_sr = (
            20 * log10(1 + 0.361 * theta_r_prime2 * sqrt(pp.freq * pp.d_lr)) +
            0.264 * theta_r_prime2 * cpower(pp.freq, 1. / 3.)
            )
    else:
        A_sr = 0.

    if (pp.omega >= 75.) and (pp.d_ct <= pp.d_lt) and (pp.d_ct <= 5.):
        A_ct = (
            -3 * exp(-0.25 * pp.d_ct ** 2) * (1. + tanh(3.5 - 0.07 * pp.h_ts))
            )
    else:
        A_ct = 0.

    if (pp.omega >= 75.) and (pp.d_cr <= pp.d_lr) and (pp.d_cr <= 5.):
        A_cr = (
            -3 * exp(-0.25 * pp.d_cr ** 2) * (1. + tanh(3.5 - 0.07 * pp.h_rs))
            )
    else:
        A_cr = 0.

    A_f = (
        102.45 + 20 * log10(pp.freq) + 20 * log10(pp.d_lt + pp.d_lr) +
        A_lf + A_st + A_sr + A_ct + A_cr
        )

    A_p = (
        -12 +
        (1.2 + 3.7e-3 * pp.distance) * log10(pp.time_percent / beta) +
        12. * cpower(pp.time_percent / beta, Gamma)
        )

    A_d = gamma_d * theta_prime + A_p

    L_ba = A_f + A_d + A_g

    return L_ba


def ducting_loss_ba_cython(
        _PathProp pathprop
        ):
    '''
    Calculate the ducting/layer reflection loss, L_ba, of a propagating radio
    wave according to ITU-R P.452-16 Eq. (46-56).

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    pathprop -

    Returns
    -------
    L_ba - Ducting/layer reflection loss [dB]

    Notes
    -----
    - Path profile parameters (PathProps object) can be derived using the
        [TODO.
    '''

    return _ducting_loss_ba(pathprop._pp)


cdef inline double _J_edgeknife(double nu) nogil:

    if nu < -0.78:
        return 0.
    else:
        return (
            6.9 + 20 * log10(
                sqrt((nu - 0.1) ** 2 + 1) + nu - 0.1
                )
            )


cdef inline double _diffraction_deygout_helper(
        double dist, double nu_m, double nu_t, double nu_r
        ) nogil:

    cdef:
        double L_d
        double L_m = _J_edgeknife(nu_m)
        double L_t = _J_edgeknife(nu_t)
        double L_r = _J_edgeknife(nu_r)

    L_d = L_m + (1 - exp(-L_m / 6.)) * (L_t + L_r + 10 + 0.04 * dist)

    return L_d


cdef inline double _diffraction_bullington_helper(
        double dist, double nu_bull
        ) nogil:

    cdef double L_uc = _J_edgeknife(nu_bull), L_bull

    L_bull = L_uc + (1 - exp(-L_uc / 6.)) * (10 + 0.02 * dist)

    return L_bull


cdef double _L_dft_G_helper(
        double beta_dft, double Y, double K,
        ) nogil:

    cdef:

        double B, res, tmp

    B = beta_dft * Y
    if B > 2.:
        res = 17.6 * sqrt(B - 1.1) - 5 * log10(B - 1.1) - 8
    else:
        res = 20 * log10(B + 0.1 * B ** 3)

    tmp = 2 + 20 * log10(K)
    if res < tmp:
        res = tmp

    return res


cdef double _L_dft_helper(
        double a_dft,
        double dist, double freq,
        double h_te, double h_re,
        int pol,
        double eps_r, double sigma
        ) nogil:

    cdef:

        double K, K2, K4
        double beta_dft
        double X, Y, Y_t, Y_r, F_X

    K = 0.036 * cpower(a_dft * freq, -1. / 3.) * cpower(
        (eps_r - 1) ** 2 + (18. * sigma / freq) ** 2, -0.25
        )

    if pol == 1:

        K *= cpower(eps_r ** 2 + (18. * sigma / freq) ** 2, 0.5)

    K2 = K * K
    K4 = K2 * K2
    beta_dft = (1. + 1.6 * K2 + 0.67 * K4) / (1. + 4.5 * K2 + 1.53 * K4)

    X = 21.88 * beta_dft * cpower(freq / a_dft ** 2, 1. / 3.) * dist
    Y = 0.9575 * beta_dft * cpower(freq ** 2 / a_dft, 1. / 3.)
    Y_t = Y * h_te
    Y_r = Y * h_re

    if X >= 1.6:
        F_X = 11. + 10 * log10(X) - 17.6 * X
    else:
        F_X = -20 * log10(X) - 5.6488 * cpower(X, 1.425)

    return (
        -F_X -
        _L_dft_G_helper(beta_dft, Y_t, K) -
        _L_dft_G_helper(beta_dft, Y_r, K)
        )


cdef double _L_dft(
        double a_dft,
        double dist, double freq,
        double h_te, double h_re,
        double omega_frac,
        int pol
        ) nogil:

    cdef:

        double L_dft_land, L_dft_sea, L_dft

    L_dft_land = _L_dft_helper(
        a_dft, dist, freq, h_te, h_re, pol, 22., 0.003
        )
    L_dft_sea = _L_dft_helper(
        a_dft, dist, freq, h_te, h_re, pol, 80., 5.0
        )

    L_dft = omega_frac * L_dft_sea + (1. - 0.01 * omega_frac) * L_dft_land

    return L_dft


cdef double _diffraction_spherical_earth_loss_helper(
        double dist, double freq,
        double a_p,
        double h_te, double h_re,
        double omega_frac,
        int pol
        ) nogil:

    cdef:

        double wavelen = 0.299792458 / freq
        double d_los, a_dft, a_em, L_dft, L_dsph
        double c, b, m, d_se1, d_se2, h_se, h_req

    d_los = sqrt(2 * a_p) * (sqrt(0.001 * h_te) + sqrt(0.001 * h_re))

    if dist >= d_los:

        a_dft = a_p
        return _L_dft(a_dft, dist, freq, h_te, h_re, omega_frac, pol)

    else:

        c = (h_te - h_re) / (h_te + h_re)
        m = 250. * dist ** 2 / a_p / (h_te + h_re)

        b = 2 * sqrt((m + 1.) / 3. / m) * cos(
            M_PI / 3. +
            1. / 3. * acos(3. * c / 2. * sqrt(3. * m / (m + 1.) ** 3))
            )

        d_se1 = 0.5 * dist * (1. + b)
        d_se2 = dist - d_se1

        h_se = (
            (h_te - 500 * d_se1 ** 2 / a_p) * d_se2 +
            (h_re - 500 * d_se2 ** 2 / a_p) * d_se1
            ) / dist

        h_req = 17.456 * sqrt(d_se1 * d_se2 * wavelen / dist)

        if h_se > h_req:

            return 0.

        a_em = 500. * (dist / (sqrt(h_te) + sqrt(h_re))) ** 2
        a_dft = a_em

        L_dft = _L_dft(a_dft, dist, freq, h_te, h_re, omega_frac, pol)

        if L_dft < 0:
            return 0.

        L_dsph = (1. - h_se / h_req) * L_dft

        return L_dsph


cdef double _delta_bullington_loss(
        ppstruct pp,
        int do_beta,
        ) nogil:

    cdef:

        double nu_bull, nu_bull_zh, a_p
        double h_te, h_re
        double omega_frac = 0.01 * pp.omega
        double L_bulla, L_bulls, L_dsph, L_d

    # median Earth radius, with height profile:

    if do_beta:
        nu_bull = pp.nu_bull_b0
        nu_bull_zh = pp.nu_bull_zh_b0
        a_p = pp.a_e_b0
    else:
        nu_bull = pp.nu_bull_50
        nu_bull_zh = pp.nu_bull_zh_50
        a_p = pp.a_e_50

    L_bulla = _diffraction_bullington_helper(
        pp.distance, nu_bull
        )
    L_bulls = _diffraction_bullington_helper(
        pp.distance, nu_bull_zh
        )

    h_te = pp.h_ts - pp.h_std  # != pp.h_te
    h_re = pp.h_rs - pp.h_srd  # != pp.h_re

    L_dsph = _diffraction_spherical_earth_loss_helper(
        pp.distance, pp.freq, a_p, h_te, h_re, omega_frac, pp.polarization
        )

    L_d = L_bulla + max(L_dsph - L_bulls, 0)

    return L_d


cdef inline double _I_helper(
        double x) nogil:

    cdef:

        double T, Z

    T = sqrt(-2 * log(x))
    Z = (
        (
            ((0.010328 * T + 0.802853) * T) + 2.515516698
            ) /
        (
            ((0.001308 * T + 0.189269) * T + 1.432788) * T + 1.
            )
        )

    return Z - T


cdef (double, double, double, double, double) _diffraction_loss_complete(
        ppstruct pp,
        ) nogil:

    cdef:

        double L_d_50, L_dp, L_d_beta, L_b0p, L_b0beta, L_bd_50, L_bd
        double L_bfsg, E_sp, E_sbeta, L_min_b0p
        double F_i

    if pp.version == 16:
        L_d_50 = _delta_bullington_loss(pp, 0)
    elif pp.version == 14:
        L_d_50 = _diffraction_deygout_helper(
            pp.distance, pp.nu_m50, pp.nu_t50, pp.nu_r50
            )

    if pp.time_percent > pp.beta0:
        F_i = _I_helper(pp.time_percent / 100.) / _I_helper(pp.beta0 / 100.)
    else:
        F_i = 1.

    if fabs(pp.time_percent - 50.) < 1.e-3:

        L_dp = L_d_50

    else:

        if pp.version == 16:
            L_d_beta = _delta_bullington_loss(pp, 1)
        elif pp.version == 14:
            L_d_beta = _diffraction_deygout_helper(
                pp.distance, pp.nu_mbeta, pp.nu_tbeta, pp.nu_rbeta
                )

        L_dp = L_d_50 + F_i * (L_d_beta - L_d_50)

    L_bfsg, E_sp, E_sbeta = _free_space_loss_bfsg(pp)

    L_b0p = L_bfsg + E_sp
    L_b0beta = L_bfsg + E_sbeta
    L_bd_50 = L_bfsg + L_d_50
    L_bd = L_b0p + L_dp

    # also calculate notional minimum basic transmission loss associated with
    # LoS propagation and over-sea sub-path diffraction;
    # this is needed for overall path attenuation calculation, but needs
    # the F_i factor, so we do it here

    if pp.time_percent < pp.beta0:
        L_min_b0p = L_b0p + (1. - 0.01 * pp.omega) * L_dp
    else:
        L_min_b0p = L_bd_50 + F_i * (
            L_b0beta + (1. - 0.01 * pp.omega) * L_dp - L_bd_50
            )

    return L_d_50, L_dp, L_bd_50, L_bd, L_min_b0p


def diffraction_loss_complete_cython(
        _PathProp pathprop
        ):
    '''
    Calculate the Diffraction loss of a propagating radio
    wave according to ITU-R P.452-16 Eq. (14-44).

    Parameters
    ----------
    pathprop -

    Returns
    -------
    (L_d_50, L_dp, L_bd_50, L_bd, L_min_b0p)
        L_d_50 - Median diffraction loss [dB]
        L_dp - Diffraction loss not exceeded for p% time, [dB]
        L_bd_50 - Median basic transmission loss associated with
            diffraction [dB]; L_bd_50 = L_bfsg + L_d50
        L_bd - Basic transmission loss associated with diffraction not
            exceeded for p% time [dB]; L_bd = L_b0p + L_dp
        L_min_b0p - Notional minimum basic transmission loss associated with
            LoS propagation and over-sea sub-path diffraction
        Note: L_d_50 and L_dp are just intermediary values; the complete
            diffraction loss is L_bd_50 or L_bd, respectively (taking into
            account a free-space loss component for the diffraction path)

    Notes
    -----
    - Path profile parameters (PathProps object) can be derived using the
        [TODO]
    '''

    return _diffraction_loss_complete(pathprop._pp)


cdef (double, double, double, double, double, double, double) _path_attenuation_complete(
        ppstruct pp,
        double G_t, double G_r,
        ) nogil:

    cdef:

        double _THETA = 0.3  # mrad
        double _XI = 0.8

        double _D_SW = 20  # km
        double _KAPPA = 0.5

        double _ETA = 2.5

        double F_j, F_k

        double L_bfsg, E_sp, E_sbeta, L_b0p, L_bs, L_ba
        double L_d_50, L_dp, L_bd_50, L_bd, L_min_b0p
        double L_min_bap, L_bam, L_b, L_b_corr

        double A_ht = 0., A_hr = 0.

    if pp.zone_t != CLUTTER.UNKNOWN:
        A_ht = _clutter_correction(
            pp.h_tg_in, pp.zone_t, pp.freq
            )
    if pp.zone_r != CLUTTER.UNKNOWN:
        A_hr = _clutter_correction(
            pp.h_rg_in, pp.zone_r, pp.freq
            )

    # not sure, if the 50% S_tim and S_tr values are to be used here...
    if pp.version == 16:
        F_j = 1 - 0.5 * (1. + tanh(
            3. * _XI * (pp.S_tim_50 - pp.S_tr_50) / _THETA
            ))
    elif pp.version == 14:
        F_j = 1 - 0.5 * (1. + tanh(
            3. * _XI * (pp.theta - _THETA) / _THETA
            ))
    F_k = 1 - 0.5 * (1. + tanh(
        3. * _KAPPA * (pp.distance - _D_SW) / _D_SW
        ))

    # free-space loss is not needed as an ingredient for final calculation
    # in itself (is included in diffraction part)
    # we use it here for debugging/informational aspects
    L_bfsg, E_sp, E_sbeta = _free_space_loss_bfsg(pp)
    L_b0p = L_bfsg + E_sp

    L_bs = _tropospheric_scatter_loss_bs(pp, G_t, G_r)
    L_ba = _ducting_loss_ba(pp)

    L_d_50, L_dp, L_bd_50, L_bd, L_min_b0p = _diffraction_loss_complete(pp)

    L_min_bap = _ETA * log(exp(L_ba / _ETA) + exp(L_b0p / _ETA))

    if L_bd < L_min_bap:
        L_bda = L_bd
    else:
        L_bda = L_min_bap + (L_bd - L_min_bap) * F_k

    L_bam = L_bda + (L_min_b0p - L_bda) * F_j

    L_b = -5 * log10(
        cpower(10, -0.2 * L_bs) +
        cpower(10, -0.2 * L_bam)
        )
    L_b_corr = L_b + A_ht + A_hr
    L = L_b_corr - G_t - G_r

    return L_b0p, L_bd, L_bs, L_ba, L_b, L_b_corr, L


def path_attenuation_complete_cython(
        _PathProp pathprop, double G_t=0., double G_r=0.
        ):
    '''
    Calculate the total loss of a propagating radio
    wave according to ITU-R P.452-16 Eq. (58-64).

    Parameters
    ----------
    pathprop -
    G_t, G_r - Antenna gain (transmitter, receiver) in the direction of the
        horizon(!) along the great-circle interference path [dBi]

    Returns
    -------
    (L_b0p, L_bd, L_bs, L_ba, L_b)
        L_b0p - Free-space loss [dB]
        L_bd - Basic transmission loss associated with diffraction not
            exceeded for p% time [dB]; L_bd = L_b0p + L_dp
        L_bs - Tropospheric scatter loss [dB]
        L_ba - Ducting/layer reflection loss [dB]
        L_b - Complete path propagation loss [dB]
        L_b_corr - As L_b but with clutter correction [dB]
        L - As L_b_corr but with gain correction [dB]

    Notes
    -----
    - Path profile parameters (PathProps object) can be derived using the
        [TODO]
    '''

    return _path_attenuation_complete(pathprop._pp, G_t, G_r)


cdef double _clutter_correction(
        double h_g, int zone, double freq
        ) nogil:

    cdef:

        double h_a, d_k
        double F_fc
        double A_h

    if zone < 0:
        return 0

    h_a = CLUTTER_DATA_V[zone, 0]
    d_k = CLUTTER_DATA_V[zone, 1]

    F_fc = 0.25 + 0.375 * (1. + tanh(7.5 * (freq - 0.5)))

    A_h = 10.25 * F_fc * exp(-d_k) * (
        1. - tanh(6 * (h_g / h_a - 0.625))
        ) - 0.33

    return A_h


def clutter_correction_cython(
        double h_g, int zone, double freq
        ):
    '''
    Calculate the Clutter loss of a propagating radio
    wave according to ITU-R P.452-16 Eq. (57).

    Parameters
    ----------
    h_g - height above ground [m]
    zone - Clutter category (see CLUTTER enum)
    freq - frequency [GHz]

    Returns
    -------
    A_h - Clutter correction to path attenuation [dB]

    Notes
    -----
    - Path profile parameters (PathProps object) can be derived using the
        [TODO]
    '''

    return _clutter_correction(h_g, zone, freq)


# ############################################################################
#
# Attenuation map making (fast)
#
# ############################################################################
#
# Idea: only calculate Geodesics/height profiles to map edges and apply hashes
#
# ############################################################################


# TODO: d_tm, d_lm, d_ct, and d_cr need to be calculated for each pixel!
# Likewise for clutter zones.
def height_map_data_cython(
        double lon_t, double lat_t,
        double map_size_lon, double map_size_lat,
        double map_resolution=3. / 3600.,
        int do_cos_delta=1,
        int zone_t=CLUTTER.UNKNOWN, int zone_r=CLUTTER.UNKNOWN,
        d_tm=None, d_lm=None,
        d_ct=None, d_cr=None,
        omega=None,
        ):

    '''
    Calculate height profiles and auxillary maps needed for atten_map_fast.

    This can be used to cache height-profile data. Since it is independent
    of frequency, timepercent, Tx and Rx heights, etc., one can re-use
    it to save computing time when doing batch jobs.

    Parameters
    ----------
    lon_t, lat_t : double
        Geographic longitude/latitude of transmitter [deg]
    map_size_lon, map_size_lat : double
        Map size in longitude/latitude[deg]
    map_resolution : double, optional
        Pixel resolution of map [deg] (default: 3 arcsec)
    do_cos_delta : int, optional
        If True, divide `map_size_lon` by `cos(lat_t)` to produce a more
        square-like map. (default: True)
    zone_t, zone_r : CLUTTER enum, optional
        Clutter type for transmitter/receiver terminal.
        (default: CLUTTER.UNKNOWN)
    d_tm : double, optional
        longest continuous land (inland + coastal) section of the
        great-circle path [km]
        (default: distance between Tx and Rx)
    d_lm : double, optional
        longest continuous inland section of the great-circle path [km]
        (default: distance between Tx and Rx)
    d_ct, d_cr : double, optional
        Distance over land from transmitter/receiver antenna to the coast
        along great circle interference path [km]
        (default: 50000 km)
    omega : double, optional
        Fraction of the path over water [%] (see Table 3)
        (default: 0%)

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
          needed by `~pycraf.atten_map_fast_cython`!

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
          needed by `~pycraf.atten_map_fast_cython`!

        - "N0_map", "delta_N_map", "beta0_map" : `~numpy.ndarray` 2D (float; (mx, my))

          The `N0`, `delta_N`, and `beta0` values for each pixel in the map.

        - "path_idx_map" : `~numpy.ndarray` 2D (int; (mx, my))

          Map of path IDs for each pixel. With this index, one can query
          the associated height profile from `height_profs`.

        - "pix_dist_map" : `~numpy.ndarray` 2D (float; (mx, my))

          Angular distance of the closest path to each of the map pixels.

        - "dist_end_idx_map" : `~numpy.ndarray` 2D (int; (mx, my))

          Index of the last element in the dist/height profiles to be used
          when querying the profiles from `dist_prof` and
          `height_profs`.

        - "dist_prof" : `~numpy.ndarray` 1D (float, (mh, ))

          Distance values for each of the paths stored in `height_profs`.

        - "bearing_prof" : `~numpy.ndarray` 1D (float, (me, ))

          Bearings for each of the paths stored in `height_profs`
          (w.r.t. map center).

        - "height_profs" : `~numpy.ndarray` 2D (float, (me, mh))

          Height profiles to each of the pixels on the map edge, zero padded.

        - "zheight_prof" : `~numpy.ndarray` 1D (float, (mh, ))

          Zero-valued array of the same length as `height_profs` for
          convenience.

    Notes
    -----
    - Path attenuation is completely symmetric, i.e., it doesn't matter if
      the transmitter or the receiver is situated in the map center.
    - If you need to calculate the full lon/lat coordinates of the (hi-res)
      samples in "height_profs", you can use the bearings ("bearing_prof")
      and distances ("dist_prof") and the `~pycraf.pathprof.geoid_direct`
      function to do this.
    '''

    cdef:

        # need 3x better resolution than map_resolution
        double hprof_step = map_resolution * 3600. / 1. * 30. / 3.
        double cosdelta
        double max_distance, min_pa_res

        int xi, yi, i, xidx, yidx, mid_idx
        int bidx, didx
        double refx, refy
        double lon_t_rad, lat_t_rad, lon_r, lat_r
        double pdist

        # need views on all relevant numpy arrays for faster access
        np.float64_t[::1] _xcoords, _ycoords
        np.float64_t[::1] _distances, _start_bearings
        np.float64_t[:, ::1] _lons, _lats, _back_bearings
        np.float64_t[:, ::1] _heights
        np.int32_t[:, ::1] _path_idx_map, _dist_end_idx_map
        np.float64_t[:, ::1] _pix_dist_map
        np.float64_t[:, ::1] _lon_mid_map, _lat_mid_map
        np.float64_t[:, ::1] _dist_map
        np.float64_t[:, ::1] _bearing_map, _backbearing_map

    # print('using hprof_step = {:.1f} m'.format(hprof_step))

    cosdelta = 1. / cos(DEG2RAD * lat_t) if do_cos_delta else 1.

    # construction map arrays
    _xcoords = xcoords = np.arange(
        lon_t - cosdelta * map_size_lon / 2,
        lon_t + cosdelta * map_size_lon / 2 + 1.e-6,
        cosdelta * map_resolution,
        )
    _ycoords = ycoords = np.arange(
        lat_t - map_size_lat / 2,
        lat_t + map_size_lat / 2 + 1.e-6,
        map_resolution,
        )
    lon_t_rad, lat_t_rad = DEG2RAD * lon_t, DEG2RAD * lat_t
    xcoords_rad = np.radians(xcoords)
    ycoords_rad = np.radians(ycoords)
    # print(
    #     xcoords[0], xcoords[len(xcoords) - 1],
    #     ycoords[0], ycoords[len(ycoords) - 1]
    #     )

    # find max distance (will be one of the edges)
    max_distance = max([
        cygeodesics.inverse_cython(
            lon_t_rad, lat_t_rad, xcoords_rad[i], ycoords_rad[j],
            )[0]
        for i, j in [
            (0, 0),
            (0, ycoords_rad.size - 1),
            (xcoords_rad.size - 1, 0),
            (xcoords_rad.size - 1, ycoords_rad.size - 1)
            ]
        ])  # m
    # print('max distance (to corner coords)', max_distance / 1000, 'km')

    # find necessary position angle resolution (again, using edges)
    min_pa_res = min([
        fabs(
            cygeodesics.inverse_cython(
                lon_t_rad, lat_t_rad, xcoords_rad[i], ycoords_rad[k],
                )[1] -
            cygeodesics.inverse_cython(
                lon_t_rad, lat_t_rad, xcoords_rad[j], ycoords_rad[k],
                )[1]
            )
        for i, j, k in [
            (0, 1, 0),
            (0, 1, ycoords_rad.size - 1),
            (xcoords_rad.size - 1, xcoords_rad.size - 2, 0),
            (xcoords_rad.size - 1, xcoords_rad.size - 2, ycoords_rad.size - 1)
            ]
        ]) / 2  # rad
    # print('min pos angle resolution (at corner coords)', min_pa_res)


    # path_idx_map stores the index of the edge-path that is closest
    # to any given map pixel
    _path_idx_map = path_idx_map = np.zeros(
        (ycoords.size, xcoords.size), dtype=np.int32
        )

    # to define and find closest paths, we store the true angular distance
    # in pix_dist_map; Note, since distances are small, it is ok to do
    # this on the sphere (and not on geoid)
    _pix_dist_map = pix_dist_map = np.full(
        (ycoords.size, xcoords.size), 1.e30, dtype=np.float64
        )

    # dist_end_idx_map stores the (distance) index in the height profile
    # of the closest edge path, such that one can use a slice (0, end_idx)
    # to get a height profile approximately valid for any given pixel
    _dist_end_idx_map = dist_end_idx_map = np.zeros(
        (ycoords.size, xcoords.size), dtype=np.int32
        )

    # store lon_mid, lat_mid,
    _lon_mid_map = lon_mid_map = np.zeros(
        (ycoords.size, xcoords.size), dtype=np.float64
        )
    _lat_mid_map = lat_mid_map = np.zeros(
        (ycoords.size, xcoords.size), dtype=np.float64
        )
    _dist_map = dist_map = np.zeros(
        (ycoords.size, xcoords.size), dtype=np.float64
        )

    # store bearings
    _bearing_map = bearing_map = np.zeros(
        (ycoords.size, xcoords.size), dtype=np.float64
        )
    _backbearing_map = backbearing_map = np.zeros(
        (ycoords.size, xcoords.size), dtype=np.float64
        )

    # obtain all path's height profiles
    # generate start bearings:
    _start_bearings = start_bearings = np.arange(0, 2 * np.pi, min_pa_res)

    # calculate path positions
    _distances = distances = np.arange(
        0, max_distance + hprof_step, hprof_step
        )

    lons_rad, lats_rad, back_bearings_rad = cygeodesics.direct_cython(
        lon_t_rad, lat_t_rad,
        start_bearings[:, np.newaxis],
        distances[np.newaxis]
        )
    _lons, _lats = lons, lats = np.degrees(lons_rad), np.degrees(lats_rad)
    _back_bearings = back_bearings = np.degrees(back_bearings_rad)

    # print(xcoords, ycoords)
    # print(lons.min(), lons.max(), lats.min(), lats.max())

    # hgt_res may not be set correctly yet, if no call to srtm was made before
    # let's do a simple query to make sure, it is set
    srtm._srtm_height_data(lon_t, lat_t)

    if hprof_step > srtm.SrtmConf.hgt_res / 1.5:
        hdistances = np.arange(
            0, max_distance + hprof_step / 3, hprof_step / 3
            )

        hlons_rad, hlats_rad, _ = cygeodesics.direct_cython(
            lon_t_rad, lat_t_rad,
            start_bearings[:, np.newaxis],
            hdistances[np.newaxis]
            )

        hheights = srtm._srtm_height_data(
            np.degrees(hlons_rad), np.degrees(hlats_rad)
            ).astype(np.float64)
        heights = np.empty_like(lons_rad)
        # now smooth/interpolate this to the desired step width
        cygeodesics.regrid2d_with_x(
            hdistances, hheights, distances, heights,
            hprof_step / 2.35, regular=True
            )

    else:

        heights = srtm._srtm_height_data(lons, lats).astype(np.float64)

    _heights = heights
    distances *= 1e-3  # convert to km
    _distances = distances

    refx, refy = _xcoords[0], _ycoords[0]

    with nogil:
        for bidx in range(_start_bearings.shape[0]):

            for didx in range(_distances.shape[0]):

                lon_r, lat_r = _lons[bidx, didx], _lats[bidx, didx]

                # need to find closest pixel index in map
                xidx = int((lon_r - refx) / cosdelta / map_resolution + 0.5)
                yidx = int((lat_r - refy) / map_resolution + 0.5)

                if xidx < 0:
                    xidx = 0
                if xidx >= _xcoords.shape[0]:
                    xidx = _xcoords.shape[0] - 1
                if yidx < 0:
                    yidx = 0
                if yidx >= _ycoords.shape[0]:
                    yidx = _ycoords.shape[0] - 1

                pdist = true_angular_distance(
                    _xcoords[xidx], _ycoords[yidx], lon_r, lat_r
                    )

                if pdist < _pix_dist_map[yidx, xidx]:
                    _pix_dist_map[yidx, xidx] = pdist
                    _path_idx_map[yidx, xidx] = bidx
                    _dist_end_idx_map[yidx, xidx] = didx
                    mid_idx = didx // 2
                    _lon_mid_map[yidx, xidx] = _lons[bidx, mid_idx]
                    _lat_mid_map[yidx, xidx] = _lats[bidx, mid_idx]
                    _dist_map[yidx, xidx] = _distances[didx]
                    _bearing_map[yidx, xidx] = _start_bearings[bidx]
                    _backbearing_map[yidx, xidx] = _back_bearings[bidx, didx]

    # store delta_N, beta0, N0
    delta_N_map, beta0_map, N0_map = helper._radiomet_data_for_pathcenter(
        lon_mid_map, lat_mid_map, dist_map, dist_map
        )

    # TODO: derive the following programmatically
    zone_t_map = np.full_like(path_idx_map, zone_t)
    zone_r_map = np.full_like(path_idx_map, zone_r)

    # TODO: derive the following programmatically
    if d_tm is None:
        d_tm_map = dist_map
    else:
        d_tm_map = np.full_like(dist_map, d_tm)

    if d_lm is None:
        d_lm_map = dist_map
    else:
        d_lm_map = np.full_like(dist_map, d_lm)

    if d_ct is None:
        d_ct_map = np.full_like(dist_map, 50000.)
    else:
        d_ct_map = np.full_like(dist_map, d_ct)

    if d_cr is None:
        d_cr_map = np.full_like(dist_map, 50000.)
    else:
        d_cr_map = np.full_like(dist_map, d_cr)

    if omega is None:
        omega_map = np.full_like(dist_map, 0.)
    else:
        omega_map = np.full_like(dist_map, omega)

    dist_prof = distances
    height_profs = heights
    zheight_prof = np.zeros_like(distances)

    hprof_data = {}
    hprof_data['lon_t'] = lon_t
    hprof_data['lat_t'] = lat_t
    hprof_data['xcoords'] = xcoords
    hprof_data['ycoords'] = ycoords
    hprof_data['map_size_lon'] = map_size_lon
    hprof_data['map_size_lat'] = map_size_lat
    hprof_data['hprof_step'] = hprof_step
    hprof_data['map_resolution'] = map_resolution
    hprof_data['do_cos_delta'] = do_cos_delta

    hprof_data['path_idx_map'] = path_idx_map
    hprof_data['pix_dist_map'] = pix_dist_map
    hprof_data['dist_end_idx_map'] = dist_end_idx_map
    hprof_data['lon_mid_map'] = lon_mid_map
    hprof_data['lat_mid_map'] = lat_mid_map
    hprof_data['dist_map'] = dist_map
    hprof_data['bearing_map'] = bearing_map
    hprof_data['back_bearing_map'] = backbearing_map

    hprof_data['delta_N_map'] = delta_N_map
    hprof_data['beta0_map'] = beta0_map
    hprof_data['N0_map'] = N0_map

    hprof_data['zone_t_map'] = zone_t_map
    hprof_data['zone_r_map'] = zone_r_map

    hprof_data['d_tm_map'] = d_tm_map
    hprof_data['d_lm_map'] = d_lm_map
    hprof_data['d_ct_map'] = d_ct_map
    hprof_data['d_cr_map'] = d_cr_map
    hprof_data['omega_map'] = omega_map

    hprof_data['dist_prof'] = dist_prof
    hprof_data['bearing_prof'] = (
        ((start_bearings * RAD2DEG) + 180.) % 360. - 180.
        )
    hprof_data['height_profs'] = height_profs
    hprof_data['zheight_prof'] = zheight_prof

    return hprof_data


def atten_map_fast_cython(
        double freq,
        double temperature,
        double pressure,
        double h_tg, double h_rg,
        double time_percent,
        object hprof_data not None,  # dict_like
        int polarization=0,
        int version=16,
        ):
    '''
    Calculate attenuation maps using a fast method.

    Parameters
    ----------
    freq : double
        Frequency of radiation [GHz]
    temperature : double
        Temperature (K)
    pressure : double
        Pressure (hPa)
    h_tg, h_rg : double
        Transmitter/receiver heights over ground [m]
    timepercent : double
        Time percentage [%] (maximal 50%)
    hprof_data : dict, dict-like
        Dictionary with height profiles and auxillary maps as
        calculated with `~pycraf.pathprof.height_map_data`.
    polarization : int, optional
        Polarization (default: 0)
        Allowed values are: 0 - horizontal, 1 - vertical
    version : int, optional
        ITU-R Rec. P.452 version. Allowed values are: 14, 16

    Returns
    -------
    float_results : 3D `~numpy.ndarray`

        Results of the calculation. The first and second dimension
        refer to the maps, while the third axis has the following
        meaning:

        0-5: Attenuation maps (i.e., the output of
            path_attenuation_complete without gain-corrected values)
        6-7: Path elevation angles
        8-9: Path horizon distances (for LoS paths, this is distance
            to Bullington point)

        0) L_b0p - Free-space loss including focussing effects
           (for p% of time) [dB]
        1) L_bd - Basic transmission loss associated with diffraction
           not exceeded for p% time [dB]; L_bd = L_b0p + L_dp
        2) L_bs - Tropospheric scatter loss [dB]
        3) L_ba - Ducting/layer reflection loss [dB]
        4) L_b - Complete path propagation loss [dB]
        5) L_b_corr - As L_b but with clutter correction [dB]
        6) eps_pt - Elevation angle of paths w.r.t. Tx [deg]
        7) eps_pr - Elevation angle of paths w.r.t. Rx [deg]
        8) d_lt - Distance to horizon w.r.t. Tx [km]
        9) d_lr - Distance to horizon w.r.t. Rx [km]

    int_results : 3D `~numpy.ndarray`

        As `float_results` but for integer-typed values:

        0) path_type - Regular path type (0 - LoS, 1 - Trans-horizon)

    Notes
    -----
    - The diffraction-loss algorithm was changed between ITU-R P.452
      version 14 and 15. The former used a Deygout method, the new one
      is based on a Bullington calculation with correction terms.
    '''

    # TODO: implement map-based clutter handling; currently, only a single
    # clutter zone type is possible for each of Tx and Rx

    assert time_percent <= 50.
    assert version == 14 or version == 16

    cdef:
        # must set gains to zero, because gain is direction dependent
        double G_t = 0., G_r = 0.
        ppstruct *pp
        int xi, yi, xlen, ylen
        int eidx, didx

        double[:, ::1] clutter_data_v = CLUTTER_DATA

        double L_b0p, L_bd, L_bs, L_ba, L_b, L_b_corr, L_dummy

    xcoords, ycoords = hprof_data['xcoords'], hprof_data['ycoords']

    float_res = np.zeros((10, len(ycoords), len(xcoords)), dtype=np.float64)
    int_res = np.zeros((1, len(ycoords), len(xcoords)), dtype=np.int32)

    cdef:
        double[:, :, :] float_res_v = float_res
        int[:, :, :] int_res_v = int_res

        # since we allow all dict_like objects for hprof_data,
        # we have to make sure, that arrays are numpy and contiguous
        # (have in mind, that one might use hdf5 data sets)

        _cf = np.ascontiguousarray

        double[::1] xcoords_v = _cf(hprof_data['xcoords'])
        double[::1] ycoords_v = _cf(hprof_data['ycoords'])
        double lon_t = np.double(hprof_data['lon_t'])
        double lat_t = np.double(hprof_data['lat_t'])
        double hprof_step = np.double(hprof_data['hprof_step'])

        int[:, :] path_idx_map_v = _cf(hprof_data['path_idx_map'])
        int[:, :] dist_end_idx_map_v = _cf(hprof_data['dist_end_idx_map'])
        double[:, :] dist_map_v = _cf(hprof_data['dist_map'])
        double[:, :] delta_N_map_v = _cf(hprof_data['delta_N_map'])
        double[:, :] beta0_map_v = _cf(hprof_data['beta0_map'])
        double[:, :] N0_map_v = _cf(hprof_data['N0_map'])

        int[:, :] zone_t_map_v = _cf(hprof_data['zone_t_map'])
        int[:, :] zone_r_map_v = _cf(hprof_data['zone_r_map'])
        double[:, :] d_tm_map_v = _cf(hprof_data['d_tm_map'])
        double[:, :] d_lm_map_v = _cf(hprof_data['d_lm_map'])
        double[:, :] d_ct_map_v = _cf(hprof_data['d_ct_map'])
        double[:, :] d_cr_map_v = _cf(hprof_data['d_cr_map'])
        double[:, :] omega_map_v = _cf(hprof_data['omega_map'])

        double[::1] dist_prof_v = _cf(hprof_data['dist_prof'])
        double[:, ::1] height_profs_v = _cf(hprof_data['height_profs'])
        double[::1] zheight_prof_v = _cf(hprof_data['zheight_prof'])

    xlen = len(xcoords)
    ylen = len(ycoords)

    with nogil, parallel():

        pp = <ppstruct *> malloc(sizeof(ppstruct))
        if pp == NULL:
            abort()

        pp.version = version
        pp.freq = freq
        pp.wavelen = 0.299792458 / freq
        pp.temperature = temperature
        pp.pressure = pressure
        pp.lon_t = lon_t
        pp.lat_t = lat_t
        pp.h_tg = h_tg
        pp.h_rg = h_rg
        pp.h_tg_in = h_tg
        pp.h_rg_in = h_rg

        pp.hprof_step = hprof_step
        pp.time_percent = time_percent
        # TODO: add functionality to produce the following
        # five parameters programmatically (using some kind of Geo-Data)
        pp.polarization = polarization

        for yi in prange(ylen, schedule='guided', chunksize=10):

            for xi in range(xlen):

                eidx = path_idx_map_v[yi, xi]
                didx = dist_end_idx_map_v[yi, xi]

                if didx < 4:
                    continue

                pp.lon_r = xcoords_v[xi]
                pp.lat_r = ycoords_v[yi]
                pp.zone_t = zone_t_map_v[yi, xi]
                pp.zone_r = zone_r_map_v[yi, xi]

                if pp.zone_t == CLUTTER.UNKNOWN:
                    pp.h_tg = h_tg
                else:
                    pp.h_tg = f_max(clutter_data_v[pp.zone_t, 0], h_tg)

                if pp.zone_r == CLUTTER.UNKNOWN:
                    pp.h_rg = h_rg
                else:
                    pp.h_rg = f_max(clutter_data_v[pp.zone_r, 0], h_rg)

                pp.d_tm = d_tm_map_v[yi, xi]
                pp.d_lm = d_lm_map_v[yi, xi]
                pp.d_ct = d_ct_map_v[yi, xi]
                pp.d_cr = d_cr_map_v[yi, xi]
                pp.omega = omega_map_v[yi, xi]

                pp.distance = dist_map_v[yi, xi]

                pp.delta_N = delta_N_map_v[yi, xi]
                pp.beta0 = beta0_map_v[yi, xi]
                pp.N0 = N0_map_v[yi, xi]

                # assigning not possible in prange, but can use directly below
                # dists_v = dist_prof_v[0:didx + 1]
                # heights_v = height_profs_v[eidx, 0:didx + 1]
                # zheights_v = zheight_prof_v[0:didx + 1]

                _process_path(
                    pp,
                    # dists_v,
                    # heights_v,
                    # zheights_v,
                    dist_prof_v[0:didx + 1],
                    height_profs_v[eidx, 0:didx + 1],
                    zheight_prof_v[0:didx + 1],
                    )

                (
                    L_b0p, L_bd, L_bs, L_ba, L_b, L_b_corr, L_dummy
                    ) = _path_attenuation_complete(pp[0], G_t, G_r)

                float_res_v[0, yi, xi] = L_b0p
                float_res_v[1, yi, xi] = L_bd
                float_res_v[2, yi, xi] = L_bs
                float_res_v[3, yi, xi] = L_ba
                float_res_v[4, yi, xi] = L_b
                float_res_v[5, yi, xi] = L_b_corr
                float_res_v[6, yi, xi] = pp.eps_pt
                float_res_v[7, yi, xi] = pp.eps_pr
                float_res_v[8, yi, xi] = pp.d_lt
                float_res_v[9, yi, xi] = pp.d_lr

                int_res_v[0, yi, xi] = pp.path_type

        free(pp)

    return float_res, int_res


def atten_path_fast_cython(
        double freq,
        double temperature,
        double pressure,
        double h_tg, double h_rg,
        double time_percent,
        object hprof_data not None,  # dict_like
        int polarization=0,
        int version=16,
        ):

    '''
    Calculate attenuation along a path using a parallelized method.

    Parameters
    ----------
    freq : double
        Frequency of radiation [GHz]
    temperature : double
        Temperature (K)
    pressure : double
        Pressure (hPa)
    h_tg, h_rg : double
        Transmitter/receiver heights over ground [m]
    timepercent : double
        Time percentage [%] (maximal 50%)
    hprof_data : dict, dict-like
        Dictionary with height profile auxillary data as
        calculated with `~pycraf.pathprof.height_path_data`.
    polarization : int, optional
        Polarization (default: 0)
        Allowed values are: 0 - horizontal, 1 - vertical
    version : int, optional
        ITU-R Rec. P.452 version. Allowed values are: 14, 16

    Returns
    -------
    float_results : 2D `~numpy.ndarray`

        Results of the calculation. The second dimension refers to
        the path, while the first axis has the following meaning:

        0-5: Attenuation maps (i.e., the output of
            path_attenuation_complete without gain-corrected values)
        6-7: Path elevation angles
        8-9: Path horizon distances (for LoS paths, this is distance
            to Bullington point)

        0) L_b0p - Free-space loss including focussing effects
           (for p% of time) [dB]
        1) L_bd - Basic transmission loss associated with diffraction
           not exceeded for p% time [dB]; L_bd = L_b0p + L_dp
        2) L_bs - Tropospheric scatter loss [dB]
        3) L_ba - Ducting/layer reflection loss [dB]
        4) L_b - Complete path propagation loss [dB]
        5) L_b_corr - As L_b but with clutter correction [dB]
        6) eps_pt - Elevation angle of paths w.r.t. Tx [deg]
        7) eps_pr - Elevation angle of paths w.r.t. Rx [deg]
        8) d_lt - Distance to horizon w.r.t. Tx [km]
        9) d_lr - Distance to horizon w.r.t. Rx [km]

    int_results : 2D `~numpy.ndarray`

        As `float_results` but for integer-typed values:

        0) path_type - Regular path type (0 - LoS, 1 - Trans-horizon)

    Notes
    -----
    - The diffraction-loss algorithm was changed between ITU-R P.452
      version 14 and 15. The former used a Deygout method, the new one
      is based on a Bullington calculation with correction terms.
    '''

    # TODO: implement map-based clutter handling; currently, only a single
    # clutter zone type is possible for each of Tx and Rx

    assert time_percent <= 50.
    assert version == 14 or version == 16

    cdef:
        # must set gains to zero, because gain is direction dependent
        double G_t = 0., G_r = 0.
        ppstruct *pp

        double[:, ::1] clutter_data_v = CLUTTER_DATA

        double L_b0p, L_bd, L_bs, L_ba, L_b, L_b_corr, L_dummy

        _cf = np.ascontiguousarray

        double[::1] distances_v = _cf(hprof_data['distances'])
        double[::1] heights_v = _cf(hprof_data['heights'])
        double[::1] zheights_v = np.zeros_like(_cf(hprof_data['heights']))
        double[::1] omega_v = _cf(hprof_data['omega'])
        double[::1] d_tm_v = _cf(hprof_data['d_tm'])
        double[::1] d_lm_v = _cf(hprof_data['d_lm'])
        double[::1] d_ct_v = _cf(hprof_data['d_ct'])
        double[::1] d_cr_v = _cf(hprof_data['d_cr'])
        int zone_t = hprof_data['zone_t']
        int[::1] zone_r_v = _cf(hprof_data['zone_r'])
        double[::1] delta_N_v = _cf(hprof_data['delta_N'])
        double[::1] N0_v = _cf(hprof_data['N0'])
        double[::1] beta0_v = _cf(hprof_data['beta0'])

        int i, max_path_length = distances_v.size

    float_res = np.zeros((10, max_path_length), dtype=np.float64)
    int_res = np.zeros((1, max_path_length), dtype=np.int32)

    cdef:
        double[:, :] float_res_v = float_res
        int[:, :] int_res_v = int_res

    assert (
        distances_v.size == heights_v.size == zheights_v.size ==
        omega_v.size ==
        d_tm_v.size == d_lm_v.size == d_ct_v.size == d_cr_v.size ==
        zone_r_v.size
        )
    assert zone_t >= -1 and zone_t <= 11
    assert np.all(hprof_data['zone_r'] >= -1)
    assert np.all(hprof_data['zone_r'] <= 11)

    with nogil, parallel():

        pp = <ppstruct *> malloc(sizeof(ppstruct))
        if pp == NULL:
            abort()

        pp.version = version
        pp.freq = freq
        pp.wavelen = 0.299792458 / freq
        pp.temperature = temperature
        pp.pressure = pressure
        pp.zone_t = zone_t
        pp.h_tg = h_tg
        pp.h_rg = h_rg
        pp.h_tg_in = h_tg
        pp.h_rg_in = h_rg

        pp.hprof_step = 30.  # dummy
        pp.time_percent = time_percent
        pp.polarization = polarization

        # for algorithmic reasons, it is not possible to calculated the
        # attens for the first 5 or so steps; start at index 6
        for i in prange(6, max_path_length, schedule='guided', chunksize=10):

            pp.zone_r = zone_r_v[i]

            if pp.zone_t == CLUTTER.UNKNOWN:
                pp.h_tg = h_tg
            else:
                pp.h_tg = f_max(clutter_data_v[pp.zone_t, 0], h_tg)

            if pp.zone_r == CLUTTER.UNKNOWN:
                pp.h_rg = h_rg
            else:
                pp.h_rg = f_max(clutter_data_v[pp.zone_r, 0], h_rg)

            pp.d_tm = d_tm_v[i]
            pp.d_lm = d_lm_v[i]
            pp.d_ct = d_ct_v[i]
            pp.d_cr = d_cr_v[i]
            pp.omega = omega_v[i]

            pp.delta_N = delta_N_v[i]
            pp.beta0 = beta0_v[i]
            pp.N0 = N0_v[i]

            pp.distance = distances_v[i]

            _process_path(
                pp,
                distances_v[0:i + 1],
                heights_v[0:i + 1],
                zheights_v[0:i + 1],
                )

            (
                L_b0p, L_bd, L_bs, L_ba, L_b, L_b_corr, L_dummy
                ) = _path_attenuation_complete(pp[0], G_t, G_r)

            float_res_v[0, i] = L_b0p
            float_res_v[1, i] = L_bd
            float_res_v[2, i] = L_bs
            float_res_v[3, i] = L_ba
            float_res_v[4, i] = L_b
            float_res_v[5, i] = L_b_corr
            float_res_v[6, i] = pp.eps_pt
            float_res_v[7, i] = pp.eps_pr
            float_res_v[8, i] = pp.d_lt
            float_res_v[9, i] = pp.d_lr

            int_res_v[0, i] = pp.path_type

        free(pp)

    return float_res, int_res


def losses_complete_cython(
        frequency,
        temperature,
        pressure,
        double lon_t, double lat_t,
        double lon_r, double lat_r,
        h_tg, h_rg,
        double hprof_step,
        time_percent,
        G_t=None, G_r=None,
        omega=None,
        d_tm=None, d_lm=None,
        d_ct=None, d_cr=None,
        zone_t=None, zone_r=None,
        polarization=0,
        version=16,
        # override if you don't want builtin method:
        delta_N=None, N0=None,
        # override if you don't want builtin method:
        hprof_dists=None,
        hprof_heights=None,
        hprof_bearing=None, hprof_backbearing=None,
        generic_heights=False,
        ):

    cdef:
        # work arrays (for inner loop of nditer)
        np.ndarray[double] _freq, _temp, _press
        np.ndarray[double] _h_tg, _h_rg, _G_t, _G_r
        np.ndarray[double] _time_percent, _omega
        np.ndarray[double] _d_tm, _d_lm, _d_ct, _d_cr
        np.ndarray[int] _zone_t, _zone_r, _polarization, _version

        # output arrays
        np.ndarray[double] _L_b0p, _L_bd, _L_bs, _L_ba, _L_b, _L_b_corr
        np.ndarray[double] _eps_pt, _eps_pr, _d_lt, _d_lr
        np.ndarray[int] _path_type

        # other

        np.ndarray[double] lons, lats, distances, heights, zheights
        double[::1] distances_v, heights_v, zheights_v
        double distance, bearing, back_bearing
        double lon_mid, lat_mid
        double _delta_N, _N0

        int hsize, mid_idx

        ppstruct *pp

        double[:, ::1] _clut_data = CLUTTER_DATA

        double L_dummy

        double *last_freq
        double *last_h_tg
        double *last_h_rg
        int *last_version
        int *last_zone_t
        int *last_zone_r

        int i, size

    assert np.all(time_percent <= 50.)
    assert np.all((version == 14) | (version == 16))

    assert np.all((zone_t >= -1) & (zone_t <= 11))
    assert np.all((zone_r >= -1) & (zone_r <= 11))

    assert (delta_N is None) == (N0 is None), (
        'delta_N and N0 must both be None or both be provided'
        )

    assert (
        (hprof_dists is None) == (hprof_heights is None) ==
        (hprof_bearing is None) == (hprof_backbearing is None)
        ), (
            'hprof_dists, hprof_heights, bearing, and back_bearing '
            'must all be None or all be provided'
            )

    if hprof_dists is None:
        (
            lons, lats, distance, distances, heights,
            bearing, back_bearing, _,
            ) = heightprofile._srtm_height_profile(
                lon_t, lat_t,
                lon_r, lat_r,
                hprof_step,
                generic_heights=generic_heights,
                )
    else:
        distances = hprof_dists.astype(np.float64, order='C', copy=False)
        heights = hprof_heights.astype(np.float64, order='C', copy=False)
        hsize = distances.size
        distance = distances[hsize - 1]
        bearing = hprof_bearing
        back_bearing = hprof_backbearing

    if len(distances) < 5:
        raise ValueError('Height profile must have at least 5 steps.')

    zheights = np.zeros_like(heights)

    distances_v = distances
    heights_v = heights
    zheights_v = zheights

    hsize = distances.size
    mid_idx = hsize // 2

    if hprof_dists is None:
        lon_mid = lons[mid_idx]
        lat_mid = lats[mid_idx]
    else:
        lon_mid = 0.5 * (lon_t + lon_r)
        lat_mid = 0.5 * (lat_t + lat_r)

    if delta_N is None:
        _delta_N, _N0 = helper._DN_N0_from_map(lon_mid, lat_mid)
    else:
        _delta_N, _N0 = delta_N, N0

    if G_t is None:
        G_t = np.array([0.])

    if G_r is None:
        G_r = np.array([0.])

    if d_tm is None:
        d_tm = np.array([distance])

    if d_lm is None:
        d_lm = np.array([distance])

    if d_ct is None:
        d_ct = np.array([50000.])

    if d_cr is None:
        d_cr = np.array([50000.])

    if omega is None:
        omega = np.array([0.])

    if zone_t is None:
        zone_t = np.array([-1])

    if zone_r is None:
        zone_r = np.array([-1])


    # in the nditer, we first put all entities that have impact on the
    # path geometry; this is to avoid unnecessary re-calculations

    # these entities are:
    # frequency, h_tg, h_rg, version, zone_t, zone_r

    it = np.nditer(
        [
            frequency, h_tg, h_rg, G_t, G_r, version, zone_t, zone_r,
            temperature, pressure, time_percent, omega,
            d_tm, d_lm, d_ct, d_cr, polarization,
            # L_b0p, L_bd, L_bs, L_ba, L_b, L_b_corr,
            None, None, None, None, None, None,
            # eps_pt, eps_pr, d_lt, d_lr, path_type
            None, None, None, None, None,
            ],
        flags=['external_loop', 'buffered', 'delay_bufalloc'],
        op_flags=[['readonly']] * 17 + [['readwrite', 'allocate']] * 11,
        op_dtypes=(
            ['float64'] * 5 + ['int32'] * 3 +
            ['float64'] * 8 + ['int32'] * 1 +
            ['float64'] * 10 + ['int32'] * 1
            ),
        )

    it.reset()
    for (
            _freq, _h_tg, _h_rg, _G_t, _G_r, _version, _zone_t, _zone_r,
            _temp, _press, _time_percent, _omega,
            _d_tm, _d_lm, _d_ct, _d_cr, _polarization,
            _L_b0p, _L_bd, _L_bs, _L_ba, _L_b, _L_b_corr,
            _eps_pt, _eps_pr, _d_lt, _d_lr, _path_type,
            ) in it:

        with nogil, parallel():

            # could not find a solution to make last_h_tg a thread-local
            # variable; using an array does the trick (but is ugly!!!)
            last_freq = <double *> malloc(sizeof(double))
            if last_freq == NULL:
                abort()
            last_h_tg = <double *> malloc(sizeof(double))
            if last_h_tg == NULL:
                abort()
            last_h_rg = <double *> malloc(sizeof(double))
            if last_h_rg == NULL:
                abort()
            last_version = <int *> malloc(sizeof(int))
            if last_version == NULL:
                abort()
            last_zone_t = <int *> malloc(sizeof(int))
            if last_zone_t == NULL:
                abort()
            last_zone_r = <int *> malloc(sizeof(int))
            if last_zone_r == NULL:
                abort()

            last_freq[0] = NAN
            last_h_tg[0] = NAN
            last_h_rg[0] = NAN
            last_version[0] = -1000
            last_zone_t[0] = -1000
            last_zone_r[0] = -1000

            pp = <ppstruct *> malloc(sizeof(ppstruct))
            if pp == NULL:
                abort()

            # pp.lon_t = lon_t
            # pp.lat_t = lat_t
            # pp.lon_r = lon_r
            # pp.lat_r = lat_r
            pp.lon_mid = lon_mid
            pp.lat_mid = lat_mid
            pp.hprof_step = hprof_step  # dummy
            pp.distance = distance
            pp.bearing = bearing
            pp.back_bearing = back_bearing
            pp.alpha_tr = bearing
            pp.alpha_rt = back_bearing
            pp.delta_N = _delta_N
            pp.N0 = _N0

            size = _freq.shape[0]

            for i in prange(size):

                if (
                        _freq[i] != last_freq[0] or
                        _h_tg[i] != last_h_tg[0] or
                        _h_rg[i] != last_h_rg[0] or
                        _version[i] != last_version[0] or
                        _zone_t[i] != last_zone_t[0] or
                        _zone_r[i] != last_zone_r[0]
                        ):
                    last_freq[0] = _freq[i]
                    last_h_tg[0] = _h_tg[i]
                    last_h_rg[0] = _h_rg[i]
                    last_version[0] = _version[i]
                    last_zone_t[0] = _zone_t[i]
                    last_zone_r[0] = _zone_r[i]

                    # need to re-process path geometry ...
                    pp.version = _version[i]
                    pp.freq = _freq[i]
                    pp.wavelen = 0.299792458 / pp.freq
                    pp.zone_t = _zone_t[i]
                    pp.zone_r = _zone_r[i]
                    if pp.zone_t == CLUTTER.UNKNOWN:
                        pp.h_tg = _h_tg[i]
                    else:
                        pp.h_tg = f_max(_clut_data[pp.zone_t, 0], _h_tg[i])

                    if pp.zone_r == CLUTTER.UNKNOWN:
                        pp.h_rg = _h_rg[i]
                    else:
                        pp.h_rg = f_max(_clut_data[pp.zone_r, 0], _h_rg[i])
                    pp.h_tg_in = _h_tg[i]
                    pp.h_rg_in = _h_rg[i]

                    _process_path(
                        pp,
                        distances_v,
                        heights_v,
                        zheights_v,
                        )

                pp.temperature = _temp[i]
                pp.pressure = _press[i]
                pp.d_tm = _d_tm[i]
                pp.d_lm = _d_lm[i]
                pp.d_ct = _d_ct[i]
                pp.d_cr = _d_cr[i]
                pp.time_percent = _time_percent[i]
                pp.polarization = _polarization[i]
                pp.omega = _omega[i]
                pp.beta0 = _beta_from_DN_N0(
                    pp.lat_mid, pp.delta_N, pp.N0, pp.d_tm, pp.d_lm
                    )

                (
                    _L_b0p[i],
                    _L_bd[i],
                    _L_bs[i],
                    _L_ba[i],
                    _L_b[i],
                    _L_b_corr[i],
                    L_dummy,
                    ) = _path_attenuation_complete(pp[0], _G_t[i], _G_r[i])

                _eps_pt[i] = pp.eps_pt
                _eps_pr[i] = pp.eps_pr
                _d_lt[i] = pp.d_lt
                _d_lr[i] = pp.d_lr
                _path_type[i] = pp.path_type

            free(last_freq)
            free(last_h_tg)
            free(last_h_rg)
            free(last_version)
            free(last_zone_t)
            free(last_zone_r)
            free(pp)

    out = it.operands[17:]
    return out


# ############################################################################
# Atmospheric attenuation (Annex 2)
# ############################################################################


cdef inline double _phi_helper(
        double r_p, double r_t, double phi0,
        double a, double b, double c, double d,
        ) nogil:

    return phi0 * r_p ** a * r_t ** b * exp(
        c * (1. - r_p) + d * (1. - r_t)
        )


cdef inline double _g_helper(
        double f, double f_i,
        ) nogil:

    return 1. + ((f - f_i) / (f + f_i)) ** 2


cdef inline double _eta_helper1(
        double f, double r_t,
        double a, double eta, double b, double c, double d
        ) nogil:
    # applies g_correction

    return (
        a * eta * exp(b * (1 - r_t)) /
        ((f - c) ** 2 + d * eta ** 2) *
        _g_helper(f, floor(c + 0.5))
        )


cdef inline double _eta_helper2(
        double f, double r_t,
        double a, double eta, double b, double c, double d
        ) nogil:

    return (
        a * eta * exp(b * (1 - r_t)) /
        ((f - c) ** 2 + d * eta ** 2)
        )


cdef (double, double) _specific_attenuation_annex2(
        double freq, double pressure, double rho_water, double temperature
        ) nogil:

    cdef:

        double r_p, r_t
        double atten_dry, atten_wet

        double xi1, xi2, xi3, xi4, xi5, xi6, xi7, delta
        double gamma54, gamma58, gamma60, gamma62, gamma64, gamma66

        double eta_1, eta_2

    r_p = pressure / 1013.
    r_t = 288. / (temperature - 0.15)

    if freq <= 54.:

        xi1 = _phi_helper(r_p, r_t, 1., 0.0717, -1.8132, 0.0156, -1.6515)
        xi2 = _phi_helper(r_p, r_t, 1., 0.5146, -4.6368, -0.1921, -5.7416)
        xi3 = _phi_helper(r_p, r_t, 1., 0.3414, -6.5851, 0.2130, -8.5854)

        atten_dry = freq ** 2 * r_p ** 2 * 1.e-3 * (
            7.2 * r_t ** 2.8 / (freq ** 2 + 0.34 * r_p ** 2 * r_t ** 1.6) +
            0.62 * xi3 / ((54. - freq) ** (1.16 * xi1) + 0.83 * xi2)
            )

    elif freq <= 60.:

        gamma54 = _phi_helper(
            r_p, r_t, 2.192, 1.8286, -1.9487, 0.4051, -2.8509
            )
        gamma58 = _phi_helper(
            r_p, r_t, 12.59, 1.0045, 3.5610, 0.1588, 1.2834
            )
        gamma60 = _phi_helper(
            r_p, r_t, 15., 0.9003, 4.1335, 0.0427, 1.6088
            )

        atten_dry = exp(
            log(gamma54) / 24. * (freq - 58.) * (freq - 60.) -
            log(gamma58) / 8. * (freq - 54.) * (freq - 60.) +
            log(gamma60) / 12. * (freq - 54.) * (freq - 58.)
            )

    elif freq <= 62.:

        gamma60 = _phi_helper(
            r_p, r_t, 15., 0.9003, 4.1335, 0.0427, 1.6088
            )
        gamma62 = _phi_helper(
            r_p, r_t, 14.28, 0.9886, 3.4176, 0.1827, 1.3429
            )

        atten_dry = gamma60 + (gamma62 - gamma60) * (freq - 60.) / 2.

    elif freq <= 66.:

        gamma62 = _phi_helper(
            r_p, r_t, 14.28, 0.9886, 3.4176, 0.1827, 1.3429
            )
        gamma64 = _phi_helper(
            r_p, r_t, 6.819, 1.4320, 0.6258, 0.3177, -0.5914
            )
        gamma66 = _phi_helper(
            r_p, r_t, 1.908, 2.0717, -4.1404, 0.4910, -4.8718
            )

        atten_dry = exp(
            log(gamma62) / 8. * (freq - 64.) * (freq - 66.) -
            log(gamma64) / 4. * (freq - 62.) * (freq - 66.) +
            log(gamma66) / 8. * (freq - 62.) * (freq - 64.)
            )
    elif freq <= 120.:

        xi4 = _phi_helper(r_p, r_t, 1., -0.0112, 0.0092, -0.1033, -0.0009)
        xi5 = _phi_helper(r_p, r_t, 1., 0.2705, -2.7192, -0.3016, -4.1033)
        xi6 = _phi_helper(r_p, r_t, 1., 0.2445, -5.9191, 0.0422, -8.0719)
        xi7 = _phi_helper(r_p, r_t, 1., -0.1833, 6.5589, -0.2402, 6.131)

        atten_dry = freq ** 2 * r_p ** 2 * 1.e-3 * (
            3.02e-4 * r_t ** 3.5 +
            0.283 * r_t ** 3.8 / (
                (freq - 118.75) ** 2 + 2.91 * r_p ** 2 * r_t ** 1.6
                ) +
            0.502 * xi6 * (1. - 0.0163 * xi7 * (freq - 66.)) / (
                (freq - 66.) ** (1.4346 * xi4) + 1.15 * xi5
                )
            )

    elif freq <= 350.:

        delta = _phi_helper(r_p, r_t, -0.00306, 3.211, -14.94, 1.583, -16.37)

        atten_dry = delta + freq ** 2 * r_p ** 3.5 * 1.e-3 * (
            3.02e-4 / (1. + 1.9e-5 * freq ** 1.5) +
            0.283 * r_t ** 0.3 / (
                (freq - 118.75) ** 2 + 2.91 * r_p ** 2 * r_t ** 1.6
                )
            )
    else:

        # annex2 model only valid up to 350. GHz
        atten_dry = NAN

    eta_1 = 0.955 * r_p * r_t ** 0.68 + 0.006 * rho_water
    eta_2 = 0.735 * r_p * r_t ** 0.5 + 0.0353 * r_t ** 4 * rho_water

    atten_wet = (
        _eta_helper1(freq, r_t, 3.98, eta_1, 2.23, 22.235, 9.42) +
        _eta_helper2(freq, r_t, 11.96, eta_1, 0.7, 183.31, 11.14) +
        _eta_helper2(freq, r_t, 0.081, eta_1, 6.44, 321.226, 6.29) +
        _eta_helper2(freq, r_t, 3.66, eta_1, 1.6, 325.153, 9.22) +
        _eta_helper2(freq, r_t, 25.37, eta_1, 1.09, 380, 0) +
        _eta_helper2(freq, r_t, 17.4, eta_1, 1.46, 448, 0) +
        _eta_helper1(freq, r_t, 844.6, eta_1, 0.17, 557, 0) +
        _eta_helper1(freq, r_t, 290., eta_1, 0.41, 752, 0) +
        _eta_helper1(freq, r_t, 83328., eta_2, 0.99, 1780, 0)
        )

    atten_wet *= freq ** 2 * r_t ** 2.5 * rho_water * 1.e-4

    return atten_dry, atten_wet


def specific_attenuation_annex2(
        double freq, double pressure, double rho_water, double temperature
        ):
    '''
    Calculate specific attenuation using a simplified algorithm
    (ITU-R P.676-10 Annex 2.1).

    Parameters
    ----------
    freq_grid - Frequencies (GHz)
    pressure - total air pressure (dry + wet) (hPa)
    rho_water - water vapor density (g / m^3)
    temperature - temperature (K)

    Returns
    -------
    dry_attenuation, wet_attenuation (dB / km)
    '''

    return _specific_attenuation_annex2(
        freq, pressure, rho_water, temperature
        )


cdef double true_angular_distance(
        double l1, double b1, double l2, double b2
        ) nogil:
    '''
    Calculate true angular distance between two points on the sphere.

    Parameters
    ----------
    l1, b1; l2, b2 : double
        Longitude and latitude (in deg) of two points on the sphere.

    Notes
    -----
    1. Based on Haversine formula. Good accuracy for distances < 2*pi
     See http://en.wikipedia.org/wiki/Haversine_formula.

    2. Cython-only to allow GIL-releasing.
    '''

    return 360. / M_PI * asin(sqrt(
        sin((b1 - b2) * M_PI / 360.) ** 2 +
        cos(b1 * M_PI / 180.) * cos(b2 * M_PI / 180.) *
        sin((l1 - l2) * M_PI / 360.) ** 2
        ))


