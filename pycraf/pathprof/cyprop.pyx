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
import numpy as np
from . import heightprofile
from . import helper

np.import_array()


__all__ = [
    'CLUTTER', 'CLUTTER_DATA', 'PathProp', 'set_num_threads',
    'specific_attenuation_annex2',
    'free_space_loss_bfsg_cython', 'tropospheric_scatter_loss_bs_cython',
    'ducting_loss_ba_cython', 'diffraction_loss_complete_cython',
    'path_attenuation_complete_cython', 'clutter_correction_cython',
    'atten_map_fast', 'height_profile_data', 'beta_from_DN_N0',
    ]


cdef double NAN = np.nan

cpdef enum CLUTTER:
    UNKNOWN = -1
    SPARSE = 0
    VILLAGE = 1
    DECIDIOUS_TREES = 2
    CONIFEROUS_TREES = 3
    TROPICAL_FOREST = 4
    SUBURBAN = 5
    DENSE_SUBURBAN = 6
    URBAN = 7
    DENSE_URBAN = 8
    HIGH_URBAN = 9
    INDUSTRIAL_ZONE = 10

CLUTTER_DATA = np.array(
    [
        [4., 0.1],
        [5., 0.07],
        [15., 0.05],
        [20., 0.05],
        [20., 0.03],
        [9., 0.025],
        [12., 0.02],
        [20., 0.02],
        [25., 0.02],
        [35., 0.02],
        [20., 0.05],
    ],
    dtype=np.float64)

cdef double[:, ::1] CLUTTER_DATA_V = CLUTTER_DATA

cdef object PARAMETERS_BASIC = [
    ('version', '12d', '(P.452 version; 14 or 16)'),
    ('freq', '12.6f', 'GHz'),
    ('wavelen', '12.6f', 'm'),
    ('polarization', '12d', '(0 - horizontal, 1 - vertical)'),
    ('temperature', '12.6f', 'K'),
    ('pressure', '12.6f', 'hPa'),
    ('time_percent', '12.6f', 'percent'),
    ('beta0', '12.6f', 'percent'),
    ('omega', '12.6f', 'percent'),
    ('lon_t', '12.6f', 'deg'),
    ('lat_t', '12.6f', 'deg'),
    ('lon_r', '12.6f', 'deg'),
    ('lat_r', '12.6f', 'deg'),
    ('lon_mid', '12.6f', 'deg'),
    ('lat_mid', '12.6f', 'deg'),
    ('delta_N', '12.6f', 'dimless / km'),
    ('N0', '12.6f', 'dimless'),
    ('distance', '12.6f', 'km'),
    ('bearing', '12.6f', 'deg'),
    ('back_bearing', '12.6f', 'deg'),
    ('hprof_step', '12.6f', 'm'),
    ('zone_t', '12d', ''),
    ('zone_r', '12d', ''),
    ('h_tg', '12.6f', 'm'),
    ('h_rg', '12.6f', 'm'),
    ('h_tg_in', '12.6f', 'm'),
    ('h_rg_in', '12.6f', 'm'),
    ('h0', '12.6f', 'm'),
    ('hn', '12.6f', 'm'),
    ('h_ts', '12.6f', 'm'),
    ('h_rs', '12.6f', 'm'),
    ('h_st', '12.6f', 'm'),
    ('h_sr', '12.6f', 'm'),
    ('h_std', '12.6f', 'm'),
    ('h_srd', '12.6f', 'm'),
    ('h_te', '12.6f', 'm'),
    ('h_re', '12.6f', 'm'),
    ('d_lm', '12.6f', 'km'),
    ('d_tm', '12.6f', 'km'),
    ('d_ct', '12.6f', 'km'),
    ('d_cr', '12.6f', 'km'),
    ('path_type', '12d', '(0 - LOS, 1 - transhoriz)'),
    ('theta_t', '12.6f', 'mrad'),
    ('theta_r', '12.6f', 'mrad'),
    ('alpha_tr', '12.6f', 'deg'),
    ('alpha_rt', '12.6f', 'deg'),
    ('eps_pt', '12.6f', 'deg'),
    ('eps_pr', '12.6f', 'deg'),
    ('theta', '12.6f', 'mrad'),
    ('d_lt', '12.6f', 'km'),
    ('d_lr', '12.6f', 'km'),
    ('h_m', '12.6f', 'm'),
    ('duct_slope', '12.6f', 'm / km'),
    ('a_e_50', '12.6f', 'km'),
    ('a_e_b0', '12.6f', 'km'),
    ]


cdef object PARAMETERS_V16 = [
    ('path_type_50', '12d', '(0 - LOS, 1 - transhoriz)'),
    ('nu_bull_50', '12.6f', 'dimless'),
    ('nu_bull_idx_50', '12d', 'dimless'),
    ('S_tim_50', '12.6f', 'm / km'),
    ('S_rim_50', '12.6f', 'm / km'),
    ('S_tr_50', '12.6f', 'm / km'),
    ('path_type_b0', '12d', '(0 - LOS, 1 - transhoriz)'),
    ('nu_bull_b0', '12.6f', 'dimless'),
    ('nu_bull_idx_b0', '12d', 'dimless'),
    ('S_tim_b0', '12.6f', 'm / km'),
    ('S_rim_b0', '12.6f', 'm / km'),
    ('S_tr_b0', '12.6f', 'm / km'),
    # ('a_e_zh_50', '12.6f', 'km'),
    ('path_type_zh_50', '12d', '(0 - LOS, 1 - transhoriz)'),
    ('nu_bull_zh_50', '12.6f', 'dimless'),
    ('nu_bull_idx_zh_50', '12d', 'dimless'),
    ('S_tim_zh_50', '12.6f', 'm / km'),
    ('S_rim_zh_50', '12.6f', 'm / km'),
    ('S_tr_zh_50', '12.6f', 'm / km'),
    # ('a_e_zh_b0', '12.6f', 'km'),
    ('path_type_zh_b0', '12d', '(0 - LOS, 1 - transhoriz)'),
    ('nu_bull_zh_b0', '12.6f', 'dimless'),
    ('nu_bull_idx_zh_b0', '12d', 'dimless'),
    ('S_tim_zh_b0', '12.6f', 'm / km'),
    ('S_rim_zh_b0', '12.6f', 'm / km'),
    ('S_tr_zh_b0', '12.6f', 'm / km'),
    ]


cdef object PARAMETERS_V14 = [
    ('zeta_m', '12.6f', 'dimless'),
    ('nu_m50', '12.6f', 'dimless'),
    ('nu_mbeta', '12.6f', 'dimless'),
    ('i_m50', '12d', 'dimless'),
    ('zeta_t', '12.6f', 'dimless'),
    ('nu_t50', '12.6f', 'dimless'),
    ('nu_tbeta', '12.6f', 'dimless'),
    ('i_t50', '12d', 'dimless'),
    ('zeta_r', '12.6f', 'dimless'),
    ('nu_r50', '12.6f', 'dimless'),
    ('nu_rbeta', '12.6f', 'dimless'),
    ('i_r50', '12d', 'dimless'),
    ]


cdef struct _PathProp:
    int version  # P.452 version (14 or 16)
    double freq  # GHz
    double wavelen  # m
    int polarization  # 0 - horizontal, 1 - vertical
    double temperature  # K
    double pressure  # hPa
    double time_percent  # percent
    double beta0  # percent
    double omega  # percent
    double lon_t  # deg
    double lat_t  # deg
    double lon_r  # deg
    double lat_r  # deg
    double lon_mid  # deg
    double lat_mid  # deg
    double delta_N  # dimless / km
    double N0  # dimless
    double distance  # km
    double bearing  # deg
    double back_bearing  # deg
    double hprof_step  # m
    int zone_t  #  clutter zone code
    int zone_r  #  clutter zone code
    double h_tg  # m; clutter height, if appropriate
    double h_rg  # m; clutter height, if appropriate
    double h_tg_in  # m
    double h_rg_in  # m
    double h0  # m
    double hn  # m
    double h_ts  # m
    double h_rs  # m
    double h_st  # m
    double h_sr  # m
    double h_std  # m
    double h_srd  # m
    double h_te  # m
    double h_re  # m
    double d_lm  # km
    double d_tm  # km
    double d_ct  # km
    double d_cr  # km
    int path_type  # 0 - LOS, 1 - transhoriz
    double theta_t  # mrad
    double theta_r  # mrad
    double alpha_tr  # deg == bearing
    double alpha_rt  # deg == backbearing
    double eps_pt  # deg, elevation angle of path at tx
    double eps_pr  # deg, elevation angle of path at rx
    double theta  # mrad
    double d_lt  # km
    double d_lr  # km
    double h_m  # m
    double duct_slope  # m / km
    double a_e_50  # km
    double a_e_b0  # km

    # V16 diffraction calculation parameters
    int path_type_50  # 0 - LOS, 1 - transhoriz
    double nu_bull_50  # dimless
    int nu_bull_idx_50  # dimless
    double S_tim_50  # m / km
    double S_rim_50  # m / km
    double S_tr_50  # m / km

    int path_type_b0  # 0 - LOS, 1 - transhoriz
    double nu_bull_b0  # dimless
    int nu_bull_idx_b0  # dimless
    double S_tim_b0  # m / km
    double S_rim_b0  # m / km
    double S_tr_b0  # m / km
    # double a_e_zh_50  # km

    int path_type_zh_50  # 0 - LOS, 1 - transhoriz
    double nu_bull_zh_50  # dimless
    int nu_bull_idx_zh_50  # dimless
    double S_tim_zh_50  # m / km
    double S_rim_zh_50  # m / km
    double S_tr_zh_50  # m / km
    # double a_e_zh_b0  # km

    int path_type_zh_b0  # 0 - LOS, 1 - transhoriz
    double nu_bull_zh_b0  # dimless
    int nu_bull_idx_zh_b0  # dimless
    double S_tim_zh_b0  # m / km
    double S_rim_zh_b0  # m / km
    double S_tr_zh_b0  # m / km

    # V14 diffraction calculation parameters
    double zeta_m  # dimless
    double nu_m50  # dimless
    double nu_mbeta  # dimless
    int i_m50  # dimless
    double zeta_t  # dimless
    double nu_t50  # dimless
    double nu_tbeta  # dimless
    int i_t50  # dimless
    double zeta_r  # dimless
    double nu_r50  # dimless
    double nu_rbeta  # dimless
    int i_r50  # dimless


def set_num_threads(int nthreads):
    '''
    Change maximum number of threads to use.

    This is a convenience function, to call omp_set_num_threads(),
    which is not possible during runtime from python.
    '''

    openmp.omp_set_num_threads(nthreads)


cdef inline double f_max(double a, double b) nogil:

    return a if a >= b else b

cdef inline double f_min(double a, double b) nogil:

    return a if a <= b else b


cdef class PathProp(object):
    '''
    Calculate path profile properties.

    Parameters
    ----------
    freq - Frequency of radiation [GHz]
    temperature - Temperature (K)
    pressure - Pressure (hPa)
    lon_t, lat_t - Transmitter coordinates [deg]
    lon_r, lat_r - Receiver coordinates [deg]
    h_tg, h_rg - Transmitter/receiver heights over ground [m]
    hprof_step - Distance resolution of height profile along path [m]
    time_percent - Time percentage [%] (maximal 50%)
    omega - Fraction of the path over water [%] (see Table 3)
    d_tm - longest continuous land (inland + coastal) section of the
        great-circle path [km]
    d_lm - longest continuous inland section of the great-circle path [km]
    d_ct, d_cr - Distance over land from transmit/receive antenna to the coast
        along great circle interference path [km]
        (set to zero for terminal on ship/sea platform; only relevant if less
        than 5 km)
    polarization - Polarization (0 - horizontal, 1 - vertical; default: 0)

    Returns
    -------
    Path Properties object
    '''

    cdef:
        _PathProp _pp

    def __init__(
            self,
            double freq,
            double temperature,
            double pressure,
            double lon_t, double lat_t,
            double lon_r, double lat_r,
            double h_tg, double h_rg,
            double hprof_step,
            double time_percent,
            double omega=0,
            double d_tm=-1, double d_lm=-1,
            double d_ct=50000, double d_cr=50000,
            int zone_t=CLUTTER.UNKNOWN, int zone_r=CLUTTER.UNKNOWN,
            int polarization=0,
            int version=16,
            tuple DN_N0=None,  # override if you don't want builtin method
            tuple hprofdata=None,  # override if you don't want builtin method
            double bearing=NAN, double back_bearing=NAN,
            ):

        assert time_percent <= 50.
        assert version == 14 or version == 16

        assert zone_t >= -1 and zone_t <= 11
        assert zone_r >= -1 and zone_r <= 11

        self._pp.version = version
        self._pp.freq = freq
        self._pp.wavelen = 0.299792458 / freq
        self._pp.temperature = temperature
        self._pp.pressure = pressure
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
        self._pp.time_percent = time_percent
        # TODO: add functionality to produce the following
        # five parameters programmatically (using some kind of Geo-Data)
        self._pp.omega = omega
        self._pp.d_tm = d_tm
        self._pp.d_lm = d_lm
        self._pp.d_ct = d_ct
        self._pp.d_cr = d_cr
        self._pp.polarization = polarization

        if hprofdata is None:
            (
                lons,
                lats,
                distances,
                heights,
                bearing,
                back_bearing,
                back_bearings,
                distance,
                ) = heightprofile._srtm_height_profile(
                    lon_t, lat_t,
                    lon_r, lat_r,
                    hprof_step
                    )
        else:
            distances, heights = hprofdata
            distances = distances.astype(np.float64, order='C', copy=False)
            heights = heights.astype(np.float64, order='C', copy=False)
            hsize = distances.size
            distance = distances[hsize - 1]

        zheights = np.zeros_like(heights)

        self._pp.distance = distance
        self._pp.bearing = bearing
        self._pp.back_bearing = back_bearing
        self._pp.alpha_tr = bearing
        self._pp.alpha_rt = back_bearing

        if self._pp.d_tm < 0:
            self._pp.d_tm = self._pp.distance
        if self._pp.d_lm < 0:
            self._pp.d_lm = self._pp.distance

        hsize = distances.size
        mid_idx = hsize // 2

        if hprofdata is None:
            self._pp.lon_mid = lons[mid_idx]
            self._pp.lat_mid = lats[mid_idx]
        else:
            self._pp.lon_mid = 0.5 * (lon_t + lon_r)
            self._pp.lat_mid = 0.5 * (lat_t + lat_r)

        # TODO: cythonize _radiomet_data_for_pathcenter
        if DN_N0 is None:
            delta_N, N0 = helper._N_from_map(
                self._pp.lon_mid, self._pp.lat_mid
                )
        else:
            delta_N, N0 = DN_N0

        self._pp.delta_N = delta_N
        self._pp.N0 = N0

        beta0 = _beta_from_DN_N0(
            self._pp.lat_mid,
            self._pp.delta_N, self._pp.N0,
            self._pp.d_tm, self._pp.d_lm
            )

        self._pp.beta0 = beta0

        _process_path(
            &self._pp,
            # lons,
            # lats,
            distances,
            heights,
            zheights,
            )

    def __repr__(self):

        return 'PathProp<Freq: {:.3f} GHz>'.format(self._pp.freq)

    def __str__(self):

        params = list(PARAMETERS_BASIC)  # make a copy
        if self._pp.version == 14:
            params += PARAMETERS_V14
        elif self._pp.version == 16:
            params += PARAMETERS_V16

        return '\n'.join(
            '{}: {{:{}}} {}'.format(
                '{:15s}', p[1], '{:10s}'
                ).format(
                p[0],
                getattr(self, p[0]), p[2]
                )
            for p in params
            )

    # How to do this programmatically?
    @property
    def version(self):
        return self._pp.version

    @property
    def freq(self):
        return self._pp.freq

    @property
    def polarization(self):
        return self._pp.polarization

    @property
    def temperature(self):
        return self._pp.temperature

    @property
    def pressure(self):
        return self._pp.pressure

    @property
    def wavelen(self):
        return self._pp.wavelen

    @property
    def time_percent(self):
        return self._pp.time_percent

    @property
    def beta0(self):
        return self._pp.beta0

    @property
    def omega(self):
        return self._pp.omega

    @property
    def lon_t(self):
        return self._pp.lon_t

    @property
    def lat_t(self):
        return self._pp.lat_t

    @property
    def lon_r(self):
        return self._pp.lon_r

    @property
    def lat_r(self):
        return self._pp.lat_r

    @property
    def lon_mid(self):
        return self._pp.lon_mid

    @property
    def lat_mid(self):
        return self._pp.lat_mid

    @property
    def delta_N(self):
        return self._pp.delta_N

    @property
    def N0(self):
        return self._pp.N0

    @property
    def distance(self):
        return self._pp.distance

    @property
    def bearing(self):
        return self._pp.bearing

    @property
    def back_bearing(self):
        return self._pp.back_bearing

    @property
    def hprof_step(self):
        return self._pp.hprof_step

    @property
    def h_tg(self):
        return self._pp.h_tg

    @property
    def h_rg(self):
        return self._pp.h_rg

    @property
    def h0(self):
        return self._pp.h0

    @property
    def hn(self):
        return self._pp.hn

    @property
    def h_ts(self):
        return self._pp.h_ts

    @property
    def h_rs(self):
        return self._pp.h_rs

    @property
    def h_st(self):
        return self._pp.h_st

    @property
    def h_sr(self):
        return self._pp.h_sr

    @property
    def h_std(self):
        return self._pp.h_std

    @property
    def h_srd(self):
        return self._pp.h_srd

    @property
    def h_te(self):
        return self._pp.h_te

    @property
    def h_re(self):
        return self._pp.h_re

    @property
    def d_lm(self):
        return self._pp.d_lm

    @property
    def d_tm(self):
        return self._pp.d_tm

    @property
    def d_ct(self):
        return self._pp.d_ct

    @property
    def d_cr(self):
        return self._pp.d_cr

    @property
    def path_type(self):
        return self._pp.path_type

    @property
    def theta_t(self):
        return self._pp.theta_t

    @property
    def theta_r(self):
        return self._pp.theta_r

    @property
    def alpha_tr(self):
        return self._pp.alpha_tr

    @property
    def alpha_rt(self):
        return self._pp.alpha_rt

    @property
    def eps_pt(self):
        return self._pp.eps_pt

    @property
    def eps_pr(self):
        return self._pp.eps_pr

    @property
    def theta(self):
        return self._pp.theta

    @property
    def d_lt(self):
        return self._pp.d_lt

    @property
    def d_lr(self):
        return self._pp.d_lr

    @property
    def h_m(self):
        return self._pp.h_m

    @property
    def a_e_50(self):
        return self._pp.a_e_50

    @property
    def duct_slope(self):
        return self._pp.duct_slope

    @property
    def path_type_50(self):
        return self._pp.path_type_50

    @property
    def nu_bull_50(self):
        return self._pp.nu_bull_50

    @property
    def nu_bull_idx_50(self):
        return self._pp.nu_bull_idx_50

    @property
    def S_tim_50(self):
        return self._pp.S_tim_50

    @property
    def S_rim_50(self):
        return self._pp.S_rim_50

    @property
    def S_tr_50(self):
        return self._pp.S_tr_50

    @property
    def a_e_b0(self):
        return self._pp.a_e_b0

    @property
    def path_type_b0(self):
        return self._pp.path_type_b0

    @property
    def nu_bull_b0(self):
        return self._pp.nu_bull_b0

    @property
    def nu_bull_idx_b0(self):
        return self._pp.nu_bull_idx_b0

    @property
    def S_tim_b0(self):
        return self._pp.S_tim_b0

    @property
    def S_rim_b0(self):
        return self._pp.S_rim_b0

    @property
    def S_tr_b0(self):
        return self._pp.S_tr_b0

    # @property
    # def a_e_zh_50(self):
    #     return self._pp.a_e_zh_50

    @property
    def path_type_zh_50(self):
        return self._pp.path_type_zh_50

    @property
    def nu_bull_zh_50(self):
        return self._pp.nu_bull_zh_50

    @property
    def nu_bull_idx_zh_50(self):
        return self._pp.nu_bull_idx_zh_50

    @property
    def S_tim_zh_50(self):
        return self._pp.S_tim_zh_50

    @property
    def S_rim_zh_50(self):
        return self._pp.S_rim_zh_50

    @property
    def S_tr_zh_50(self):
        return self._pp.S_tr_zh_50

    # @property
    # def a_e_zh_b0(self):
    #     return self._pp.a_e_zh_b0

    @property
    def path_type_zh_b0(self):
        return self._pp.path_type_zh_b0

    @property
    def nu_bull_zh_b0(self):
        return self._pp.nu_bull_zh_b0

    @property
    def nu_bull_idx_zh_b0(self):
        return self._pp.nu_bull_idx_zh_b0

    @property
    def S_tim_zh_b0(self):
        return self._pp.S_tim_zh_b0

    @property
    def S_rim_zh_b0(self):
        return self._pp.S_rim_zh_b0

    @property
    def S_tr_zh_b0(self):
        return self._pp.S_tr_zh_b0

    @property
    def zeta_m(self):
        return self._pp.zeta_m

    @property
    def nu_m50(self):
        return self._pp.nu_m50

    @property
    def nu_mbeta(self):
        return self._pp.nu_mbeta

    @property
    def i_m50(self):
        return self._pp.i_m50

    @property
    def zeta_t(self):
        return self._pp.zeta_t

    @property
    def nu_t50(self):
        return self._pp.nu_t50

    @property
    def nu_tbeta(self):
        return self._pp.nu_tbeta

    @property
    def i_t50(self):
        return self._pp.i_t50

    @property
    def zeta_r(self):
        return self._pp.zeta_r

    @property
    def nu_r50(self):
        return self._pp.nu_r50

    @property
    def nu_rbeta(self):
        return self._pp.nu_rbeta

    @property
    def i_r50(self):
        return self._pp.i_r50


# Doesn't work :-(
# cannot assign attributes to C-level extension types at runtime
# for p in PARAMETERS:
#     setattr(PathProp, p[0], property(lambda self: getattr(self._pp, p[0])))

# PathProp.freq2 = property(lambda self: self._pp.freq)

cdef double _beta_from_DN_N0(
        double lat_mid, double DN, double N0, double d_tm, double d_lm
        ) nogil:

    cdef:
        double tau, a, b, mu1, log_mu1, mu4, beta0

    tau = 1. - exp(-4.12e-4 * cpower(d_lm, 2.41))
    lat_mid = fabs(lat_mid)

    a = cpower(10, -d_tm / (16. - 6.6 * tau))
    b = cpower(10, -5 * (0.496 + 0.354 * tau))
    mu1 = cpower(a + b, 0.2)
    if mu1 > 1.:
        mu1 = 1.
    log_mu1 = log10(mu1)

    if lat_mid <= 70.:
        mu4 = cpower(10, (-0.935 + 0.0176 * lat_mid) * log_mu1)
    else:
        mu4 = cpower(10, 0.3 * log_mu1)

    if lat_mid <= 70.:
        beta_0 = cpower(10, -0.015 * lat_mid + 1.67) * mu1 * mu4
    else:
        beta_0 = 4.17 * mu1 * mu4

    return beta_0


def beta_from_DN_N0(
        double lat_mid, double DN, double N0, double d_tm, double d_lm
        ):
    '''
    Calculate radiometeorological data, beta0.

    Parameters
    ----------
    lat - path center coordinates [deg]
    delta_N - average radio-refractive index lapse-rate through the
            lowest 1 km of the atmosphere [N-units/km]
    N_0 - sea-level surface refractivity [N-units]
    d_tm - longest continuous land (inland + coastal) section of the
        great-circle path [km]
    d_lm - longest continuous inland section of the great-circle path [km]

    Returns
    -------
    beta_0 - the time percentage for which refractive index lapse-rates
        exceeding 100 N-units/km can be expected in the first 100 m
        of the lower atmosphere [%]

    Notes
    -----
    - ΔN and N_0 can be derived from digitized maps (shipped with P.452).
    - Radio-climaticzones can be queried from ITU Digitized World Map (IDWM).
      For many applications, it is probably the case, that only inland
      zones are present along the path of length d.
      In this case, set d_tm = d_lm = d.
    '''

    return _beta_from_DN_N0(
        lat_mid, DN, N0, d_tm, d_lm
        )


cdef void _process_path(
        _PathProp *pp,
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

    # smooth-earth height profile
    pp.h_st, pp.h_sr = _smooth_earth_heights(
        pp.distance, distances_view, heights_view,
        )

    # print('_smooth_earth_heights', time.time() - _time)
    # _time = time.time()

    # effective antenna heights for diffraction model
    pp.h_std, pp.h_srd = _effective_antenna_heights(
        pp.distance,
        distances_view, heights_view,
        pp.h_ts, pp.h_rs,
        pp.h_st, pp.h_sr
        )

    # print('_effective_antenna_heights', time.time() - _time)
    # _time = time.time()

    # parameters for ducting/layer-reflection model
    # (use these only for ducting or also for smooth-earth?)
    pp.h_st = min(pp.h_st, pp.h0)
    pp.h_sr = min(pp.h_sr, pp.hn)

    pp.duct_slope = (pp.h_sr - pp.h_st) / pp.distance

    pp.h_te = pp.h_tg + pp.h0 - pp.h_st
    pp.h_re = pp.h_rg + pp.hn - pp.h_sr

    pp.a_e_50 = 6371. * 157. / (157. - pp.delta_N)
    pp.a_e_b0 = 6371. * 3.

    if pp.version == 16:
        (
            pp.path_type_50, pp.nu_bull_50,
            pp.nu_bull_idx_50,
            pp.S_tim_50, pp.S_rim_50, pp.S_tr_50
            ) = _diffraction_helper_v16(
            pp.a_e_50, pp.distance,
            distances_view, heights_view,
            pp.h_ts, pp.h_rs,
            pp.wavelen,
            )

        (
            pp.path_type_b0, pp.nu_bull_b0,
            pp.nu_bull_idx_b0,
            pp.S_tim_b0, pp.S_rim_b0, pp.S_tr_b0
            ) = _diffraction_helper_v16(
            pp.a_e_b0, pp.distance,
            distances_view, heights_view,
            pp.h_ts, pp.h_rs,
            pp.wavelen,
            )

        # similarly, we have to repeat the game with heights set to zero

        (
            pp.path_type_zh_50, pp.nu_bull_zh_50,
            pp.nu_bull_idx_zh_50,
            pp.S_tim_zh_50, pp.S_rim_zh_50, pp.S_tr_zh_50
            ) = _diffraction_helper_v16(
            pp.a_e_50, pp.distance,
            distances_view, zheights_view,
            pp.h_ts - pp.h_std, pp.h_rs - pp.h_srd,
            pp.wavelen,
            )

        (
            pp.path_type_zh_b0, pp.nu_bull_zh_b0,
            pp.nu_bull_idx_zh_b0,
            pp.S_tim_zh_b0, pp.S_rim_zh_b0, pp.S_tr_zh_b0
            ) = _diffraction_helper_v16(
            pp.a_e_b0, pp.distance,
            distances_view, zheights_view,
            pp.h_ts - pp.h_std, pp.h_rs - pp.h_srd,
            pp.wavelen,
            )

    if pp.version == 14:

        (
            pp.zeta_m, pp.i_m50, pp.nu_m50,
            pp.nu_mbeta,
            pp.zeta_t, pp.i_t50, pp.nu_t50,
            pp.nu_tbeta,
            pp.zeta_r, pp.i_r50, pp.nu_r50,
            pp.nu_rbeta,
            ) = _diffraction_helper_v14(
            pp.a_e_50, pp.a_e_b0, pp.distance,
            distances_view, heights_view,
            pp.h_ts, pp.h_rs,
            pp.wavelen,
            )

    # print('_diffraction_helpers', time.time() - _time)
    # _time = time.time()

    # finally, determine remaining path geometry properties
    # note, this can depend on the bullington point (index) derived in
    # _diffraction_helper for 50%

    if pp.version == 14:
        diff_edge_idx = pp.i_m50
    elif pp.version == 16:
        diff_edge_idx = pp.nu_bull_idx_50

    (
        pp.path_type, pp.theta_t, pp.theta_r, pp.eps_pt, pp.eps_pr,
        pp.theta,
        pp.d_lt, pp.d_lr, pp.h_m
        ) = _path_geometry_helper(
        pp.a_e_50, pp.distance,
        distances_view, heights_view,
        pp.h_ts, pp.h_rs, pp.h_st,
        diff_edge_idx, pp.duct_slope,
        )

    # print('_path_geometry_helper', time.time() - _time)
    # _time = time.time()

    return


cdef (double, double) _smooth_earth_heights(
        double distance,
        double[::1] d_v,
        double[::1] h_v,
        ) nogil:

    cdef:
        int i, dsize
        double d = distance, nu_1, nu_2
        double h_st, h_sr

    dsize = d_v.shape[0]

    nu_1 = 0.
    nu_2 = 0.
    for i in range(1, dsize):

        nu_1 += (d_v[i] - d_v[i - 1]) * (h_v[i] + h_v[i - 1])
        nu_2 += (d_v[i] - d_v[i - 1]) * (
            h_v[i] * (2 * d_v[i] + d_v[i - 1]) +
            h_v[i - 1] * (d_v[i] + 2 * d_v[i - 1])
            )

    h_st = (2 * nu_1 * d - nu_2) / d ** 2
    h_sr = (nu_2 - nu_1 * d) / d ** 2

    return (h_st, h_sr)


cdef (double, double) _effective_antenna_heights(
        double distance,
        double[::1] d_v,
        double[::1] h_v,
        double h_ts, double h_rs,
        double h_st, double h_sr,
        ) nogil:

    cdef:
        int i, dsize
        double d = distance, h0, hn

        double H_i, h_obs = -1.e31, alpha_obt = -1.e31, alpha_obr = -1.e31
        double tmp_alpha_obt, tmp_alpha_obr

        double h_stp, h_srp, g_t, g_r
        double h_std, h_srd

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
        h_stp = h_st
        h_srp = h_sr
    else:
        g_t = alpha_obt / (alpha_obt + alpha_obr)
        g_r = alpha_obr / (alpha_obt + alpha_obr)

        h_stp = h_st - h_obs * g_t
        h_srp = h_sr - h_obs * g_r

    if h_stp > h0:
        h_std = h0
    else:
        h_std = h_stp

    if h_srp > hn:
        h_srd = hn
    else:
        h_srd = h_srp

    return (h_std, h_srd)


cdef (int, double, int, double, double, double) _diffraction_helper_v16(
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

        d_bp = (h_rs - h_ts + S_rim * d) / (S_tim + S_rim)

        nu_bull = (
            h_ts + S_tim * d_bp -
            (
                h_ts * (d - d_bp) + h_rs * d_bp
                ) / d
            ) * sqrt(
                0.002 * d / lam / d_bp / (d - d_bp)
                )  # == nu_b in Eq. 20
        nu_bull_idx = -1  # dummy value

    else:
        # LOS

        # find Bullington point, etc.

        S_rim = NAN

        # diffraction parameter
        for i in range(1, dsize - 1):
            nu_i = (
                h_v[i] +
                C_e500 * d_v[i] * (d - d_v[i]) -
                (h_ts * (d - d_v[i]) + h_rs * d_v[i]) / d
                ) * sqrt(
                    0.002 * d / lam / d_v[i] / (d - d_v[i])
                    )
            if nu_i > nu_bull:
                nu_bull = nu_i
                nu_bull_idx = i

    return (path_type, nu_bull, nu_bull_idx, S_tim, S_rim, S_tr)


cdef (
        double, int, double, double,
        double, int, double, double,
        double, int, double, double,
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

        double h50, d50, ht50, dt50, hr50, dr50

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
            nu_m50 = nu_i
            i_m50 = i

    if nu_m50 < -0.78:

        # every L will be zero
        return (
            zeta_m, i_m50, nu_m50, nu_mbeta,
            zeta_t, i_t50, nu_t50, nu_tbeta,
            zeta_r, i_r50, nu_r50, nu_rbeta,
            )

    h50 = h_v[i_m50]
    d50 = d_v[i_m50]

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
            zeta_m, i_m50, nu_m50, nu_mbeta,
            zeta_t, i_t50, nu_t50, nu_tbeta,
            zeta_r, i_r50, nu_r50, nu_rbeta,
            )


cdef (int, double, double, double, double, double, double, double, double) _path_geometry_helper(
        double a_e,
        double distance,
        double[::1] d_v,
        double[::1] h_v,
        double h_ts, double h_rs, double h_st,
        int nu_bull_idx, double duct_slope,
        ) nogil:

    cdef:
        int i, dsize
        double d = distance, m = duct_slope
        int path_type

        double theta_i, theta_j, theta_t, theta_r, theta
        double theta_i_max = -1.e31, theta_j_max = -1.e31, theta_td
        double eps_pt, eps_pr

        int lt_idx, lr_idx
        double d_lt, d_lr

        double h_m_i, h_m = -1.e31

    dsize = d_v.shape[0]

    for i in range(1, dsize - 1):

        theta_i = 1000. * atan(
            (h_v[i] - h_ts) / 1.e3 / d_v[i] - d_v[i] / 2. / a_e
            )
        if theta_i > theta_i_max:
            theta_i_max = theta_i
            lt_idx = i

    theta_td = 1000. * atan(
        (h_rs - h_ts) / 1.e3 / d - d / 2. / a_e
        )

    if theta_i_max > theta_td:
        path_type = 1
    else:
        path_type = 0

    if path_type == 1:
        # transhorizon

        theta_t = theta_i_max
        d_lt = d_v[lt_idx]

        for i in range(1, dsize - 1):
            theta_j = 1000. * atan(
                (h_v[i] - h_rs) / 1.e3 / (d - d_v[i]) -
                (d - d_v[i]) / 2. / a_e
                )
            if theta_j > theta_j_max:
                theta_j_max = theta_j
                lr_idx = i

        theta_r = theta_j_max
        d_lr = d - d_v[lr_idx]

        theta = 1.e3 * d / a_e + theta_t + theta_r

        # calculate elevation angles of path
        eps_pt = theta_t * 1.e-3 * 180. / M_PI
        eps_pr = theta_r * 1.e-3 * 180. / M_PI

        # calc h_m
        for i in range(lt_idx, lr_idx + 1):
            h_m_i = h_v[i] - (h_st + m * d_v[i])

            if h_m_i > h_m:
                h_m = h_m_i

    else:
        # LOS

        theta_t = theta_td

        theta_r = 1000. * atan(
            (h_ts - h_rs) / 1.e3 / d - d / 2. / a_e
            )

        theta = 1.e3 * d / a_e + theta_t + theta_r  # is this correct?

        # calculate elevation angles of path
        eps_pt = (
            (h_rs - h_ts) * 1.e-3 / d - d / 2. / a_e
            ) * 180. / M_PI
        eps_pr = (
            (h_ts - h_rs) * 1.e-3 / d - d / 2. / a_e
            ) * 180. / M_PI

        # horizon distance for LOS paths has to be set to distance to
        # Bullington point in diffraction method
        # assert (nu_bull_idx >= 0) and (nu_bull_idx < dsize)  # TODO
        d_lt = d_v[nu_bull_idx]
        d_lr = d - d_v[nu_bull_idx]

        # calc h_m
        # it seems, that h_m is calculated just from the profile height
        # at the Bullington point???
        h_m = h_v[nu_bull_idx] - (h_st + m * d_v[nu_bull_idx])

    return (
        path_type, theta_t, theta_r, eps_pt, eps_pr, theta, d_lt, d_lr, h_m
        )


cdef (double, double, double) _free_space_loss_bfsg_cython(
        _PathProp pp,
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


def free_space_loss_bfsg_cython(
        PathProp pathprop
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

    return _free_space_loss_bfsg_cython(pathprop._pp)


cdef double _tropospheric_scatter_loss_bs_cython(
        _PathProp pp, double G_t, double G_r
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
        PathProp pathprop, double G_t=0., double G_r=0.
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

    return _tropospheric_scatter_loss_bs_cython(pathprop._pp, G_t, G_r)


cdef double _ducting_loss_ba_cython(
        _PathProp pp
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
        PathProp pathprop
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

    return _ducting_loss_ba_cython(pathprop._pp)


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
        _PathProp pp,
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


cdef (double, double, double, double, double) _diffraction_loss_complete_cython(
        _PathProp pp,
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

    L_bfsg, E_sp, E_sbeta = _free_space_loss_bfsg_cython(pp)

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
        PathProp pathprop
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

    return _diffraction_loss_complete_cython(pathprop._pp)


cdef (double, double, double, double, double, double, double) _path_attenuation_complete_cython(
        _PathProp pp,
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
        A_ht = _clutter_correction_cython(
            pp.h_tg_in, pp.zone_t, pp.freq
            )
    if pp.zone_r != CLUTTER.UNKNOWN:
        A_hr = _clutter_correction_cython(
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
    L_bfsg, E_sp, E_sbeta = _free_space_loss_bfsg_cython(pp)
    L_b0p = L_bfsg + E_sp

    L_bs = _tropospheric_scatter_loss_bs_cython(pp, G_t, G_r)
    L_ba = _ducting_loss_ba_cython(pp)

    L_d_50, L_dp, L_bd_50, L_bd, L_min_b0p = _diffraction_loss_complete_cython(pp)

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

    return L_bfsg, L_bd, L_bs, L_ba, L_b, L_b_corr, L


def path_attenuation_complete_cython(
        PathProp pathprop, double G_t=0., double G_r=0.
        ):
    '''
    Calculate the Diffraction loss of a propagating radio
    wave according to ITU-R P.452-16 Eq. (14-44).

    Parameters
    ----------
    pathprop -
    G_t, G_r - Antenna gain (transmitter, receiver) in the direction of the
        horizon(!) along the great-circle interference path [dBi]

    Returns
    -------
    (L_bfsg, L_bd, L_bs, L_ba, L_b)
        L_bfsg - Free-space loss [dB]
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

    return _path_attenuation_complete_cython(pathprop._pp, G_t, G_r)


cdef double _clutter_correction_cython(
        double h_g, int zone, double freq
        ) nogil:

    cdef:

        double h_a, d_k
        double F_fc
        double A_h

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
    h_g - height above ground
    zone - Clutter category (see CLUTTER enum)

    Returns
    -------
    A_h - Clutter correction to path attenuation [dB]

    Notes
    -----
    - Path profile parameters (PathProps object) can be derived using the
        [TODO]
    '''

    return _clutter_correction_cython(h_g, zone, freq)


# ############################################################################
#
# Attenuation map making (fast)
#
# ############################################################################
#
# Idea: only calculate Geodesics/height profiles to map edges and apply hashes
#
# ############################################################################


def height_profile_data(
        double lon_t, double lat_t,
        double map_size_lon, double map_size_lat,
        double map_resolution=3. / 3600.,
        int do_cos_delta=1,
        ):

    '''
    Calculate height profiles and auxillary maps needed for atten_map_fast.

    This can be used to cache height-profile data. Since it is independent
    of frequency, time_percent, Tx and Rx heights, etc., one can re-use
    it to save computing time when doing batch jobs.
    '''

    cdef:

        # need 3x better resolution than map_resolution
        double hprof_step = map_resolution * 3600. / 1. * 30. / 3.

        int xi, yi, i
        int eidx, didx

    print('using hprof_step = {:.1f} m'.format(hprof_step))

    cosdelta = 1. / np.cos(np.radians(lat_t)) if do_cos_delta else 1.

    # construction map arrays
    xcoords = np.arange(
        lon_t - cosdelta * map_size_lon / 2,
        lon_t + cosdelta * map_size_lon / 2 + 1.e-6,
        cosdelta * map_resolution,
        )
    ycoords = np.arange(
        lat_t - map_size_lat / 2,
        lat_t + map_size_lat / 2 + 1.e-6,
        map_resolution,
        )
    print(
        xcoords[0], xcoords[len(xcoords) - 1],
        ycoords[0], ycoords[len(ycoords) - 1]
        )

    # use a 3x higher resolution version for edge coords for better accuracy
    xcoords_hi = np.arange(
        lon_t - cosdelta * map_size_lon / 2,
        lon_t + cosdelta * map_size_lon / 2 + 1.e-6,
        cosdelta * map_resolution / 3,
        )
    ycoords_hi = np.arange(
        lat_t - map_size_lat / 2,
        lat_t + map_size_lat / 2 + 1.e-6,
        map_resolution / 3,
        )


    # path_idx_map stores the index of the edge-path that is closest
    # to any given map pixel
    path_idx_map = np.zeros((len(ycoords), len(xcoords)), dtype=np.int32)

    # to define and find closest paths, we store the true angular distance
    # in pix_dist_map; Note, since distances are small, it is ok to do
    # this on the sphere (and not on geoid)
    pix_dist_map = np.ones((len(ycoords), len(xcoords)), dtype=np.float64)
    pix_dist_map *= 1.e30

    # dist_end_idx_map stores the (distance) index in the height profile
    # of the closest edge path, such that one can use a slice (0, end_idx)
    # to get a height profile approximately valid for any given pixel
    dist_end_idx_map = np.zeros((len(ycoords), len(xcoords)), dtype=np.int32)

    # store lon_mid, lat_mid,
    lon_mid_map = np.zeros((len(ycoords), len(xcoords)), dtype=np.float64)
    lat_mid_map = np.zeros((len(ycoords), len(xcoords)), dtype=np.float64)
    dist_map = np.zeros((len(ycoords), len(xcoords)), dtype=np.float64)

    # store bearings
    bearing_map = np.zeros((len(ycoords), len(xcoords)), dtype=np.float64)
    backbearing_map = np.zeros((len(ycoords), len(xcoords)), dtype=np.float64)

    # obtain all edge's height profiles
    edge_coords = list(zip(
        np.hstack([
            xcoords_hi,
            xcoords_hi[len(xcoords_hi) - 1] + 0. * xcoords_hi,
            xcoords_hi[::-1],
            xcoords_hi[0] + 0. * xcoords_hi,
            ]),
        np.hstack([
            ycoords_hi[0] + 0. * ycoords_hi,
            ycoords_hi,
            ycoords_hi[len(ycoords_hi) - 1] + 0. * ycoords_hi,
            ycoords_hi[::-1],
            ]),
        ))

    print('len(edge_coords)', len(edge_coords))

    refx, refy = xcoords[0], ycoords[0]
    cdef dict dist_dict = {}, height_dict = {}

    for eidx, (x, y) in enumerate(edge_coords):

        res = heightprofile._srtm_height_profile(
            lon_t, lat_t,
            x, y,
            hprof_step
            )
        (
            lons, lats, dists, heights,
            bearing, back_bearing, back_bearings, _
            ) = res
        dist_dict[eidx] = dists
        height_dict[eidx] = heights

        for didx, (lon_r, lat_r) in enumerate(zip(lons, lats)):

            # need to find closest pixel index in map
            xidx = int((lon_r - refx) / cosdelta / map_resolution + 0.5)
            yidx = int((lat_r - refy) / map_resolution + 0.5)

            if xidx < 0:
                xidx = 0
            if xidx >= len(xcoords):
                xidx = len(xcoords) - 1
            if yidx < 0:
                yidx = 0
            if yidx >= len(ycoords):
                yidx = len(ycoords) - 1

            pdist = true_angular_distance(
                xcoords[xidx], ycoords[yidx], lon_r, lat_r
                )

            if pdist < pix_dist_map[yidx, xidx]:
                pix_dist_map[yidx, xidx] = pdist
                path_idx_map[yidx, xidx] = eidx
                dist_end_idx_map[yidx, xidx] = didx
                mid_idx = didx // 2
                lon_mid_map[yidx, xidx] = lons[mid_idx]
                lat_mid_map[yidx, xidx] = lats[mid_idx]
                dist_map[yidx, xidx] = dists[didx]
                bearing_map[yidx, xidx] = bearing
                backbearing_map[yidx, xidx] = back_bearings[didx]

    # store delta_N, beta0, N0
    delta_N_map, beta0_map, N0_map = helper._radiomet_data_for_pathcenter(
        lon_mid_map, lat_mid_map, dist_map, dist_map
        )

    # dict access not possible with nogil
    # will store height profile dict in a 2D array, even though this
    # needs somewhat more memory

    # first, find max length that is needed:
    proflengths = np.array([
        len(dist_dict[k])
        for k in sorted(dist_dict.keys())
        ])
    maxlen_idx = np.argmax(proflengths)
    maxlen = proflengths[maxlen_idx]
    print('maxlen', maxlen)

    # we can re-use the distances vector, because of equal spacing
    dist_prof = dist_dict[maxlen_idx]
    height_profs = np.zeros(
        (len(height_dict), maxlen), dtype=np.float64
        )
    zheight_prof = np.zeros_like(dist_dict[maxlen_idx])

    cdef double[:, ::1] height_profs_v = height_profs

    for eidx, prof in height_dict.items():
        for i in range(len(prof)):
            height_profs_v[eidx, i] = prof[i]

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

    hprof_data['dist_prof'] = dist_prof
    hprof_data['height_profs'] = height_profs
    hprof_data['zheight_prof'] = zheight_prof

    return hprof_data


def atten_map_fast(
        double freq,
        double temperature,
        double pressure,
        double lon_t, double lat_t,
        double h_tg, double h_rg,
        double time_percent,
        object hprof_data=None,  # dict_like
        double map_size_lon=1., double map_size_lat=1.,
        double map_resolution=3. / 3600.,
        double omega=0,
        int zone_t=CLUTTER.UNKNOWN, int zone_r=CLUTTER.UNKNOWN,
        double d_tm=-1, double d_lm=-1,
        double d_ct=50000, double d_cr=50000,
        int polarization=0,
        int version=16,
        int do_cos_delta=1
        ):

    # TODO: implement map-based clutter handling; currently, only a single
    # clutter zone type is possible for each of Tx and Rx

    assert time_percent <= 50.
    assert version == 14 or version == 16

    cdef:
        _PathProp *pp
        double G_t = 0., G_r = 0.
        int xi, yi, xlen, ylen
        int eidx, didx

        double[:, ::1] clutter_data_v = CLUTTER_DATA

        double L_bfsg, L_bd, L_bs, L_ba, L_b, L_b_corr, L

    if hprof_data is None:

        hprof_data = height_profile_data(
            lon_t, lat_t,
            map_size_lon, map_size_lat,
            map_resolution, do_cos_delta
            )

    xcoords, ycoords = hprof_data['xcoords'], hprof_data['ycoords']

    # atten_map stores path attenuation
    atten_map = np.zeros((7, len(ycoords), len(xcoords)), dtype=np.float64)

    # also store path elevation angles as seen at Rx/Tx
    eps_pt_map = np.zeros((len(ycoords), len(xcoords)), dtype=np.float64)
    eps_pr_map = np.zeros((len(ycoords), len(xcoords)), dtype=np.float64)

    cdef:
        double[:, :, :] atten_map_v = atten_map
        double[:, :] eps_pt_map_v = eps_pt_map
        double[:, :] eps_pr_map_v = eps_pr_map

        # since we allow all dict_like objects for hprof_data,
        # we have to make sure, that arrays are numpy and contiguous
        # (have in mind, that one might use hdf5 data sets)

        _cf = np.ascontiguousarray

        double[::1] xcoords_v = _cf(hprof_data['xcoords'])
        double[::1] ycoords_v = _cf(hprof_data['ycoords'])
        double hprof_step = np.double(hprof_data['hprof_step'])

        int[:, :] path_idx_map_v = _cf(hprof_data['path_idx_map'])
        int[:, :] dist_end_idx_map_v = _cf(hprof_data['dist_end_idx_map'])
        double[:, :] lon_mid_map_v = _cf(hprof_data['lon_mid_map'])
        double[:, :] lat_mid_map_v = _cf(hprof_data['lat_mid_map'])
        double[:, :] dist_map_v = _cf(hprof_data['dist_map'])
        double[:, :] delta_N_map_v = _cf(hprof_data['delta_N_map'])
        double[:, :] beta0_map_v = _cf(hprof_data['beta0_map'])
        double[:, :] N0_map_v = _cf(hprof_data['N0_map'])

        double[:, :] bearing_map_v = _cf(hprof_data['bearing_map'])
        double[:, :] back_bearing_map_v = _cf(hprof_data['back_bearing_map'])

        double[::1] dist_prof_v = _cf(hprof_data['dist_prof'])
        double[:, ::1] height_profs_v = _cf(hprof_data['height_profs'])
        double[::1] zheight_prof_v = _cf(hprof_data['zheight_prof'])

        # double[::1] dists_v, heights_v, zheights_v

    xlen = len(xcoords)
    ylen = len(ycoords)

    with nogil, parallel():

        pp = <_PathProp *> malloc(sizeof(_PathProp))
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
        pp.zone_t = zone_t
        pp.zone_r = zone_r
        pp.h_tg_in = h_tg
        pp.h_rg_in = h_rg

        if zone_t == CLUTTER.UNKNOWN:
            pp.h_tg = h_tg
        else:
            pp.h_tg = f_max(clutter_data_v[zone_t, 0], h_tg)

        if zone_r == CLUTTER.UNKNOWN:
            pp.h_rg = h_rg
        else:
            pp.h_rg = f_max(clutter_data_v[zone_r, 0], h_rg)

        pp.hprof_step = hprof_step
        pp.time_percent = time_percent
        # TODO: add functionality to produce the following
        # five parameters programmatically (using some kind of Geo-Data)
        pp.omega = omega
        pp.d_tm = d_tm
        pp.d_lm = d_lm
        pp.d_ct = d_ct
        pp.d_cr = d_cr
        pp.polarization = polarization

        for yi in prange(ylen, schedule='guided', chunksize=10):

            for xi in range(xlen):

                eidx = path_idx_map_v[yi, xi]
                didx = dist_end_idx_map_v[yi, xi]

                if didx < 4:
                    continue

                pp.lon_r = xcoords_v[xi]
                pp.lat_r = ycoords_v[yi]

                # assigning not possible in prange, but can use directly below
                # dists_v = dist_prof_v[0:didx + 1]
                # heights_v = height_profs_v[eidx, 0:didx + 1]
                # zheights_v = zheight_prof_v[0:didx + 1]

                pp.distance = dist_map_v[yi, xi]
                pp.bearing = bearing_map_v[yi, xi]
                pp.back_bearing = back_bearing_map_v[yi, xi]

                pp.d_tm = pp.distance
                pp.d_lm = pp.distance

                pp.lon_mid = lon_mid_map_v[yi, xi]
                pp.lat_mid = lat_mid_map_v[yi, xi]

                pp.delta_N = delta_N_map_v[yi, xi]
                pp.beta0 = beta0_map_v[yi, xi]
                pp.N0 = N0_map_v[yi, xi]

                _process_path(
                    # &pp,
                    pp,
                    # dists_v,
                    # heights_v,
                    # zheights_v,
                    dist_prof_v[0:didx + 1],
                    height_profs_v[eidx, 0:didx + 1],
                    zheight_prof_v[0:didx + 1],
                    )

                (
                    L_bfsg, L_bd, L_bs, L_ba, L_b, L_b_corr, L
                    ) = _path_attenuation_complete_cython(pp[0], G_t, G_r)

                atten_map_v[0, yi, xi] = L_bfsg
                atten_map_v[1, yi, xi] = L_bd
                atten_map_v[2, yi, xi] = L_bs
                atten_map_v[3, yi, xi] = L_ba
                atten_map_v[4, yi, xi] = L_b
                atten_map_v[5, yi, xi] = L_b_corr
                atten_map_v[6, yi, xi] = L
                eps_pt_map_v[yi, xi] = pp.eps_pt
                eps_pr_map_v[yi, xi] = pp.eps_pr

        free(pp)

    return atten_map, eps_pt_map, eps_pr_map

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


