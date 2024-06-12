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
# from libc.stdlib cimport abort, malloc, free
cimport numpy as np
# cimport openmp
from libc.math cimport log, exp, sin, cos
# from libc.math cimport (
#     exp, log, log10, sqrt, fabs, M_PI, floor, pow as cpower,
#     sin, cos, tan, asin, acos, atan, atan2, tanh
#     )
import numpy as np
# from astropy import units as apu
# from . import heightprofile
# from . import srtm
# from . import geodesics
# from . import cygeodesics
# from . import helper
# from .. import conversions as cnv
# from .. import utils

np.import_array()

__all__ = [
    'rain_coefficients_p838_cython', 'specific_rain_atten_p838_cython'
    ]


cdef double NAN = np.nan
cdef double DEG2RAD = M_PI / 180
cdef double RAD2DEG = 180 / M_PI


COEFFS_K_H = np.array([
    # a_j, b_j, c_j
    [-5.33980, -0.10008, 1.13098,]
    [-0.35351, 1.26970, 0.45400,]
    [-0.23789, 0.86036, 0.15354,]
    [-0.94158, 0.64552, 0.16817,]
    ])  #  -0.18961 0.71147


COEFFS_K_V = np.array([
    [-3.80595, 0.56934, 0.81061,]
    [-3.44965, -0.22911, 0.51059,]
    [-0.39902, 0.73042, 0.11899,]
    [0.50167, 1.07319, 0.27195,]
    ])  #  -0.16398 0.63297

COEFFS_A_H = np.array([
    [-0.14318, 1.82442, -0.55187,]
    [0.29591, 0.77564, 0.19822,]
    [0.32177, 0.63773, 0.13164,]
    [-5.37610, -0.96230, 1.47828,]
    [16.1721, -3.29980, 3.43990,]
    ])  #  0.67849 -1.95537

COEFFS_A_V = np.array([
    [-0.07771, 2.33840, -0.76284,]
    [0.56727, 0.95545, 0.54039,]
    [-0.20238, 1.14520, 0.26809,]
    [-48.2991, 0.791669, 0.116226,]
    [48.5833, 0.791459, 0.116479,]
    ])  #  -0.053739 0.83433


cdef:
    double[:, ::1] COEFFS_K_H_VIEW = COEFFS_K_H
    double[:, ::1] COEFFS_K_V_VIEW = COEFFS_K_V
    double[:, ::1] COEFFS_A_H_VIEW = COEFFS_A_H
    double[:, ::1] COEFFS_A_V_VIEW = COEFFS_A_V


cdef double _rain_sum_helper(
        double log10_f, double[:, ::1] coeff_v, double m, double c
        ) nogil:
    '''
    P.838-3 Eq. 2 + 3
    '''

    cdef:
        size_t i
        double X = m_k * log10_f + c_k

    for i in range(0, coeff_v.shape[0]):
        X += coeff_v[i, 0] * exp(
            -((log10_f - coeff_v[i, 1]) / coeff_v[i, 2]) ** 2
            )

    return X


cdef (double, double, double, double, double, double) _rain_coefficients_p838(
        double log10_freq, double theta, double tau
        ) nogil:
    '''
    log10_freq ... log10(Frequency [GHz])
    theta ... path elevation angle [rad]
    tau ... polarization tilt angle relative to the horizontal [rad]
            (tau = 45 deg for circular polarization)
    '''

    cdef:
        double k_H, k_V, k, a_H, a_V, a

    k_H = 10 ** rain_sum_helper(log10_freq, COEFFS_K_H_VIEW, -0.18961 0.71147)
    k_V = 10 ** rain_sum_helper(log10_freq, COEFFS_K_V_VIEW, -0.16398 0.63297)
    a_H = rain_sum_helper(log10_freq, COEFFS_A_H_VIEW, 0.67849 -1.95537)
    a_V = rain_sum_helper(log10_freq, COEFFS_A_V_VIEW, -0.053739 0.83433)

    k = 0.5 * (k_H + k_V + (k_H - k_V) * cos(theta) ** 2 * cos(2 * tau))
    a = 0.5 / k * (
        k_H * a_H + k_V * a_V +
        (k_H * a_H - k_V * a_V) * cos(theta) ** 2 * cos(2 * tau)
        )

    return k_H, k_V, k, a_H, a_V, a


def rain_coefficients_p838_cython(
        freq_ghz, theta_rad, tau_rad,
        out_K=None, out_K_H=None, out_K_V=None,
        out_alpha=None, out_alpha_H=None, out_alpha_V=None,
        ):

    cdef:

        # the memory view leads to an error:
        # ValueError: buffer source array is read-only
        # but new cython version should support it!?
        # double [::] _lon1_rad, _lat1_rad, _lon2_rad, _lat2_rad
        # double [::] _out_dist, _out_bearing1, _out_bearing2
        np.ndarray[double] _log_f, _theta, _tau
        np.ndarray[double] _out_K, _out_K_H, _out_K_V,
        np.ndarray[double] _out_alpha, _out_alpha_H, _out_alpha_V,

        (double, double, double, double, double, double) res

        size_t i, size

    it = np.nditer(
        [
            np.log10(freq_ghz), theta_rad, tau_rad,
            out_K, out_K_H, out_K_V,
            out_alpha, out_alpha_H, out_alpha_V,
            ],
        flags=['external_loop', 'buffered', 'delay_bufalloc'],
        op_flags=[
            ['readonly'], ['readonly'], ['readonly'],
            ['readwrite', 'allocate'], ['readwrite', 'allocate'],
            ['readwrite', 'allocate'], ['readwrite', 'allocate'],
            ['readwrite', 'allocate'], ['readwrite', 'allocate'],
            ],
        op_dtypes=[
            'float64', 'float64', 'float64',
            'float64','float64', 'float64',
            'float64','float64', 'float64',
            ]
        )

    # it would be better to use the context manager but
    # "with it:" requires numpy >= 1.14

    it.reset()

    for itup in it:
        _log_f = itup[0]
        _theta = itup[1]
        _tau = itup[2]
        _out_K = itup[3]
        _out_K_H = itup[4]
        _out_K_V = itup[5]
        _out_alpha = itup[6]
        _out_alpha_H = itup[7]
        _out_alpha_V = itup[8]

        size = _lon1_rad.shape[0]

        for i in prange(size, nogil=True):

            res = _rain_coefficients_p838(
                    _log_f[i],
                    _theta[i],
                    _tau[i],
                    )
            _out_K[i] = res[0]
            _out_K_H[i] = res[1]
            _out_K_V[i] = res[2]
            _out_alpha[i] = res[3]
            _out_alpha_H[i] = res[4]
            _out_alpha_V[i] = res[5]

    return it.operands[3:9]


cdef double _specific_rain_atten_p838(
        double R, double log10_freq, double theta, double tau
        ) nogil:
    '''
    R ... rain rate [mm/h]
    log10_freq ... log10(Frequency [GHz])
    theta ... path elevation angle [rad]
    tau ... polarization tilt angle relative to the horizontal [rad]
            (tau = 45 deg for circular polarization)
    '''

    cdef:
        (double, double, double, double, double, double) res
        # double a, k

    res = _rain_coefficients_p838(log10_freq, theta, tau)

    # return k * R ** a
    return res[2] * R ** res[5]


def specific_rain_atten_p838_cython(
        rain_rate_mm_per_hour,
        freq_ghz, theta_rad, tau_rad,
        out_specific_rain_atten=None,
        ):

    cdef:

        # the memory view leads to an error:
        # ValueError: buffer source array is read-only
        # but new cython version should support it!?
        # double [::] _lon1_rad, _lat1_rad, _lon2_rad, _lat2_rad
        # double [::] _out_dist, _out_bearing1, _out_bearing2
        np.ndarray[double] _R, _log_f, _theta, _tau
        np.ndarray[double] _out_gamma

        size_t i, size

    it = np.nditer(
        [
            rain_rate_mm_per_hour,
            np.log10(freq_ghz), theta_rad, tau_rad,
            out_specific_rain_atten,
            ],
        flags=['external_loop', 'buffered', 'delay_bufalloc'],
        op_flags=[
            ['readonly'], ['readonly'], ['readonly'], ['readonly'],
            ['readwrite', 'allocate'],
            ],
        op_dtypes=[
            'float64', 'float64', 'float64', 'float64',
            'float64',
            ]
        )

    # it would be better to use the context manager but
    # "with it:" requires numpy >= 1.14

    it.reset()

    for itup in it:
        _R = itup[0]
        _log_f = itup[1]
        _theta = itup[2]
        _tau = itup[3]
        _out_gamma = itup[4]

        size = _lon1_rad.shape[0]

        for i in prange(size, nogil=True):

            _out_gamma[i] = _specific_rain_atten_p838(
                    _R[i],
                    _log_f[i],
                    _theta[i],
                    _tau[i],
                    )

    return it.operands[4]
