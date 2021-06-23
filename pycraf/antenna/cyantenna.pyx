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
cimport numpy as np
from numpy cimport uint16_t, float64_t
from cython.parallel import prange, parallel
from numpy cimport PyArray_MultiIter_DATA as Py_Iter_DATA
from libc.math cimport M_PI, NAN
from libc.math cimport (
    exp, sqrt, fabs, sin, cos, tan, asin, acos, atan2, fmod, log10
    )
import numpy as np


np.import_array()

# __all__ = ['inverse', 'direct']


UINT16 = np.dtype(np.uint16)
FLOAT64 = np.dtype(np.float64)


cdef double DEG2RAD = M_PI / 180.
cdef double RAD2DEG = 180. / M_PI
cdef double M_2PI = 2 * M_PI


cdef inline float64_t _ras_pattern(
        float64_t phi, float64_t d_wlen,
        float64_t gmax, float64_t g1,
        float64_t phi_m, float64_t phi_r,
        ) nogil:

    if (120. <= phi) and (phi <= 180.):
        return -12.
    elif (80. <= phi) and (phi < 120.):
        return -7.
    elif (34.1 <= phi) and (phi < 80.):
        return -12.
    elif (10. <= phi) and (phi < 34.1):
        return 34. - 30. * log10(phi)
    elif (phi_r <= phi) and (phi < 10.):
        return 29. - 25. * log10(phi)
    elif (phi_m <= phi) and (phi < phi_r):
        return g1
    elif (0 <= phi) and (phi < phi_m):
        return gmax - 2.5e-3 * (d_wlen * phi) ** 2
    else:
        # huh?
        return 0.


def ras_pattern_cython(
        phi, d_wlen, gmax, g1, phi_m, phi_r,
        gain=None,
        ):
    '''
    Parallelized RAS pattern (non-bessel part only).
    '''

    cdef:

        # the memory view leads to an error:
        # ValueError: buffer source array is read-only
        # but new cython version should support it!?
        # double [::] _lon1_rad, _lat1_rad, _lon2_rad, _lat2_rad
        # double [::] _out_ang_dist
        np.ndarray[float64_t] _phi, _d_wlen, _gmax, _g1, _phi_m, _phi_r
        np.ndarray[float64_t] _gain

        int i, size

    it = np.nditer(
        [
            phi, d_wlen, gmax, g1, phi_m, phi_r,
            gain
            ],
        flags=['external_loop', 'buffered', 'delay_bufalloc'],
        op_flags=[
            ['readonly'], ['readonly'], ['readonly'], ['readonly'],
            ['readonly'], ['readonly'],
            ['readwrite', 'allocate'],
            ],
        op_dtypes=[
            FLOAT64, FLOAT64, FLOAT64, FLOAT64,
            FLOAT64, FLOAT64,
            FLOAT64,
            ]
        )

    # it would be better to use the context manager but
    # "with it:" requires numpy >= 1.14

    it.reset()

    for itup in it:
        _phi = itup[0]
        _d_wlen = itup[1]
        _gmax = itup[2]
        _g1 = itup[3]
        _phi_m = itup[4]
        _phi_r = itup[5]
        _gain = itup[6]

        size = _phi.shape[0]

        for i in prange(size, nogil=True):

            _gain[i] = _ras_pattern(
                _phi[i], _d_wlen[i], _gmax[i], _g1[i], _phi_m[i], _phi_r[i]
                )

    return it.operands[6]


cdef inline float64_t _A_EH(
        float64_t phi, float64_t A_m, float64_t phi_3db, float64_t k
        ) nogil:

    cdef float64_t gain = k * (phi / phi_3db) ** 2

    if gain < A_m:
        return -gain
    else:
        return -A_m


cdef inline float64_t _A_EV(
        float64_t theta, float64_t SLA_nu, float64_t theta_3db, float64_t k
        ) nogil:

    cdef float64_t gain = k * ((theta - 90.) / theta_3db) ** 2

    if gain < SLA_nu:
        return -gain
    else:
        return -SLA_nu


cdef inline float64_t  _imt2020_single_element_pattern(
        float64_t azim, float64_t elev,
        float64_t G_Emax,
        float64_t A_m, float64_t SLA_nu,
        float64_t phi_3db, float64_t theta_3db,
        float64_t k,
        ) nogil:

    cdef:
        float64_t phi = azim
        float64_t theta = 90. - elev
        float64_t _gain = (
            -_A_EH(phi, A_m, phi_3db, k) -
            _A_EV(theta, SLA_nu, theta_3db, k)
            )

    if _gain < A_m:
        return G_Emax - _gain
    else:
        return G_Emax - A_m


def imt2020_single_element_pattern_cython(
        azim, elev,
        G_Emax,
        A_m, SLA_nu,
        phi_3db, theta_3db,
        k=12.,
        gain=None,
        ):
    '''
    Parallelized IMT-2020 single element pattern.
    '''

    cdef:

        # the memory view leads to an error:
        # ValueError: buffer source array is read-only
        # but new cython version should support it!?
        # double [::] _lon1_rad, _lat1_rad, _lon2_rad, _lat2_rad
        # double [::] _out_ang_dist
        np.ndarray[float64_t] _azim, _elev, _G_Emax, _A_m, _SLA_nu
        np.ndarray[float64_t] _phi_3db, _theta_3db, _k
        np.ndarray[float64_t] _gain

        int i, size

    it = np.nditer(
        [
            azim, elev, G_Emax, A_m, SLA_nu, phi_3db, theta_3db, k,
            gain
            ],
        flags=['external_loop', 'buffered', 'delay_bufalloc'],
        op_flags=[
            ['readonly'], ['readonly'], ['readonly'], ['readonly'],
            ['readonly'], ['readonly'], ['readonly'], ['readonly'],
            ['readwrite', 'allocate'],
            ],
        op_dtypes=[
            FLOAT64, FLOAT64, FLOAT64, FLOAT64,
            FLOAT64, FLOAT64, FLOAT64, FLOAT64,
            FLOAT64,
            ]
        )

    # it would be better to use the context manager but
    # "with it:" requires numpy >= 1.14

    it.reset()

    for itup in it:
        _azim = itup[0]
        _elev = itup[1]
        _G_Emax = itup[2]
        _A_m = itup[3]
        _SLA_nu = itup[4]
        _phi_3db = itup[5]
        _theta_3db = itup[6]
        _k = itup[7]
        _gain = itup[8]

        size = _gain.shape[0]

        for i in prange(size, nogil=True):

            _gain[i] = _imt2020_single_element_pattern(
                _azim[i], _elev[i], _G_Emax[i], _A_m[i], _SLA_nu[i],
                _phi_3db[i], _theta_3db[i], _k[i],
                )

    return it.operands[8]


def imt2020_composite_pattern_cython(
        azim, elev,
        azim_i, elev_i,
        G_Emax,
        A_m, SLA_nu,
        phi_3db, theta_3db,
        d_H, d_V,
        N_H, N_V,
        rho,
        k=12.,
        gain=None,
        ):
    '''
    Parallelized IMT-2020 composite pattern.
    '''

    cdef:

        # the memory view leads to an error:
        # ValueError: buffer source array is read-only
        # but new cython version should support it!?
        # double [::] _lon1_rad, _lat1_rad, _lon2_rad, _lat2_rad
        # double [::] _out_ang_dist
        np.ndarray[float64_t] _A_E
        np.ndarray[uint16_t] _N_H, _N_V
        np.ndarray[float64_t] _dV_cos_theta, _dH_sin_theta_sin_phi
        np.ndarray[float64_t] _dV_sin_theta_i, _dH_cos_theta_i_sin_phi_i
        np.ndarray[float64_t] _gain
        float64_t _exp_arg, _gain_re, _gain_im

        int i, size, m, n

    # pre-compute some quantities

    A_E = imt2020_single_element_pattern_cython(
        azim, elev,
        G_Emax,
        A_m, SLA_nu,
        phi_3db, theta_3db,
        k=k,
        )

    phi = azim
    theta = 90. - elev
    phi_i = azim_i
    theta_i = -elev_i  # sic! (tilt angle in imt.model is -elevation)

    dV_cos_theta = d_V * np.cos(np.radians(theta))
    dH_sin_theta_sin_phi = (
        d_H * np.sin(np.radians(theta)) * np.sin(np.radians(phi))
        )

    dV_sin_theta_i = d_V * np.sin(np.radians(theta_i))
    dH_cos_theta_i_sin_phi_i = (
        d_H * np.cos(np.radians(theta_i)) * np.sin(np.radians(phi_i))
        )

    it = np.nditer(
        [
            A_E,
            dV_cos_theta, dH_sin_theta_sin_phi,
            dV_sin_theta_i, dH_cos_theta_i_sin_phi_i,
            N_H, N_V,
            gain,
            ],
        flags=['external_loop', 'buffered', 'delay_bufalloc'],
        op_flags=[
            ['readonly'], ['readonly'], ['readonly'], ['readonly'],
            ['readonly'], ['readonly'], ['readonly'],
            ['readwrite', 'allocate'],
            ],
        op_dtypes=[
            FLOAT64,
            FLOAT64, FLOAT64, FLOAT64, FLOAT64,
            UINT16, UINT16,
            FLOAT64,
            ]
        )

    # it would be better to use the context manager but
    # "with it:" requires numpy >= 1.14

    it.reset()

    for itup in it:

        _A_E = itup[0]
        _dV_cos_theta = itup[1]
        _dH_sin_theta_sin_phi = itup[2]
        _dV_sin_theta_i = itup[3]
        _dH_cos_theta_i_sin_phi_i = itup[4]
        _N_H = itup[5]
        _N_V = itup[6]
        _gain = itup[7]

        size = _gain.shape[0]

        for i in prange(size, nogil=True):

            _gain_re = 0.
            _gain_im = 0.
            for m in range(_N_H[i]):
                for n in range(_N_V[i]):
                    _exp_arg = 2. * M_PI * (
                        n * _dV_cos_theta[i] +
                        m * _dH_sin_theta_sin_phi[i] +
                        n * _dV_sin_theta_i[i] -
                        m * _dH_cos_theta_i_sin_phi_i[i]
                        )
                    # Note: inplace operator doesn't work; otherwise, Cython
                    # will assume that "_gain_re" is a shared reduction
                    # variable inside of the prange() loop.
                    _gain_re = _gain_re + cos(_exp_arg)
                    _gain_im = _gain_im + sin(_exp_arg)

            _gain[i] = (
                (_gain_re ** 2 + _gain_im ** 2) / (_N_H[i] * _N_V[i])
                )

    return A_E + 10 * np.log10(1 + rho * (it.operands[7] - 1))


cdef float64_t  _G_hr(
        float64_t x_h, float64_t k_h, float64_t G180
        ) nogil:

    cdef:

        float64_t lambda_kh = 3 * (1 - 0.5 ** -k_h)
        float64_t G = -12 * x_h ** 2

    if x_h > 0.5:
        G *= x_h ** -k_h
        G -= lambda_kh

    if G < G180:
        return G180
    else:
        return G


cdef float64_t  _G_vr(
        float64_t x_v, float64_t k_v, float64_t k_p,
        float64_t theta_3db, float64_t G180
        ) nogil:

    cdef:

        float64_t x_k = sqrt(1 - 0.36 * k_v)
        float64_t C = (
            10 * log10(
                (180. / theta_3db) ** 1.5 * (4 ** -1.5) / (1 + 8 * k_p)
                ) /
            log10(22.5 / theta_3db)
            )
        float64_t lambda_kv = 12 - C * log10(4.) - 10 * log10(4 ** -1.5 + k_v)

    if x_v < x_k:
        return -12 * x_v ** 2
    elif x_k <= x_v and x_v < 4:
        return -12 + 10 * log10(x_v ** -1.5 + k_v)
    elif 4 <= x_v and x_v < 90 / theta_3db:
        return -lambda_kv - C * log10(x_v)
    else:
        # x_v == 90 / theta_3db
        return G180


cdef float64_t _imt_advanced_sectoral_peak_sidelobe_pattern(
        float64_t azim, float64_t elev,
        float64_t G0, float64_t phi_3db, float64_t theta_3db,
        float64_t k_p, float64_t k_h, float64_t k_v,
        float64_t tilt_m, float64_t tilt_e,
        ) nogil:

    cdef:

        float64_t beta = 0.
        float64_t tmp = 0.

        float64_t azim_rot = 0.
        float64_t elev_rot = 0.

        float64_t x_h = 0.
        float64_t x_v = 0.

        float64_t G180 = 0.
        float64_t R = 0.

    if tilt_m != 0.:

        azim *= DEG2RAD
        elev *= DEG2RAD
        beta = tilt_m * DEG2RAD

        elev_rot = asin(
            sin(elev) * cos(beta) +
            cos(elev) * cos(azim) * sin(beta)
            )

        tmp = (
            -sin(elev) * sin(beta) +
            cos(elev) * cos(azim) * cos(beta)
            ) / cos(elev_rot)
        if tmp > 1.:
            tmp = 1.
        if tmp < -1.:
            tmp = -1.

        azim_rot = acos(tmp)

        azim = RAD2DEG * azim_rot
        elev = RAD2DEG * elev_rot

    if tilt_e != 0.:

        tmp = elev + tilt_e
        if tmp >= 0.:
            elev_rot = 90 * tmp / (90 + tilt_e)
        else:
            elev_rot = 90 * tmp / (90 - tilt_e)

        elev = elev_rot

    G180 = -12 + 10 * log10(1 + 8 * k_p) - 15 * log10(180. / theta_3db)

    x_h = fabs(azim) / phi_3db
    x_v = fabs(elev) / theta_3db

    R = (
        (_G_hr(x_h, k_h, G180) - _G_hr(180. / phi_3db, k_h, G180)) /
        (_G_hr(0., k_h, G180) - _G_hr(180. / phi_3db, k_h, G180))
        )

    return (
        G0 +
        _G_hr(x_h, k_h, G180) +
        R * _G_vr(x_v, k_v, k_p, theta_3db, G180)
        )


def imt_advanced_sectoral_peak_sidelobe_pattern_cython(
        azim, elev,
        G0, phi_3db, theta_3db,
        k_p, k_h, k_v,
        tilt_m, tilt_e,
        gain=None,
        ):
    '''
    Parallelized IMT advanced (LTE) antenna pattern (sectoral, peak side-lobe)
    '''

    cdef:

        np.ndarray[float64_t] _azim, _elev
        np.ndarray[float64_t] _G0, _phi_3db, _theta_3db
        np.ndarray[float64_t] _k_p, _k_h, _k_v
        np.ndarray[float64_t] _tilt_m, _tilt_e
        np.ndarray[float64_t] _gain

        int i, size

    it = np.nditer(
        [
            azim, elev,
            G0, phi_3db, theta_3db,
            k_p, k_h, k_v,
            tilt_m, tilt_e,
            gain
            ],
        flags=['external_loop', 'buffered', 'delay_bufalloc'],
        op_flags=[
            ['readonly'], ['readonly'], ['readonly'], ['readonly'],
            ['readonly'], ['readonly'], ['readonly'], ['readonly'],
            ['readonly'], ['readonly'],
            ['readwrite', 'allocate'],
            ],
        op_dtypes=[
            FLOAT64, FLOAT64, FLOAT64, FLOAT64,
            FLOAT64, FLOAT64, FLOAT64, FLOAT64,
            FLOAT64, FLOAT64,
            FLOAT64,
            ]
        )

    # it would be better to use the context manager but
    # "with it:" requires numpy >= 1.14

    it.reset()

    for itup in it:
        _azim = itup[0]
        _elev = itup[1]
        _G0 = itup[2]
        _phi_3db = itup[3]
        _theta_3db = itup[4]
        _k_p = itup[5]
        _k_h = itup[6]
        _k_v = itup[7]
        _tilt_m = itup[8]
        _tilt_e = itup[9]
        _gain = itup[10]

        size = _gain.shape[0]

        for i in prange(size, nogil=True):

            _gain[i] = _imt_advanced_sectoral_peak_sidelobe_pattern(
                _azim[i], _elev[i],
                _G0[i], _phi_3db[i], _theta_3db[i],
                _k_p[i], _k_h[i], _k_v[i],
                _tilt_m[i], _tilt_e[i],
                )

    return it.operands[10]


# d_wlen = diameter / wavelength
cdef float64_t _fl_pattern_2_1(
        float64_t phi, float64_t d_wlen, float64_t G_max
        ) nogil:

    cdef:

        float64_t g1 = 2. + 15. * log10(d_wlen)  # gain of first side-lobe
        float64_t phi_m = 20. / d_wlen * sqrt(G_max - g1)
        float64_t phi_r = 15.85 * d_wlen ** -0.6

    phi = fabs(phi)

    if 0 <= phi and phi < phi_m:
        return G_max - 2.5e-3 * (d_wlen * phi) ** 2
    elif phi_m <= phi and phi < phi_r:
        return g1
    elif phi_r <= phi and phi < 48.:
        return 32. - 25. * log10(phi)
    else:
        # 48. <= phi and phi <= 180.
        return -10.


cdef float64_t _fl_pattern_2_2(
        float64_t phi, float64_t d_wlen, float64_t G_max
        ) nogil:

    cdef:

        float64_t g1 = 2. + 15. * log10(d_wlen)  # gain of first side-lobe
        float64_t phi_m = 20. / d_wlen * sqrt(G_max - g1)
        float64_t phi_r = 15.85 * d_wlen ** -0.6

    phi = fabs(phi)

    if 0 <= phi and phi < phi_m:
        return G_max - 2.5e-3 * (d_wlen * phi) ** 2
    elif phi_m <= phi and phi < phi_r:
        return g1
    elif phi_r <= phi and phi < 48.:
        return 52 - 10 * log10(d_wlen) - 25 * log10(phi)
    else:
        # 48. <= phi and phi <= 180.
        return -10. - 10 * log10(d_wlen)


cdef float64_t _fl_pattern_2_3(
        float64_t phi, float64_t d_wlen, float64_t G_max
        ) nogil:

    cdef:

        float64_t g1 = 2. + 15. * log10(d_wlen)  # gain of first side-lobe
        float64_t phi_m = 20. / d_wlen * sqrt(G_max - g1)
        float64_t phi_t = 100. / d_wlen
        float64_t phi_s = 144.5 * d_wlen ** -0.2

    phi = fabs(phi)

    if 0 <= phi and phi < phi_m:
        return G_max - 2.5e-3 * (d_wlen * phi) ** 2
    elif phi_m <= phi and phi < phi_t:
        return g1
    elif phi_t <= phi and phi < phi_s:
        return 52 - 10 * log10(d_wlen) - 25 * log10(phi)
    else:
        # phi_s <= phi and phi <= 180.
        return -2. - 5 * log10(d_wlen)


cdef float64_t _fl_pattern(
        float64_t phi, float64_t diameter, float64_t wavelength,
        float64_t G_max
        ) nogil:

    cdef:
        float64_t d_wlen = diameter / wavelength
        float64_t gain = NAN

    if 0.00428 < wavelength and wavelength < 0.29979:  # 1...70 GHz
        if d_wlen > 100:
            return _fl_pattern_2_1(phi, d_wlen, G_max)
        else:
            return _fl_pattern_2_2(phi, d_wlen, G_max)
    elif 0.29979 <= wavelength and wavelength < 2.99792:  # 0.1...1 GHz
        return _fl_pattern_2_3(phi, d_wlen, G_max)

    return gain


def fl_pattern_cython(
        phi, diameter, wavelength, G_max,
        gain=None,
        ):
    '''
    Parallelized ITU-R Rec F.699 antenna pattern
    '''

    cdef:

        np.ndarray[float64_t] _phi, _diameter, _wavelength, _G_max
        np.ndarray[float64_t] _gain

        int i, size

    it = np.nditer(
        [
            phi, diameter, wavelength, G_max,
            gain
            ],
        flags=['external_loop', 'buffered', 'delay_bufalloc'],
        op_flags=[
            ['readonly'], ['readonly'], ['readonly'], ['readonly'],
            ['readwrite', 'allocate'],
            ],
        op_dtypes=[
            FLOAT64, FLOAT64, FLOAT64, FLOAT64,
            FLOAT64,
            ]
        )

    # it would be better to use the context manager but
    # "with it:" requires numpy >= 1.14

    it.reset()

    for itup in it:
        _phi = itup[0]
        _diameter = itup[1]
        _wavelength = itup[2]
        _G_max = itup[3]
        _gain = itup[4]

        size = _gain.shape[0]

        for i in prange(size, nogil=True):

            _gain[i] = _fl_pattern(
                _phi[i], _diameter[i], _wavelength[i], _G_max[i]
                )

    return it.operands[4]
