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
cimport numpy as np
from numpy cimport PyArray_MultiIter_DATA as Py_Iter_DATA
from libc.math cimport (
    exp, sqrt, fabs, M_PI, sin, cos, tan, asin, acos, atan2, fmod, log, log10
    )
import numpy as np

np.import_array()

# __all__ = ['inverse', 'direct']


cdef double NAN = np.NAN
cdef double DEG2RAD = M_PI / 180.
cdef double RAD2DEG = 180. / M_PI
cdef double M_2PI = 2 * M_PI
cdef double C = 3e8  # m/s


cdef inline double _min(double a, double b) nogil:
    if a < b:
        return a
    else:
        return b


cdef inline double _max(double a, double b) nogil:
    if a > b:
        return a
    else:
        return b


cdef (double, double) _rural_macro_losses(
        double fc_ghz, double d_2d,
        double h_bs, double h_ue,
        double W, double h,
        ) nogil:
    # all parameters (except freq) in meters;
    # see 3GPP TR 38.901 version 16.1.0 Release 16;
    # Table 7.4.1-1: Pathloss models; "RMa"

    cdef:
        double d_3d, d_bp, d_bp_3d
        double PL1, PL2, PL_los, PL_nlos_prime, PL_nlos

    d_3d = sqrt(d_2d ** 2 + (h_bs - h_ue) ** 2)
    d_bp = M_2PI * h_bs * h_ue * fc_ghz * 1e9 / C
    d_bp_3d = sqrt(d_bp ** 2 + (h_bs - h_ue) ** 2)

    if d_2d < 10 or d_2d > 10000:
        return (NAN, NAN)

    if d_2d <= d_bp:
        PL_los = PL1 = (
            20 * log10(40 * M_PI * d_3d * fc_ghz / 3) +
            _min(0.03 * h ** 1.72, 10) * log10(d_3d) -
            _min(0.044 * h ** 1.72, 14.77) +
            0.002 * log10(h) * d_3d
            )
    else:
        PL1 = (
                20 * log10(40 * M_PI * d_bp_3d * fc_ghz / 3) +
                _min(0.03 * h ** 1.72, 10) * log10(d_bp_3d) -
                _min(0.044 * h ** 1.72, 14.77) +
                0.002 * log10(h) * d_bp_3d
                )
        PL_los = PL2 = PL1 + 40 * log10(d_3d / d_bp)

    PL_nlos_prime = (
        161.04 - 7.1 * log10(W) + 7.5 * log10(h) -
        (24.37 - 3.7 * (h / h_bs) ** 2) * log10(h_bs) +
        (43.42 - 3.1 * log10(h_bs)) * (log10(d_3d) - 3) +
        20 * log10(fc_ghz) - (3.2 * (log10(11.75 * h_ue)) ** 2 - 4.97)
        )

    PL_nlos = _max(PL_los, PL_nlos_prime)

    return (PL_los, PL_nlos)


cdef (double, double) _urban_macro_losses(
        double fc_ghz, double d_2d,
        double h_bs, double h_ue,
        double h_e,
        ) nogil:
    # all parameters (except freq) in meters;
    # see 3GPP TR 38.901 version 16.1.0 Release 16;
    # Table 7.4.1-1: Pathloss models; "UMa"
    # Important: h_e parameter is random, but can be set to 1, if h_ue < 13 m
    # (see Note 1 in Table 7.4.1-1)

    cdef:
        double d_3d, d_bp_prime
        double PL1, PL2, PL_los, PL_nlos_prime, PL_nlos


    d_3d = sqrt(d_2d ** 2 + (h_bs - h_ue) ** 2)
    d_bp_prime = 4 *( h_bs - h_e) * (h_ue - h_e) * fc_ghz * 1e9 / C

    if d_2d < 10 or d_2d > 5000:
        return (NAN, NAN)

    if d_2d <= d_bp_prime:
        PL_los = PL1 = (
            28. + 22 * log10(d_3d) + 20 * log10(fc_ghz)
            )
    else:
        PL_los = PL2 = (
            28. + 40 * log10(d_3d) + 20 * log10(fc_ghz) -
            9 * log10(d_bp_prime ** 2 + (h_bs - h_ue) ** 2)
            )

    PL_nlos_prime = (
        13.54 + 39.08 * log10(d_3d) + 20 * log10(fc_ghz) - 0.6 * (h_ue - 1.5)
        )

    PL_nlos = _max(PL_los, PL_nlos_prime)

    return (PL_los, PL_nlos)


cdef (double, double) _urban_micro_losses(
        double fc_ghz, double d_2d,
        double h_bs, double h_ue,
        ) nogil:
    # all parameters (except freq) in meters;
    # see 3GPP TR 38.901 version 16.1.0 Release 16;
    # Table 7.4.1-1: Pathloss models; "UMi - Street Canyon"

    cdef:
        double d_3d, d_bp_prime
        double PL1, PL2, PL_los, PL_nlos_prime, PL_nlos
        double h_e = 1.  # m


    d_3d = sqrt(d_2d ** 2 + (h_bs - h_ue) ** 2)
    d_bp_prime = 4 *( h_bs - h_e) * (h_ue - h_e) * fc_ghz * 1e9 / C

    if d_2d < 10 or d_2d > 5000:
        return (NAN, NAN)

    if d_2d <= d_bp_prime:
        PL_los = PL1 = (
            32.4 + 21 * log10(d_3d) + 20 * log10(fc_ghz)
            )
    else:
        PL_los = PL2 = (
            32.4 + 40 * log10(d_3d) + 20 * log10(fc_ghz) -
            9.5 * log10(d_bp_prime ** 2 + (h_bs - h_ue) ** 2)
            )

    PL_nlos_prime = (
        22.4 + 35.3 * log10(d_3d) + 21.3 * log10(fc_ghz) - 0.3 * (h_ue - 1.5)
        )

    PL_nlos = _max(PL_los, PL_nlos_prime)

    return (PL_los, PL_nlos)


def rural_macro_losses_cython(
        fc_ghz, d_2d,
        h_bs, h_ue,
        W, h,
        out_PL_los=None,
        out_PL_nlos=None
        ):

    cdef:

        np.ndarray[double] _fc_ghz, _d_2d, _h_bs, _h_ue, _W, _h,
        np.ndarray[double] _out_PL_los, _out_PL_nlos

        int i, size

    it = np.nditer(
        [
            fc_ghz, d_2d, h_bs, h_ue, W, h,
            out_PL_los, out_PL_nlos
            ],
        flags=['external_loop', 'buffered', 'delay_bufalloc'],
        op_flags=[
            ['readonly'], ['readonly'], ['readonly'], ['readonly'],
            ['readonly'], ['readonly'],
            ['readwrite', 'allocate'], ['readwrite', 'allocate'],
            ],
        op_dtypes=[
            'float64', 'float64', 'float64', 'float64', 'float64', 'float64',
            'float64', 'float64'
            ]
        )

    # it would be better to use the context manager but
    # "with it:" requires numpy >= 1.14

    it.reset()

    for itup in it:
        _fc_ghz = itup[0]
        _d_2d = itup[1]
        _h_bs = itup[2]
        _h_ue = itup[3]
        _W = itup[4]
        _h = itup[5]
        _out_PL_los = itup[6]
        _out_PL_nlos = itup[7]

        size = _fc_ghz.shape[0]

        for i in prange(size, nogil=True):

            (
                _out_PL_los[i],
                _out_PL_nlos[i],
                ) = _rural_macro_losses(
                    _fc_ghz[i],
                    _d_2d[i],
                    _h_bs[i],
                    _h_ue[i],
                    _W[i],
                    _h[i],
                    )

    return it.operands[6:8]


def urban_macro_losses_cython(
        fc_ghz, d_2d,
        h_bs, h_ue,
        h_e,
        out_PL_los=None,
        out_PL_nlos=None
        ):

    cdef:

        np.ndarray[double] _fc_ghz, _d_2d, _h_bs, _h_ue, _h_e,
        np.ndarray[double] _out_PL_los, _out_PL_nlos

        int i, size

    it = np.nditer(
        [
            fc_ghz, d_2d, h_bs, h_ue, h_e,
            out_PL_los, out_PL_nlos
            ],
        flags=['external_loop', 'buffered', 'delay_bufalloc'],
        op_flags=[
            ['readonly'], ['readonly'], ['readonly'], ['readonly'],
            ['readonly'],
            ['readwrite', 'allocate'], ['readwrite', 'allocate'],
            ],
        op_dtypes=[
            'float64', 'float64', 'float64', 'float64', 'float64',
            'float64', 'float64'
            ]
        )

    # it would be better to use the context manager but
    # "with it:" requires numpy >= 1.14

    it.reset()

    for itup in it:
        _fc_ghz = itup[0]
        _d_2d = itup[1]
        _h_bs = itup[2]
        _h_ue = itup[3]
        _h_e = itup[4]
        _out_PL_los = itup[5]
        _out_PL_nlos = itup[6]

        size = _fc_ghz.shape[0]

        for i in prange(size, nogil=True):

            (
                _out_PL_los[i],
                _out_PL_nlos[i],
                ) = _urban_macro_losses(
                    _fc_ghz[i],
                    _d_2d[i],
                    _h_bs[i],
                    _h_ue[i],
                    _h_e[i],
                    )

    return it.operands[5:7]


def urban_micro_losses_cython(
        fc_ghz, d_2d,
        h_bs, h_ue,
        out_PL_los=None,
        out_PL_nlos=None
        ):

    cdef:

        np.ndarray[double] _fc_ghz, _d_2d, _h_bs, _h_ue, _h_e,
        np.ndarray[double] _out_PL_los, _out_PL_nlos

        int i, size

    it = np.nditer(
        [
            fc_ghz, d_2d, h_bs, h_ue,
            out_PL_los, out_PL_nlos
            ],
        flags=['external_loop', 'buffered', 'delay_bufalloc'],
        op_flags=[
            ['readonly'], ['readonly'], ['readonly'], ['readonly'],
            ['readwrite', 'allocate'], ['readwrite', 'allocate'],
            ],
        op_dtypes=[
            'float64', 'float64', 'float64', 'float64',
            'float64', 'float64'
            ]
        )

    # it would be better to use the context manager but
    # "with it:" requires numpy >= 1.14

    it.reset()

    for itup in it:
        _fc_ghz = itup[0]
        _d_2d = itup[1]
        _h_bs = itup[2]
        _h_ue = itup[3]
        _out_PL_los = itup[4]
        _out_PL_nlos = itup[5]

        size = _fc_ghz.shape[0]

        for i in prange(size, nogil=True):

            (
                _out_PL_los[i],
                _out_PL_nlos[i],
                ) = _urban_micro_losses(
                    _fc_ghz[i],
                    _d_2d[i],
                    _h_bs[i],
                    _h_ue[i],
                    )

    return it.operands[4:6]
