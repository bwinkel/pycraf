#!python
# -*- coding: utf-8 -*-
# cython: language_level=3
# cython: cdivision=True, boundscheck=True, wraparound=False
# cython: embedsignature=True

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

cimport cython
cimport numpy as np
from numpy cimport PyArray_MultiIter_DATA as Py_Iter_DATA
from libc.math cimport (
    exp, sqrt, fabs, M_PI, M_PI_2, NAN, sin, cos, tan, asin, acos, atan2, fmod
    )
import numpy as np

np.import_array()

# __all__ = ['inverse', 'direct']


cdef double DEG2RAD = M_PI / 180.
cdef double RAD2DEG = 180. / M_PI
cdef double EARTH_RADIUS = 6371.


A_N = np.zeros((1024, ), dtype=np.float64)
R_I = np.zeros((1024, ), dtype=np.float64)
ALPHA_N = np.zeros((1024, ), dtype=np.float64)
BETA_N = np.zeros((1024, ), dtype=np.float64)
DELTA_N = np.zeros((1024, ), dtype=np.float64)
H_I = np.zeros((1024, ), dtype=np.float64)
LAYER_IDX = np.zeros((1024, ), dtype=np.int32)


cdef double fix_arg(double arg) nogil:
    '''
    Ensure argument is in [-1., +1.] for arcsin, arccos functions.
    '''

    if arg < -1.:
        return -1.
    elif arg > 1.:
        return 1.
    else:
        return arg


cdef bint decide_propagation(
        double r_n, double d_n, double beta_n, double delta_n
        ) nogil:
    '''
    Decide if next propagation step is upwards (1) or downwards (0).
    '''

    cdef:
        double k = tan(M_PI_2 - beta_n - delta_n)
        double t0 = -r_n * k / 2 / (k ** 2 + 1)
        double t1 = (
            k ** 2 * r_n ** 2 /
            4 / (k ** 2 + 1) ** 2
            )
        double t2 = (r_n ** 2 - (r_n - d_n) ** 2) / (k ** 2 + 1)
        bint cond = (t1 >= t2) and (t0 + sqrt(t1 - t2) > 0)

    return not cond


cdef (double, double, double, double, double, double, bint) propagate_down(
        double r_n,
        double d_n,
        double alpha_n,
        double delta_n,
        double ref_index_hi,
        double ref_index_lo,
        double path_length,
        double max_path_length
        ) nogil:
    '''
    Calculate downwards-propagation path parameters.
    '''

    cdef:
        double tmp, a_n, beta_n
        bint break_imminent = 0

    tmp = (
        r_n ** 2 * cos(M_PI - alpha_n) ** 2 -
        d_n * (-d_n + 2 * r_n)
        )
    if tmp < 0:
        tmp = 0.
    a_n = r_n * cos(M_PI - alpha_n) - sqrt(tmp)

    path_length += a_n
    if path_length >= max_path_length:
        a_n -= path_length - max_path_length
        d_n = r_n - sqrt(
            r_n ** 2 + a_n ** 2 -
            2 * a_n * r_n * cos(M_PI - alpha_n)
            )
        break_imminent = 1

    if fabs(a_n - d_n) < 1.e-8:
        a_n = d_n
        beta_n = M_PI
    else:
        beta_n = M_PI - acos(fix_arg(
            (d_n * (-d_n + 2 * r_n) - a_n ** 2) /
            2. / (r_n - d_n) / a_n
            ))

    delta_n += alpha_n - beta_n
    if delta_n < 0.:
        delta_n = 0.
        beta_n = M_PI

    alpha_n = M_PI - asin(
        fix_arg(ref_index_hi / ref_index_lo * sin(beta_n))
        )

    return a_n, alpha_n, beta_n, delta_n, d_n, path_length, break_imminent


cdef (double, double, double, double, double, double, bint) propagate_up(
        double r_n,
        double d_n,
        double beta_n,
        double delta_n,
        double ref_index_hi,
        double ref_index_lo,
        double path_length,
        double max_path_length
        ) nogil:
    '''
    Calculate upwards-propagation path parameters.
    '''

    cdef:
        double a_n, alpha_n
        bint break_imminent = 0

    a_n = -r_n * cos(beta_n) + sqrt(
        r_n ** 2 * cos(beta_n) ** 2 + d_n * (d_n + 2 * r_n)
        )

    path_length += a_n
    if path_length >= max_path_length:
        a_n -= path_length - max_path_length
        d_n = -r_n + sqrt(
            r_n ** 2 + a_n ** 2 +
            2 * a_n * r_n * cos(beta_n)
            )
        break_imminent = 1

    alpha_n = M_PI - acos(fix_arg(
        (-a_n ** 2 - 2 * r_n * d_n - d_n ** 2) /
        2. / a_n / (r_n + d_n)
        ))

    delta_n += beta_n - alpha_n
    if delta_n < 0:
        delta_n = alpha_n = 0.

    beta_n = asin(
        fix_arg(ref_index_lo / ref_index_hi * sin(alpha_n))
        )

    return a_n, alpha_n, beta_n, delta_n, d_n, path_length, break_imminent


def path_helper_cython(
        int start_i,
        int max_i,
        double elev,  # deg
        double obs_alt,  # km
        double max_path_length,  # km
        double[::1] radii,
        double[::1] deltas,
        double[::1] ref_index,
        ):
    '''
    This works on a bunch of pre-allocated buffers (that are large enough
    for all cases) and returns copies of slices of these buffers.

    When parallelizing this, one will need one buffer per thread!
    '''

    cdef:
        int i, this_i, counter = 0
        bint break_imminent = 0
        bint is_space_path = 0  # path goes into space? (i.e. above max layer)
        bint first_iter = 1

        double path_length = 0
        double delta_n = 0
        double alpha_0 = DEG2RAD * (90. - elev)
        double beta_0 = alpha_0
        double alpha_n = alpha_0
        double beta_n = alpha_0
        double r_n, d_n, a_n
        double tmp

        bint cond = 0

        double[::1] _a_n = A_N
        double[::1] _r_i = R_I
        double[::1] _alpha_n = ALPHA_N
        double[::1] _delta_n = DELTA_N
        double[::1] _beta_n = BETA_N
        double[::1] _h_i = H_I
        int[::1] _idx = LAYER_IDX

    _a_n[counter] = 0.
    _r_i[counter] = EARTH_RADIUS + obs_alt
    _alpha_n[counter] = NAN
    _delta_n[counter] = delta_n
    _beta_n[counter] = beta_n
    _h_i[counter] = obs_alt
    _idx[counter] = start_i
    counter += 1

    i = start_i
    while i > 0 and i < max_i:

        if first_iter:
            r_n = _r_i[0]
            d_n = r_n - radii[i - 1]
            # print(i, radii[i], r_n, d_n)
            # first_iter = 0
        else:
            r_n = radii[i]
            d_n = deltas[i]

        cond = decide_propagation(r_n, d_n, beta_n, delta_n)

        if cond:

            if first_iter:
                d_n = radii[i] - r_n

            this_i = i
            (
                a_n, alpha_n, beta_n, delta_n, d_n,
                path_length, break_imminent
                ) = propagate_up(
                r_n, d_n, beta_n, delta_n,
                ref_index[i + 1], ref_index[i],
                path_length, max_path_length
                )

            r_i = r_n + d_n
            h_i = r_n + d_n - EARTH_RADIUS

            # update refraction every time
            refraction = -RAD2DEG * (beta_n + delta_n - beta_0)

            if first_iter:
                first_iter = 0
            else:
                i += 1

        else:

            this_i = i - 1
            if first_iter:
                first_iter = 0
            else:
                d_n = deltas[this_i]

            (
                a_n, alpha_n, beta_n, delta_n, d_n,
                path_length, break_imminent
                ) = propagate_down(
                r_n, d_n, alpha_n, delta_n,
                ref_index[i], ref_index[i - 1],
                path_length, max_path_length
                )

            r_i = r_n - d_n
            h_i = r_n - d_n - EARTH_RADIUS

            refraction = -RAD2DEG * (alpha_n + delta_n - alpha_0)

            i -= 1

        # print(r_n, d_n, a_n, alpha_n, beta_n, delta_n)
        if i == max_i:
            is_space_path = 1

        _a_n[counter] = a_n
        _r_i[counter] = r_i
        _alpha_n[counter] = alpha_n
        _delta_n[counter] = delta_n
        _beta_n[counter] = beta_n
        _h_i[counter] = h_i
        _idx[counter] = this_i

        counter += 1

        if break_imminent:
            break

    return (
        np.core.records.fromarrays(
            [
                A_N[:counter],
                R_I[:counter],
                ALPHA_N[:counter],
                BETA_N[:counter],
                DELTA_N[:counter],
                H_I[:counter],
                LAYER_IDX[:counter],
                ],
            names='a_n, r_i, alpha_n, beta_n, delta_n, h_i, layer_idx',
            formats='f8, f8, f8, f8, f8, f8, i4',
            ),
        refraction,
        is_space_path,
        )
