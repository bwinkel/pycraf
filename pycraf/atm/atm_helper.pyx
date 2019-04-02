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
from numpy cimport PyArray_MultiIter_DATA as Py_Iter_DATA
from libc.math cimport (
    exp, sqrt, fabs, M_PI, M_PI_2, NAN, sin, cos, tan, asin, acos, atan2, fmod
    )
import numpy as np

np.import_array()

__all__ = ['path_helper_cython', 'path_endpoint_cython']


cdef double DEG2RAD = M_PI / 180.
cdef double RAD2DEG = 180. / M_PI
cdef double EARTH_RADIUS = 6371.


MAX_COUNT = 2048  # need twice the number of layers at least
A_N = np.zeros((MAX_COUNT, ), dtype=np.float64)
R_N = np.zeros((MAX_COUNT, ), dtype=np.float64)
H_N = np.zeros((MAX_COUNT, ), dtype=np.float64)
X_N = np.zeros((MAX_COUNT, ), dtype=np.float64)
Y_N = np.zeros((MAX_COUNT, ), dtype=np.float64)
ALPHA_N = np.zeros((MAX_COUNT, ), dtype=np.float64)
BETA_N = np.zeros((MAX_COUNT, ), dtype=np.float64)
DELTA_N = np.zeros((MAX_COUNT, ), dtype=np.float64)
LAYER_IDX = np.zeros((MAX_COUNT, ), dtype=np.int32)
LAYER_EDGE_LEFT_IDX = np.zeros((MAX_COUNT, ), dtype=np.int32)
LAYER_EDGE_RIGHT_IDX = np.zeros((MAX_COUNT, ), dtype=np.int32)


cdef (double, double, double) crossing_point(
        double r_1, double r_2,
        double beta_n, double delta_n
        ) nogil:
    '''
    Find crossing point (having smallest delta_n > 0 step) for a ray from
    radius r_1 with slope pi/2 - beta_n - delta_n and circle of radius r_2
    '''

    cdef:
        # double k = tan(M_PI_2 - beta_n - delta_n)
        double k = tan(M_PI_2 - beta_n)
        double t0, t1, t2
        double x1 = NAN
        double x2 = NAN
        double y1, y2, delta_1, delta_2

    t0 = -r_1 * k / (k ** 2 + 1)
    t1 = t0 ** 2
    t2 = (r_1 ** 2 - r_2 ** 2) / (k ** 2 + 1)

    if t1 >= t2:
        x1 = t0 - sqrt(t1 - t2)
        x2 = t0 + sqrt(t1 - t2)

    y1 = k * x1 + r_1
    y2 = k * x2 + r_1  # r_1 is correct!

    x1, y1 = (
        cos(delta_n) * x1 + sin(delta_n) * y1,
        -sin(delta_n) * x1 + cos(delta_n) * y1
        )
    x2, y2 = (
        cos(delta_n) * x2 + sin(delta_n) * y2,
        -sin(delta_n) * x2 + cos(delta_n) * y2
        )

    delta_1 = atan2(x1, y1)
    delta_2 = atan2(x2, y2)

    if delta_1 > delta_n + 1.e-14:
        return x1, y1, delta_1
    elif delta_2 > delta_n + 1.e-14:
        return x2, y2, delta_2
    else:
        return NAN, NAN, M_PI


cdef (
    int, double, double, double, double, double, double, bint
    ) propagate_path(
        double r_n, double r_n_below, double r_n_above,
        double ref_n_m2, double ref_n_m1, double ref_n_p1, double ref_n_p2,
        double beta_n, double delta_n, double path_length,
        double x_old, double y_old,
        bint first_iter, double max_delta_n, double max_path_length
        ) nogil:
    '''
    Find next crossing point (having smallest delta_n > 0 step) with
    radius below, current radius, or radius above.

    As r_n is the current radius (defined by x_old, y_old, i.e., the latest
    cross point), we need 4 refractive indices, the two below the current
    radius (m2, m1) and the two avove (p1, p2). Note that in the array
    of refractive indices, ref[i] is the refractive index of the layer
    below radius[i].
    '''

    cdef:
        double x1, x2, x3, y1, y2, y3, delta_1, delta_2, delta_3
        double x, y, delta_n_new
        double v1_norm, v2_norm
        double ref_in, ref_out
        double a_n
        double m, n, t
        int delta_i  # -1, 0, +1
        bint do_break = 0

    if beta_n < 1.e-6:  # 0.2"
        delta_i = 1
        if first_iter:
            delta_i = 0
        x = sin(delta_n) * r_n_above
        y = cos(delta_n) * r_n_above
        delta_n_new = delta_n
        ref_in, ref_out = ref_n_p1, ref_n_p2

        # with gil:
        #     print('r_n', r_n, r_n_below, r_n_above)
        #     print('x, y, delta', x, y, delta_n_new)

    elif M_PI - beta_n < 1.e-6:
        delta_i = -1
        x = sin(delta_n) * r_n_below
        y = cos(delta_n) * r_n_below
        delta_n_new = delta_n
        ref_in, ref_out = ref_n_m1, ref_n_m2

        # with gil:
        #     print('r_n', r_n, r_n_below, r_n_above)
        #     print('x, y, delta', x, y, delta_n_new)

    else:
        x1, y1, delta_1 = crossing_point(r_n, r_n_below, beta_n, delta_n)

        if first_iter:
            x2, y2, delta_2 = NAN, NAN, M_PI
        else:
            x2, y2, delta_2 = crossing_point(r_n, r_n, beta_n, delta_n)

        x3, y3, delta_3 = crossing_point(r_n, r_n_above, beta_n, delta_n)

        # with gil:
        #     print('r_n', r_n, r_n_below, r_n_above)
        #     print('x1, y1, delta_1', x1, y1, delta_1)
        #     print('x2, y2, delta_2', x2, y2, delta_2)
        #     print('x3, y3, delta_3', x3, y3, delta_3)

        if delta_1 < delta_2 and delta_1 < delta_3:
            delta_i = -1
            x, y, delta_n_new = x1, y1, delta_1
            ref_in, ref_out = ref_n_m1, ref_n_m2
        elif delta_2 < delta_1 and delta_2 < delta_3:
            delta_i = 0
            if first_iter:
                delta_i = -1
            x, y, delta_n_new = x2, y2, delta_2
            ref_in, ref_out = ref_n_m1, ref_n_p1
        else:
            delta_i = 1
            if first_iter:
                delta_i = 0
            x, y, delta_n_new = x3, y3, delta_3
            ref_in, ref_out = ref_n_p1, ref_n_p2

    if delta_n_new > max_delta_n:
        m = (y - y_old) / (x - x_old)
        n = (y_old * x - y * x_old) / (x - x_old)
        t = tan(M_PI_2 - max_delta_n)
        x = n / (t - m)
        y = t * x
        delta_n_new = atan2(x, y)
        do_break = 1

    # calculate a_n
    a_n = sqrt((x_old - x) ** 2 + (y_old - y) ** 2)
    if path_length + a_n > max_path_length:
        a_n_s = max_path_length - path_length
        x = x_old + a_n_s / a_n * (x - x_old)
        y = y_old + a_n_s / a_n * (y - y_old)
        delta_n_new = atan2(x, y)
        a_n = a_n_s
        do_break = 1

    # calculate alpha_n
    v1_norm = sqrt(x ** 2 + y ** 2)
    v2_norm = sqrt((x - x_old) ** 2 + (y - y_old) ** 2)
    alpha_n = acos(
        x / v1_norm * (x - x_old) / v2_norm +
        y / v1_norm * (y - y_old) / v2_norm
        )

    if alpha_n > M_PI_2:
        beta_n = M_PI - asin(
            ref_in / ref_out * sin(M_PI - alpha_n)
            )
    else:
        # checking for critical angle
        if ref_in / ref_out * sin(alpha_n) > 1:
            beta_n = M_PI - alpha_n
        else:
            beta_n = asin(
                ref_in / ref_out * sin(alpha_n)
                )

    # with gil:
    #     print('alpha_n, beta_n', RAD2DEG * alpha_n, RAD2DEG * beta_n, RAD2DEG * (alpha_n - beta_n))
    #     print('alpha_n*, beta_n*', 90 - RAD2DEG * alpha_n, 90 - RAD2DEG * beta_n)
    #     print('ref_in, ref_out', ref_in, ref_out)

    return delta_i, x, y, delta_n_new, a_n, alpha_n, beta_n, do_break


def path_helper_cython(
        int start_i,
        int space_i,
        int max_i,
        double elev,  # deg
        double obs_alt,  # km
        double max_path_length,  # km
        double max_delta_n,  # deg
        double[::1] radii,
        double[::1] ref_index,
        ):
    '''
    This works on a bunch of pre-allocated buffers (that are large enough
    for all cases) and returns copies of slices of these buffers.

    When parallelizing this, one will need one buffer per thread!
    '''

    cdef:
        int i, di, counter = 0
        bint is_space_path = 0  # path goes into space? (i.e. above max layer)
        bint first_iter = 1
        bint do_break = 0

        double path_length = 0
        double delta_n = 0
        double max_delta_n_rad = DEG2RAD * max_delta_n
        double alpha_n = NAN
        double beta_0 = DEG2RAD * (90. - elev)
        double beta_n = beta_0
        double r_n = EARTH_RADIUS + obs_alt
        double h_n, a_n, x_n, y_n
        double refraction = 0.

        double[::1] _a_n = A_N
        double[::1] _r_n = R_N
        double[::1] _h_n = H_N
        double[::1] _x_n = X_N
        double[::1] _y_n = Y_N
        double[::1] _alpha_n = ALPHA_N
        double[::1] _delta_n = DELTA_N
        double[::1] _beta_n = BETA_N
        int[::1] _layer_idx = LAYER_IDX
        int[::1] _layer_edge_left_idx = LAYER_EDGE_LEFT_IDX
        int[::1] _layer_edge_right_idx = LAYER_EDGE_RIGHT_IDX

    # the first point is not related to anything, but it is still
    # useful to have it here (e.g., if one wants to plot the full path)
    # it must be neglected from attenuation/Tebb calculations
    _a_n[counter] = 0.
    _r_n[counter] = r_n
    _h_n[counter] = obs_alt
    _x_n[counter] = x_n = 0.
    _y_n[counter] = y_n = r_n
    _alpha_n[counter] = NAN
    _beta_n[counter] = NAN
    _delta_n[counter] = 0.
    _layer_idx[counter] = -1000
    _layer_edge_left_idx[counter] = -1000
    _layer_edge_right_idx[counter] = -1000
    counter += 1

    i = start_i
    while i > 0 and i < max_i:

        # beta_n is the path angle on the left
        _beta_n[counter] = beta_n

        if first_iter:

            (
                di, x_n, y_n, delta_n, a_n, alpha_n, beta_n, do_break
                ) = propagate_path(
                r_n, radii[i - 1], radii[i],
                ref_index[i - 1], ref_index[i],
                ref_index[i], ref_index[i + 1],
                beta_n, delta_n, path_length,
                x_n, y_n, first_iter,
                max_delta_n_rad, max_path_length,
                )
            _layer_edge_left_idx[counter] = -1000
            _layer_idx[counter] = i
            first_iter = 0

        else:

            (
                di, x_n, y_n, delta_n, a_n, alpha_n, beta_n, do_break
                ) = propagate_path(
                radii[i], radii[i - 1], radii[i + 1],
                ref_index[i - 1], ref_index[i],
                ref_index[i + 1], ref_index[i + 2],
                beta_n, delta_n, path_length,
                x_n, y_n, first_iter,
                max_delta_n_rad, max_path_length,
                )
            _layer_edge_left_idx[counter] = i
            # to determine the correct atm layer index, we need to
            # account for the type of propagation (up, down, same)
            # (mind that layer n is directly above layer_edge n)
            if di == 1:
                # up
                _layer_idx[counter] = i + 1
            elif di == 0:
                # same
                _layer_idx[counter] = i
            elif di == -1:
                # down
                _layer_idx[counter] = i
            else:
                raise RuntimeError('Something went wrong with raytracing')

        # print(counter, i, di, x_n, y_n, delta_n, a_n, alpha_n, beta_n)

        path_length += a_n
        r_n = sqrt(x_n ** 2 + y_n ** 2)
        h_n = r_n - EARTH_RADIUS

        _a_n[counter] = a_n
        # the following four numbers are the coordinates of the right
        # crossing point, as the left point is already in the list!
        _r_n[counter] = r_n
        _h_n[counter] = h_n
        _x_n[counter] = x_n
        _y_n[counter] = y_n
        # alpha_n is the angle on the right edge
        _alpha_n[counter] = alpha_n
        # _beta_n[counter] = beta_n
        # delta_n is the arc length of the sector
        _delta_n[counter] = delta_n
        _layer_edge_right_idx[counter] = i + di

        counter += 1

        if do_break:
            break
        else:
            i += di

        if i == space_i:
            is_space_path = 1

    refraction = -RAD2DEG * (beta_n + delta_n - beta_0)
    # _beta_n[counter - 1] = NAN

    return (
        np.core.records.fromarrays(
            [
                A_N[:counter],
                R_N[:counter],
                H_N[:counter],
                X_N[:counter],
                Y_N[:counter],
                ALPHA_N[:counter],
                BETA_N[:counter],
                DELTA_N[:counter],
                LAYER_IDX[:counter],
                LAYER_EDGE_LEFT_IDX[:counter],
                LAYER_EDGE_RIGHT_IDX[:counter],
                ],
            names=(
                'a_n, r_n, h_n, x_n, y_n, alpha_n, beta_n, delta_n, '
                'layer_idx, layer_edge_left_idx, layer_edge_right_idx'
                ),
            formats='f8, f8, f8, f8, f8, f8, f8, f8, i4, i4, i4',
            ),
        refraction,
        is_space_path,
        )


cpdef (
    double, double, double, double, double, double, double, int,
    double, int, double, bint
    ) path_endpoint_cython(
        int start_i,
        int space_i,
        int max_i,
        double elev,  # deg
        double obs_alt,  # km
        double max_path_length,  # km
        double max_delta_n,  # deg
        double[::1] radii,
        double[::1] ref_index,
        ):
    '''
    Minimal version of `path_helper_cython` that only calculates the endpoint.
    '''

    cdef:
        int i, di, this_i, nsteps = 0
        bint is_space_path = 0  # path goes into space? (i.e. above max layer)
        bint first_iter = 1
        bint do_break = 0

        double path_length = 0
        double delta_n = 0
        double max_delta_n_rad = DEG2RAD * max_delta_n
        double alpha_n = NAN
        double beta_0 = DEG2RAD * (90. - elev)
        double beta_n = beta_0
        double r_n = EARTH_RADIUS + obs_alt
        double h_n, a_n, x_n = 0., y_n = r_n
        double refraction = 0.

    i = start_i
    while i > 0 and i < max_i:

        if first_iter:

            (
                di, x_n, y_n, delta_n, a_n, alpha_n, beta_n, do_break
                ) = propagate_path(
                r_n, radii[i - 1], radii[i],
                ref_index[i - 1], ref_index[i],
                ref_index[i], ref_index[i + 1],
                beta_n, delta_n, path_length,
                x_n, y_n, first_iter,
                max_delta_n_rad, max_path_length,
                )
            first_iter = 0

        else:

            (
                di, x_n, y_n, delta_n, a_n, alpha_n, beta_n, do_break
                ) = propagate_path(
                radii[i], radii[i - 1], radii[i + 1],
                ref_index[i - 1], ref_index[i],
                ref_index[i + 1], ref_index[i + 2],
                beta_n, delta_n, path_length,
                x_n, y_n, first_iter,
                max_delta_n_rad, max_path_length,
                )

        # print(nsteps, i, di, x_n, y_n, delta_n, a_n, alpha_n, beta_n)

        nsteps += 1
        path_length += a_n
        r_n = sqrt(x_n ** 2 + y_n ** 2)
        h_n = r_n - EARTH_RADIUS
        this_i = i

        if do_break:
            break
        else:
            i += di

        if i == space_i:
            is_space_path = 1

    refraction = -RAD2DEG * (beta_n + delta_n - beta_0)

    return (
        a_n, r_n, h_n, x_n, y_n, alpha_n, delta_n, this_i,
        path_length, nsteps,
        refraction,
        is_space_path,
        )
