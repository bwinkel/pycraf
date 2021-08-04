#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

from astropy import units as apu
import numpy as np
from .cyantenna import imt2020_single_element_pattern_cython
from .cyantenna import imt2020_composite_pattern_cython
from .cyantenna import imt_advanced_sectoral_peak_sidelobe_pattern_cython
from .cyantenna import imt_advanced_sectoral_avg_sidelobe_pattern_cython
from .. import conversions as cnv
from .. import utils


__all__ = [
    'imt2020_single_element_pattern', 'imt2020_composite_pattern',
    'imt_advanced_sectoral_peak_sidelobe_pattern_400_to_6000_mhz',
    'imt_advanced_sectoral_avg_sidelobe_pattern_400_to_6000_mhz',
    ]


def _A_EH(phi, A_m, phi_3db, k=12.):
    '''
    Antenna element's horizontal pattern according to `IMT.MODEL
    <https://www.itu.int/md/R15-TG5.1-C-0036>`_ document.

    Parameters
    ----------
    phi : np.ndarray, float
        Azimuth [deg]
    A_m : np.ndarray, float
        Front-to-back ratio (horizontal) [dB]
    phi_3db : np.ndarray, float
        Horizontal 3-dB beam width of single element [deg]
    k : float, optional
        Multiplication factor, can be used to get better match to
        measured antenna patters (default: 12). See `WP5D-C-0936
    <https://www.itu.int/md/R15-WP5D-C-0936/en>`_ document.

    Returns
    -------
    A_EH : np.ndarray, float
        Antenna element's horizontal radiation pattern [dBi]
    '''

    return -np.minimum(k * (phi / phi_3db) ** 2, A_m)


def _A_EV(theta, SLA_nu, theta_3db, k=12.):
    '''
    Antenna element's vertical pattern according to `IMT.MODEL
    <https://www.itu.int/md/R15-TG5.1-C-0036>`_ document.

    Parameters
    ----------
    theta : np.ndarray, float
        Elevation [deg]
    SLA_nu : np.ndarray, float
        Front-to-back ratio (vertical) [dB]
    theta_3db : np.ndarray, float
        Vertical 3-dB beam width of single element [deg]
    k : float, optional
        Multiplication factor, can be used to get better match to
        measured antenna patterns (default: 12). See `WP5D-C-0936

    Returns
    -------
    A_EV : np.ndarray, float
        Antenna element's vertical radiation pattern [dBi]
    '''

    return -np.minimum(k * ((theta - 90.) / theta_3db) ** 2, SLA_nu)


@utils.ranged_quantity_input(
    azim=(-180, 180, apu.deg),
    elev=(-90, 90, apu.deg),
    G_Emax=(None, None, cnv.dB),
    A_m=(0, None, cnv.dB),
    SLA_nu=(0, None, cnv.dB),
    phi_3db=(0, None, apu.deg),
    theta_3db=(0, None, apu.deg),
    strip_input_units=True, output_unit=cnv.dBi
    )
def imt2020_single_element_pattern(
        azim, elev,
        G_Emax,
        A_m, SLA_nu,
        phi_3db, theta_3db,
        k=12.,
        ):
    '''
    Single antenna element's pattern according to `IMT.MODEL
    <https://www.itu.int/md/R15-TG5.1-C-0036>`_ document.

    Parameters
    ----------
    azim : `~astropy.units.Quantity`
        Azimuth [deg]
    elev : `~astropy.units.Quantity`
        Elevation [deg]
    G_Emax : `~astropy.units.Quantity`
        Single element maximum gain [dBi]
    A_m : `~astropy.units.Quantity`
        Front-to-back ratio (horizontal) [dB]
    SLA_nu : `~astropy.units.Quantity`
        Front-to-back ratio (vertical) [dB]
    phi_3db : `~astropy.units.Quantity`
        Horizontal 3dB beam width of single element [deg]
    theta_3db : `~astropy.units.Quantity`
        Vertical 3dB beam width of single element [deg]
    k : float, optional
        Multiplication factor, can be used to get better match to
        measured antenna patters (default: 12). See `WP5D-C-0936`

    Returns
    -------
    A_E : `~astropy.units.Quantity`
        Single antenna element's pattern [dBi]

    Notes
    -----
    Further information can be found in 3GPP TR 37.840 Section 5.4.4.
    '''

    return imt2020_single_element_pattern_cython(
        azim, elev,
        G_Emax,
        A_m, SLA_nu,
        phi_3db, theta_3db,
        k=k,
        )


@utils.ranged_quantity_input(
    azim=(-180, 180, apu.deg),
    elev=(-90, 90, apu.deg),
    azim_i=(-180, 180, apu.deg),
    elev_i=(-90, 90, apu.deg),
    G_Emax=(None, None, cnv.dB),
    A_m=(0, None, cnv.dB),
    SLA_nu=(0, None, cnv.dB),
    phi_3db=(0, None, apu.deg),
    theta_3db=(0, None, apu.deg),
    d_H=(0, None, cnv.dimless),
    d_V=(0, None, cnv.dimless),
    rho=(0, 1, cnv.dimless),
    strip_input_units=True, output_unit=cnv.dB
    )
def imt2020_composite_pattern(
        azim, elev,
        azim_i, elev_i,
        G_Emax,
        A_m, SLA_nu,
        phi_3db, theta_3db,
        d_H, d_V,
        N_H, N_V,
        rho=1 * cnv.dimless,
        k=12.,
        ):
    '''
    Composite (array) antenna pattern according to `IMT.MODEL
    <https://www.itu.int/md/R15-TG5.1-C-0036>`_ document.

    Parameters
    ----------
    azim, elev : `~astropy.units.Quantity`
        Azimuth/Elevation [deg]
    azim_i, elev_i : `~astropy.units.Quantity`
        Azimuthal/Elevational pointing of beam `i` [deg]
    G_Emax : `~astropy.units.Quantity`
        Single element maximum gain [dBi]
    A_m, SLA_nu : `~astropy.units.Quantity`
        Front-to-back ratio (horizontal/vertical) [dB]
    phi_3db, theta_3db : `~astropy.units.Quantity`
        Horizontal/Vertical 3dB beam width of single element [deg]
    d_H, d_V : `~astropy.units.Quantity`
        Horizontal/Vertical separation of beams in units of wavelength
        [dimless]
    N_H, N_V : int
        Horizontal/Vertical number of single antenna elements
    rho : `~astropy.units.Quantity`, optional
        Correlation level (see 3GPP TR 37.840, 5.4.4.1.4, default: 1) [dimless]
    k : float, optional
        Multiplication factor, can be used to get better match to
        measured antenna patters (default: 12). See `WP5D-C-0936`

    Returns
    -------
    A_A : `~astropy.units.Quantity`
        Composite (array) antenna pattern of beam `i` [dB]

    Notes
    -----
    Further information can be found in 3GPP TR 37.840 Section 5.4.4.

    According to document `WP5D-C-0936 <https://www.itu.int/md/R15-WP5D-C-0936/en>`_
    the AAS pattern can still be subject to quite effective beamforming
    in the spurious domain. For such cases, one can simply change the
    `d_H` and `d_V` to fit to the out-of-band frequency, i.e.,
    `d_oob = f_oob / f * d`. For example, if `f = 26 GHz`,
    `f_oob = 23.8 GHz`, and `d = 0.5` then `d_oob = 0.46`.
    However, to match measurements, also a different `k`-factor should
    be used, i.e., 8 instead of 12.
    '''

    return imt2020_composite_pattern_cython(
        azim, elev,
        azim_i, elev_i,
        G_Emax,
        A_m, SLA_nu,
        phi_3db, theta_3db,
        d_H, d_V,
        N_H, N_V,
        rho,
        k=k,
        )


def _G_hr(x_h, k_h, G180):
    '''
    Relative reference antenna gain in the azimuth plane at the normalized
    direction of `(x_h, 0)` according to `ITU-R Rec F.1336-4
    <https://www.itu.int/rec/R-REC-F.1336-4-201402-I/en>`_ Eq. (2b2).

    Parameters
    ----------
    x_h : np.ndarray, float
        Normalized azimuth [dimless]
    k_h : float
        Azimuth pattern adjustment factor based on leaked power
        (`0 ≤ kh ≤ 1`) [dimless]
    G180 : float
        Relative minimum gain according to Eq. (2b1) [dBi]

    Returns
    -------
    G_hr : np.ndarray, float
        Relative reference antenna gain in the azimuth plane [dBi]
    '''

    x_h = np.atleast_1d(x_h)
    lambda_kh = 3 * (1 - 0.5 ** -k_h)

    G = -12 * x_h ** 2
    mask = x_h > 0.5
    G[mask] *= x_h[mask] ** -k_h
    G[mask] -= lambda_kh

    G[G < G180] = G180

    return G


def _G_vr(x_v, k_v, k_p, theta_3db, G180):
    '''
    Relative reference antenna gain in the elevation plane at the normalized
    direction of `(0, x_v)` according to `ITU-R Rec F.1336-4
    <https://www.itu.int/rec/R-REC-F.1336-4-201402-I/en>`_ Eq. (2b3).

    Parameters
    ----------
    x_v : np.ndarray
        Normalized elevation [dimless]
    k_v : float
        Elevation pattern adjustment factor based on leaked power
        (`0 ≤ kv ≤ 1`) [dimless]
    k_p : float
        Parameter which accomplishes the relative minimum gain for
        peak side-lobe patterns [dimless]
    theta_3db : float
        3-dB beamwidth in the elevation plane [degrees]
    G180 : float
        Relative minimum gain according to Eq. (2b1) [dBi]

    Returns
    -------
    G_vr : np.ndarray, float
        Relative reference antenna gain in the elevation plane [dBi]
    '''

    x_v = np.atleast_1d(x_v)
    x_k = np.sqrt(1 - 0.36 * k_v)
    C = (
        10 * np.log10(
            (180. / theta_3db) ** 1.5 * (4 ** -1.5) / (1 + 8 * k_p)
            ) /
        np.log10(22.5 / theta_3db)
        )

    lambda_kv = 12 - C * np.log10(4) - 10 * np.log10(4 ** -1.5 + k_v)

    G = np.empty_like(x_v)

    # x_v < x_k
    mask = x_v < x_k
    G[mask] = -12 * x_v[mask] ** 2

    # x_k <= x_v < 4
    mask = (x_k <= x_v) & (x_v < 4)
    G[mask] = -12 + 10 * np.log10(x_v[mask] ** -1.5 + k_v)

    # 4 <= x_v < 90 / theta_3db
    mask = (4 <= x_v) & (x_v < 90 / theta_3db)
    G[mask] = -lambda_kv - C * np.log10(x_v[mask])

    # 90 / theta_3db == x_v
    mask = x_v == 90 / theta_3db
    G[mask] = G180

    return G


@utils.ranged_quantity_input(
    azim=(-180, 180, apu.deg),
    elev=(-90, 90, apu.deg),
    G0=(None, None, cnv.dB),
    phi_3db=(0, None, apu.deg),
    theta_3db=(0, None, apu.deg),
    k_p=(0, None, cnv.dimless),
    k_h=(0, None, cnv.dimless),
    k_v=(0, None, cnv.dimless),
    tilt_m=(-90, 90, apu.deg),
    tilt_e=(-90, 90, apu.deg),
    strip_input_units=True, output_unit=cnv.dB
    )
def imt_advanced_sectoral_peak_sidelobe_pattern_400_to_6000_mhz(
        azim, elev,
        G0, phi_3db, theta_3db,
        k_p, k_h, k_v,
        tilt_m=0., tilt_e=0.,
        ):
    '''
    IMT advanced (LTE) antenna pattern (sectoral, peak side-lobe)
    for the frequency range 400 MHz to 6 GHz according to
    `ITU-R Rec F.1336-5 <https://www.itu.int/rec/R-REC-F.1336-5-201901-I/en>`_
    Section 3.1.1.

    Parameters
    ----------
    azim, elev : `~astropy.units.Quantity`
        Azimuth/Elevation [deg]
    G0 : `~astropy.units.Quantity`
        Antenna maximum gain [dBi]
    phi_3db, theta_3db : float
        3-dB beamwidth in the azimuth/elevation plane [degrees]
    k_p : float
        Parameter which accomplishes the relative minimum gain for
        peak side-lobe patterns [dimless]
    k_h : float
        Azimuth pattern adjustment factor based on leaked power
        (:math:`0 \leq k_h \leq 1`) [dimless]
    k_v : float
        Elevation pattern adjustment factor based on leaked power
        (:math:`0 \leq k_v \leq 1`) [dimless]
    tilt_m : float
        Mechanical tilt angle (downwards) [deg]
    tilt_e : float
        Electrical tilt angle (downwards) [deg]

    Returns
    -------
    G : `~astropy.units.Quantity`
        Antenna pattern [dB]

    Notes
    -----
    For typical values of :math:`k_p`, :math:`k_h`, and :math:`k_v` see
    `ITU-R Rec F.1336-5 <https://www.itu.int/rec/R-REC-F.1336-5-201901-I/en>`_
    Sections 3.1.1.1-3.

    For cases involving sectoral antennas with :math:`\\varphi_\\mathrm{3dB} \\lesssim 120^\\circ` the following
    formula can be used to calculate :math:`\\vartheta_\\mathrm{3dB}`:

    .. math::

        \\vartheta_\\mathrm{3dB}=\\frac{31000\\times10^{-0.1G_0}}{\\varphi_\\mathrm{3dB}}

    '''

    return imt_advanced_sectoral_peak_sidelobe_pattern_cython(
        azim, elev,
        G0, phi_3db, theta_3db,
        k_p, k_h, k_v,
        tilt_m, tilt_e,
        )


@utils.ranged_quantity_input(
    azim=(-180, 180, apu.deg),
    elev=(-90, 90, apu.deg),
    G0=(None, None, cnv.dB),
    phi_3db=(0, None, apu.deg),
    theta_3db=(0, None, apu.deg),
    k_a=(0, None, cnv.dimless),
    k_h=(0, None, cnv.dimless),
    k_v=(0, None, cnv.dimless),
    tilt_m=(-90, 90, apu.deg),
    tilt_e=(-90, 90, apu.deg),
    strip_input_units=True, output_unit=cnv.dB
    )
def imt_advanced_sectoral_avg_sidelobe_pattern_400_to_6000_mhz(
        azim, elev,
        G0, phi_3db, theta_3db,
        k_a, k_h, k_v,
        tilt_m=0., tilt_e=0.,
        ):
    '''
    IMT advanced (LTE) antenna pattern (sectoral, peak side-lobe)
    for the frequency range 400 MHz to 6 GHz according to
    `ITU-R Rec F.1336-5 <https://www.itu.int/rec/R-REC-F.1336-5-201901-I/en>`_
    Section 3.1.2.

    Parameters
    ----------
    azim, elev : `~astropy.units.Quantity`
        Azimuth/Elevation [deg]
    G0 : `~astropy.units.Quantity`
        Antenna maximum gain [dBi]
    phi_3db, theta_3db : float
        3-dB beamwidth in the azimuth/elevation plane [degrees]
    k_a : float
        Parameter which accomplishes the relative minimum gain for
        average side-lobe patterns [dimless]
    k_h : float
        Azimuth pattern adjustment factor based on leaked power
        (:math:`0 \leq k_h \leq 1`) [dimless]
    k_v : float
        Elevation pattern adjustment factor based on leaked power
        (:math:`0 \leq k_v \leq 1`) [dimless]
    tilt_m : float
        Mechanical tilt angle (downwards) [deg]
    tilt_e : float
        Electrical tilt angle (downwards) [deg]

    Returns
    -------
    G : `~astropy.units.Quantity`
        Antenna pattern [dB]

    Notes
    -----
    For typical values of :math:`k_p`, :math:`k_h`, and :math:`k_v` see
    `ITU-R Rec F.1336-4 <https://www.itu.int/rec/R-REC-F.1336-4-201402-I/en>`_
    Sections 3.1.1.1-3.

    For cases involving sectoral antennas with :math:`\\varphi_\\mathrm{3dB} \\lesssim 120^\\circ` the following
    formula can be used to calculate :math:`\\vartheta_\\mathrm{3dB}`:

    .. math::

        \\vartheta_\\mathrm{3dB}=\\frac{31000\\times10^{-0.1G_0}}{\\varphi_\\mathrm{3dB}}

    '''

    return imt_advanced_sectoral_avg_sidelobe_pattern_cython(
        azim, elev,
        G0, phi_3db, theta_3db,
        k_a, k_h, k_v,
        tilt_m, tilt_e,
        )


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
