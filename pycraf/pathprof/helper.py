#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

# from functools import partial, lru_cache
import os
from astropy import units as apu
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .. import conversions as cnv
from .. import helpers


__all__ = [
    'R_E', 'K_BETA', 'A_BETA_VALUE',
    'anual_time_percentage_from_worst_month',
    'radiomet_data_for_pathcenter',
    'median_effective_earth_radius_factor',
    'effective_earth_radius_factor_beta',
    'median_effective_earth_radius', 'effective_earth_radius_beta',
    'make_kmz',
    ]


# useful constants
R_E_VALUE = 6371.
R_E = R_E_VALUE * apu.km  # Earth radius
K_BETA_VALUE = 3.
K_BETA = K_BETA_VALUE * cnv.dimless  # eff. Earth radius factor for beta_0
A_BETA_VALUE = 3. * R_E_VALUE
A_BETA = K_BETA_VALUE * apu.km  # eff. Earth radius for beta_0

KML_TEMPLATE = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
    <Folder>
    <name>Attenuation map</name>
    <description>Results from pycraf package</description>
    <GroundOverlay>
        <name>Attenuation map on terrain</name>
        <description>Results from the pycraf package</description>
        <color>aaffffff</color>
        <Icon><href>pycraf_atten_map_kmz.png</href></Icon>
        <LatLonBox>
            <east>{:.6f}</east>
            <south>{:.6f}</south>
            <west>{:.6f}</west>
            <north>{:.6f}</north>
        </LatLonBox>
    </GroundOverlay>
    </Folder>
</kml>
'''

# maybe, the following should only be computed on demand?
this_dir, this_filename = os.path.split(__file__)
fname_refract_data = os.path.join(
    this_dir, '../itudata/p.452-16', 'refract_map.npz'
    )
_refract_data = np.load(fname_refract_data)


_DN_interpolator = RegularGridInterpolator(
    (_refract_data['lons'][0], _refract_data['lats'][::-1, 0]),
    _refract_data['dn50'][::-1].T
    )
_N0_interpolator = RegularGridInterpolator(
    (_refract_data['lons'][0], _refract_data['lats'][::-1, 0]),
    _refract_data['n050'][::-1].T
    )


@helpers.ranged_quantity_input(
    p_w=(0, 100, apu.percent),
    phi=(-90, 90, apu.deg),
    omega=(0, 100, apu.percent),
    strip_input_units=True, output_unit=apu.percent,
    )
def anual_time_percentage_from_worst_month(
        p_w, phi, omega
        ):
    '''
    Calculate annual equivalent time percentage, p, from worst-month time
    percentage, p_w, according to ITU-R P.452-16 Eq. (1).

    Parameters
    ----------
    p_w - worst-month time percentage in %
    phi - path center latitude in degrees
    omega - fraction of the path over water in % (see Table 3)

    Returns
    -------
    p - annual equivalent time percentage in %

    Notes
    -----
    Use this function, if you want to do path propagation calculations
    for the worst month case. The resulting time percentage, p, can then
    be plugged into other functions. If you want just annual averages,
    just use your time percentage value as is.
    '''

    omega /= 100.  # convert from percent to fraction

    tmp = np.abs(np.cos(2 * np.radians(phi))) ** 0.7
    G_l = np.sqrt(
        np.where(np.abs(phi) <= 45, 1.1 + tmp, 1.1 - tmp)
        )

    a = np.log10(p_w) + np.log10(G_l) - 0.186 * omega - 0.444
    b = 0.078 * omega + 0.816
    p = 10 ** (a / b)

    p = np.max([p, p_w / 12.], axis=0)
    return p


def _N_from_map(lon, lat):
    '''
    Query ΔN and N_0 values from digitized maps by means of bilinear interpol.


    Parameters
    ----------
    lon, lat - path center coordinates [deg]


    Returns
    -------
    delta_N, N_0 - radiometeorological data
        delta_N - average radio-refractive index lapse-rate through the
                  lowest 1 km of the atmosphere in N-units/km
        N_0 - sea-level surface refractivity in N-units
    '''

    _DN = _DN_interpolator((lon % 360, lat))
    _N0 = _N0_interpolator((lon % 360, lat))

    return _DN, _N0


def _radiomet_data_for_pathcenter(lon, lat, d_tm, d_lm):
    _DN = _DN_interpolator((lon % 360, lat))
    _N0 = _N0_interpolator((lon % 360, lat))

    _tau = 1. - np.exp(-4.12e-4 * np.power(d_lm, 2.41))
    _absphi = np.abs(lat)

    _a = np.power(10, -d_tm / (16. - 6.6 * _tau))
    _b = np.power(10, -5 * (0.496 + 0.354 * _tau))
    _mu1 = np.power(_a + _b, 0.2)
    _mu1 = np.where(_mu1 <= 1, _mu1, 1.)
    _log_mu1 = np.log10(_mu1)

    _phi_cond = _absphi <= 70.
    _mu4 = np.where(
        _phi_cond,
        np.power(10, (-0.935 + 0.0176 * _absphi) * _log_mu1),
        np.power(10, 0.3 * _log_mu1)
        )

    beta_0 = np.where(
        _phi_cond,
        np.power(10, -0.015 * _absphi + 1.67) * _mu1 * _mu4,
        4.17 * _mu1 * _mu4
        )

    return _DN, beta_0, _N0


@helpers.ranged_quantity_input(
    lon=(0, 360, apu.deg),
    lat=(-90, 90, apu.deg),
    d_tm=(0, None, apu.km),
    d_lm=(0, None, apu.km),
    strip_input_units=True,
    output_unit=(cnv.dimless / apu.km, apu.percent, cnv.dimless),
    )
def radiomet_data_for_pathcenter(lon, lat, d_tm, d_lm):
    '''
    Calculate radiometeorological data, ΔN, β_0 and N_0, from path center
    coordinates, according to ITU-R P.452-16 Eq. (2-4).

    Parameters
    ----------
    lon, lat - path center coordinates [deg]
    d_tm - longest continuous land (inland + coastal) section of the
        great-circle path [km]
    d_lm - longest continuous inland section of the great-circle path [km]

    Returns
    -------
    delta_N, beta_0, N_0 - radiometeorological data
        delta_N - average radio-refractive index lapse-rate through the
            lowest 1 km of the atmosphere [N-units/km]
        beta_0 - the time percentage for which refractive index lapse-rates
            exceeding 100 N-units/km can be expected in the first 100 m
            of the lower atmosphere [%]
        N_0 - sea-level surface refractivity [N-units]

    Notes
    -----
    - ΔN and N_0 are derived from digitized maps (shipped with P.452) by
      bilinear interpolation.
    - Radio-climaticzones can be queried from ITU Digitized World Map (IDWM).
      For many applications, it is probably the case, that only inland
      zones are present along the path of length d.
      In this case, d_tm = d_lm = d.
    '''

    return _radiomet_data_for_pathcenter(lon, lat, d_tm, d_lm)


@helpers.ranged_quantity_input(
    lon=(0, 360, apu.deg),
    lat=(-90, 90, apu.deg),
    strip_input_units=True,
    output_unit=cnv.dimless,
    )
def median_effective_earth_radius_factor(lon, lat):
    '''
    Calculate median effective Earth radius factor, k_50, according to
    ITU-R P.452-16 Eq. (5).

    Parameters
    ----------
    lon, lat - path center coordinates [deg]

    Returns
    -------
    k50 - median effective Earth radius factor [dimless]

    Notes
    -----
    - Uses ΔN, which is derived from digitized maps (shipped with P.452) by
      bilinear interpolation.
    '''

    return 157. / (157. - _DN_interpolator((lon % 360, lat)))


@helpers.ranged_quantity_input(
    strip_input_units=True,
    output_unit=cnv.dimless,
    )
def effective_earth_radius_factor_beta():
    '''
    Calculate effective Earth radius factor exceeded for beta_0 percent
    of time, k_beta, according to ITU-R P.452-16.

    Returns
    -------
    k_beta - effective Earth radius factor exceeded for beta_0 percent
        of time [dimless]

    Notes
    -----
    - This is just a constant. Better use K_BETA to avoid overhead.
    '''

    return K_BETA_VALUE


def _median_effective_earth_radius(lon, lat):

    return R_E_VALUE * 157. / (157. - _DN_interpolator((lon % 360, lat)))


@helpers.ranged_quantity_input(
    lon=(0, 360, apu.deg),
    lat=(-90, 90, apu.deg),
    strip_input_units=True,
    output_unit=apu.km,
    )
def median_effective_earth_radius(lon, lat):
    '''
    Calculate median effective Earth radius, a_e, according to
    ITU-R P.452-16 Eq. (6a).

    Parameters
    ----------
    lon, lat - path center coordinates [deg]

    Returns
    -------
    a_e - median effective Earth radius [km]

    Notes
    -----
    - Uses ΔN, which is derived from digitized maps (shipped with P.452) by
      bilinear interpolation.
    '''

    return _median_effective_earth_radius(lon, lat)


@helpers.ranged_quantity_input(
    strip_input_units=True,
    output_unit=apu.km,
    )
def effective_earth_radius_beta(lon, lat):
    '''
    Calculate effective Earth radius exceeded for beta_0 percent of time,
    a_beta, according to ITU-R P.452-16 Eq. (6b).

    Returns
    -------
    a_beta - effective Earth radius exceeded for beta_0 percent of time [km]

    Notes
    -----
    - This is just a constant. Better use A_BETA to avoid overhead.
    '''

    return A_BETA_VALUE


def make_kmz(
        kmz_filename, atten_map, bbox, vmin=None, vmax=None, cmap='inferno_r'
        ):
    '''
    Produce kmz file for use in GIS software (e.g., Google Earth).

    Parameters
    ----------
    kmz_filename - output file name
    atten_map - 2D array with path attenuation
    bbox - tuple (east, south, west, north) edges of map [deg]
    vmin, vmax - lower and upper colorbar bounds
        if None, 2.5% and 97.5% percentiles of atten_map are used
    cmap - matplotlib colormap
    '''

    # descriptive xml
    kml = KML_TEMPLATE.format(*bbox)

    # produce jpg, use python pillow for this;
    # however, we don't want this as global requirement,
    # therefore, just a local import
    # from PIL import Image
    # from matplotlib import colors, cm

    if vmin is None:
        vmin = np.percentile(atten_map.flatten(), 2.5)

    if vmax is None:
        vmax = np.percentile(atten_map.flatten(), 97.5)

    # norm = colors.Normalize(vmin=vmin, vmax=vmax)
    # csm = cm.ScalarMappable(norm=norm, cmap=cmap)

    # rgba = csm.to_rgba(atten_map)
    # jpeg = Image.fromarray(np.int32(255 * rgba + 0.5), mode='RGBA')

    from matplotlib.image import imsave
    from io import BytesIO

    png_buf = BytesIO()

    imsave(
        png_buf,
        atten_map,
        vmin=vmin, vmax=vmax, cmap=cmap,
        origin='lower'
        )

    png_buf.seek(0)

    # write as kmz (zip file)
    import zipfile

    with zipfile.ZipFile(kmz_filename, 'w') as myzip:
        myzip.writestr('pycraf_atten_map_kmz.png', png_buf.read())
        myzip.writestr('doc.kml', kml)


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
