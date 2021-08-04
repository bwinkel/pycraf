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
from astropy.utils.data import get_pkg_data_filename
from .. import conversions as cnv
from .. import utils


try:
    import matplotlib  # pylint: disable=W0611
    from matplotlib.colors import Normalize

    # On older versions of matplotlib Normalize is an old-style class
    if not isinstance(Normalize, type):
        class Normalize(Normalize, object):
            pass
except ImportError:
    class Normalize(object):
        def __init__(self, *args, **kwargs):
            raise ImportError(
                'The "matplotlib" package is necessary to use this.'
                )


__all__ = [
    'R_E', 'K_BETA', 'A_BETA',
    'annual_timepercent_from_worst_month',
    'deltaN_N0_from_map', 'radiomet_data_for_pathcenter',
    'eff_earth_radius_factor_median',
    'eff_earth_radius_factor_beta',
    'eff_earth_radius_median', 'eff_earth_radius_beta',
    'make_kmz', 'terrain_cmap_factory',
    ]


# useful constants
R_E_VALUE = 6371.
R_E = R_E_VALUE * apu.km  # Earth radius
R_E.__doc__ = '''Earth Radius'''

K_BETA_VALUE = 3.
K_BETA = K_BETA_VALUE * cnv.dimless  # eff. Earth radius factor for beta_0
K_BETA.__doc__ = '''Effective Earth radius factor for beta_0 percent'''

A_BETA_VALUE = 3. * R_E_VALUE
A_BETA = K_BETA_VALUE * apu.km  # eff. Earth radius for beta_0
A_BETA.__doc__ = '''Effective Earth radius for beta_0 percent'''

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

_refract_data = np.load(get_pkg_data_filename(
    '../itudata/p.452-16/refract_map.npz'
    ))


_DN_interpolator = RegularGridInterpolator(
    (_refract_data['lons'][0], _refract_data['lats'][::-1, 0]),
    _refract_data['dn50'][::-1].T
    )
_N0_interpolator = RegularGridInterpolator(
    (_refract_data['lons'][0], _refract_data['lats'][::-1, 0]),
    _refract_data['n050'][::-1].T
    )


@utils.ranged_quantity_input(
    p_w=(0, 100, apu.percent),
    phi=(-90, 90, apu.deg),
    omega=(0, 100, apu.percent),
    strip_input_units=True, output_unit=apu.percent,
    )
def annual_timepercent_from_worst_month(
        p_w, phi, omega
        ):
    '''
    Calculate annual equivalent time percentage, p, from worst-month time
    percentage, p_w, according to ITU-R P.452-16 Eq (1).

    Parameters
    ----------
    p_w : `~astropy.units.Quantity`
        worst-month time percentage [%]
    phi : `~astropy.units.Quantity`
        Geographic latitude of path center [deg]
    omega : `~astropy.units.Quantity`
        Fraction of the path over water (see Table 3) [%]

    Returns
    -------
    p : `~astropy.units.Quantity`
        Annual equivalent time percentage [%]

    Notes
    -----
    - Use this function, if you want to do path propagation calculations
      for the worst month case. The resulting time percentage, p, can then
      be plugged into other functions. If you want just annual averages,
      simply use your time percentage value as is.
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


def _DN_N0_from_map(lon, lat):

    _DN = _DN_interpolator((lon % 360, lat))
    _N0 = _N0_interpolator((lon % 360, lat))

    return _DN, _N0


@utils.ranged_quantity_input(
    lon=(-180, 360, apu.deg),
    lat=(-90, 90, apu.deg),
    strip_input_units=True,
    output_unit=(cnv.dimless / apu.km, cnv.dimless),
    )
def deltaN_N0_from_map(lon, lat):
    '''
    Query delta_N and N_0 values from digitized maps by means of bilinear
    interpolation.

    Parameters
    ----------
    lon, lat : `~astropy.units.Quantity`
        Geographic longitude and latitude of path center [deg]

    Returns
    -------
    delta_N : `~astropy.units.Quantity`
        Average radio-refractive index lapse-rate through the lowest 1 km of
        the atmosphere [N-units/km == 1/km]
    N_0 : `~astropy.units.Quantity`
        Sea-level surface refractivity [N-units == dimless]

    Notes
    -----
    - The values for `delta_N` and `N_0` are queried from
      a radiometeorological map provided with `ITU-R Rec. P.452
      <https://www.itu.int/rec/R-REC-P.452-16-201507-I/en>`_.
    '''

    return _DN_N0_from_map(lon, lat)


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


@utils.ranged_quantity_input(
    lon=(-180, 360, apu.deg),
    lat=(-90, 90, apu.deg),
    d_tm=(0, None, apu.km),
    d_lm=(0, None, apu.km),
    strip_input_units=True,
    output_unit=(cnv.dimless / apu.km, apu.percent, cnv.dimless),
    )
def radiomet_data_for_pathcenter(lon, lat, d_tm, d_lm):
    '''
    Calculate delta_N, beta_0, and N_0 values from digitized maps, according
    to ITU-R P.452-16 Eq (2-4).

    Parameters
    ----------
    lon, lat : `~astropy.units.Quantity`
        Geographic longitude and latitude of path center [deg]
    d_tm : `~astropy.units.Quantity`, optional
        longest continuous land (inland + coastal) section of the
        great-circle path [km]
    d_lm : `~astropy.units.Quantity`, optional
        longest continuous inland section of the great-circle path [km]

    Returns
    -------
    delta_N : `~astropy.units.Quantity`
        Average radio-refractive index lapse-rate through the lowest 1 km of
        the atmosphere [N-units/km == 1/km]
    beta_0 : `~astropy.units.Quantity`
        the time percentage for which refractive index lapse-rates
        exceeding 100 N-units/km can be expected in the first 100 m
        of the lower atmosphere [%]
    N_0 : `~astropy.units.Quantity`
        Sea-level surface refractivity [N-units == dimless]

    Notes
    -----
    - The values for `delta_N` and `N_0` are queried from
      a radiometeorological map provided with `ITU-R Rec. P.452
      <https://www.itu.int/rec/R-REC-P.452-16-201507-I/en>`_.
    - Radio-climaticzones can be obtained from
      `ITU Digitized World Map (IDWM) <http://www.itu.int/pub/R-SOFT-IDWM>`_.
      For many applications, it is probably the case, that only inland
      zones are present along the path of length d.
      In this case, d_tm = d_lm = d.
    '''

    return _radiomet_data_for_pathcenter(lon, lat, d_tm, d_lm)


@utils.ranged_quantity_input(
    lon=(-180, 360, apu.deg),
    lat=(-90, 90, apu.deg),
    strip_input_units=True,
    output_unit=cnv.dimless,
    )
def eff_earth_radius_factor_median(lon, lat):
    '''
    Calculate median effective Earth radius factor, k_50, according to
    ITU-R P.452-16 Eq (5).

    Parameters
    ----------
    lon, lat : `~astropy.units.Quantity`
        Geographic longitude and latitude of path center [deg]

    Returns
    -------
    k50 : `~astropy.units.Quantity`
        Median effective Earth radius factor [dimless]

    Notes
    -----
    - Uses delta_N, which is derived from digitized maps (shipped with P.452)
      by bilinear interpolation; see also
      `~pycraf.pathprof.deltaN_N0_from_map`.
    '''

    return 157. / (157. - _DN_interpolator((lon % 360, lat)))


@utils.ranged_quantity_input(
    strip_input_units=True,
    output_unit=cnv.dimless,
    )
def eff_earth_radius_factor_beta():
    '''
    Calculate effective Earth radius factor exceeded for beta_0 percent
    of time, k_beta, according to ITU-R P.452-16.

    Returns
    -------
    k_beta : `~astropy.units.Quantity`
        Effective Earth radius factor exceeded for beta_0 percent
        of time [dimless]

    Notes
    -----
    - This is just a constant. You could also use K_BETA to avoid overhead.
    '''

    return K_BETA_VALUE


def _eff_earth_radius_median(lon, lat):

    return R_E_VALUE * 157. / (157. - _DN_interpolator((lon % 360, lat)))


@utils.ranged_quantity_input(
    lon=(-180, 360, apu.deg),
    lat=(-90, 90, apu.deg),
    strip_input_units=True,
    output_unit=apu.km,
    )
def eff_earth_radius_median(lon, lat):
    '''
    Calculate median effective Earth radius, a_e, according to
    ITU-R P.452-16 Eq (6a).

    Parameters
    ----------
    lon, lat : `~astropy.units.Quantity`
        Geographic longitude and latitude of path center [deg]

    Returns
    -------
    a_e : `~astropy.units.Quantity`
        Median effective Earth radius [km]

    Notes
    -----
    - Uses delta_N, which is derived from digitized maps (shipped with P.452)
      by bilinear interpolation; see also
      `~pycraf.pathprof.deltaN_N0_from_map`.
    '''

    return _eff_earth_radius_median(lon, lat)


@utils.ranged_quantity_input(
    strip_input_units=True,
    output_unit=apu.km,
    )
def eff_earth_radius_beta():
    '''
    Calculate effective Earth radius exceeded for beta_0 percent of time,
    a_beta, according to ITU-R P.452-16 Eq (6b).

    Returns
    -------
    a_beta : `~astropy.units.Quantity`
        Effective Earth radius exceeded for beta_0 percent of time [km]

    Notes
    -----
    - This is just a constant. You could also use A_BETA to avoid overhead.
    '''

    return A_BETA_VALUE


def make_kmz(
        kmz_filename, atten_map, bbox, vmin=None, vmax=None, cmap='inferno_r'
        ):
    '''
    Produce kmz file for use in GIS software (e.g., Google Earth).

    Parameters
    ----------
    kmz_filename : str
        Output file name for .kmz-file
    atten_map : 2D `~numpy.ndarray` of floats
        2D array with path attenuation values
    bbox : tuple of 4 floats
        (east, south, west, north) edges of map [deg]
    vmin, vmax : float
        Lower and upper colorbar bounds.
        If None, 2.5% and 97.5% percentiles of atten_map are used
        (default: None)
    cmap : matplotlib.colormap
        (default: 'inferno_r')
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


class FixPointNormalize(Normalize):
    '''
    From http://stackoverflow.com/questions/40895021/python-equivalent-for-matlabs-demcmap-elevation-appropriate-colormap
    by ImportanceOfBeingErnest

    Inspired by http://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib
    Subclassing Normalize to obtain a colormap with a fixpoint
    somewhere in the middle of the colormap.

    This may be useful for a `terrain` map, to set the 'sea level'
    to a color in the blue/turquise range.
    '''

    def __init__(
            self,
            vmin=None, vmax=None, sealevel=0,
            col_val=0.21875, clip=False
            ):

        # sealevel is the fix point of the colormap (in data units)
        self.sealevel = sealevel
        # col_val is the color value in the range [0, 1]
        # that should represent the sealevel.
        self.col_val = col_val
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):

        x, y = [self.vmin, self.sealevel, self.vmax], [0, self.col_val, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def terrain_cmap_factory(sealevel=0.5, vmax=1200):
    '''
    Produce terrain colormap and norm to be used in plt.imshow.

    With this, one can adjust the colors in the cmap such that the sea level
    is properly defined (blue).

    A simple use case would look like the following::

        >>> vmin, vmax = -20, 1200  # doctest: +SKIP
        >>> terrain_cmap, terrain_norm = terrain_cmap_factory(vmax=vmax)  # doctest: +SKIP
        >>> plt.imshow(  # doctest: +SKIP
        ...     heights, cmap=terrain_cmap, norm=terrain_norm,
        ...     # vmin=vmin, vmax=vmax  # deprecated in newer matplotlib versions
        ...     )

    Parameters
    ----------
    sealevel : float
        The sealevel value.
    vmax : float
        Maximum height to cover in the colormap (Default: 1200)
        (in older matplotlib versions, one should call plt.imshow with the
        same `vmax` option, and `vmin=-20`!)

    Returns
    -------
    terrain_cmap : matplotlib.colors.LinearSegmentedColormap
    terrain_norm : matplotlib.colors.Normalize instance
    '''

    # Combine the lower and upper range of the terrain colormap with a gap in
    # the middle to let the coastline appear more prominently. Inspired by
    # stackoverflow.com/questions/31051488/combining-two-matplotlib-colormaps

    try:
        import matplotlib
    except ImportError:
        raise ImportError(
            'The "matplotlib" package is necessary to use this function.'
            )

    colors_undersea = matplotlib.pyplot.cm.terrain(np.linspace(0, 0.17, 56))
    colors_land = matplotlib.pyplot.cm.terrain(np.linspace(0.25, 1, 200))

    # combine them and build a new colormap
    colors = np.vstack((colors_undersea, colors_land))
    terrain_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'terrain_normed', colors
        )
    terrain_norm = FixPointNormalize(sealevel=sealevel, vmax=vmax)

    return terrain_cmap, terrain_norm


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
