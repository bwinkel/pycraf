.. pycraf-pathprof:

*****************************************************************
Path attenuation, terrain data, and geodesics (`pycraf.pathprof`)
*****************************************************************

.. currentmodule:: pycraf.pathprof

Introduction
============

The `~pycraf.pathprof` subpackage is probably the most important piece of
pycraf. It offers tools to calculate so-call path attenuation (propagation
loss), which is a key ingredient for compatibility studies in Radio spectrum
management. All kinds of Radio services have to cooperate and do their best to
not harm other users/services of the spectrum - neither in the allocated
spectral band, nor in the out-of-band or spurious domain (aka spectral
sidelobes). For this, it is fundamental to calculate the propagation loss that
a transmitted signal will experience before it is entering the receiving
terminal system.

There are a lot of parameters and environmental conditions that determine the
propagation loss, such as the frequency, :math:`f`, of radiation, the distance,
:math:`d`, between transmitter and receiver, the antenna gains and boresight
angles, but also the terrain along the propagating path. For close distances,
one often has the line-of-sight case, where the path loss is approximately
proportional to :math:`d^{-2}f^{-2}` (numbers smaller than One, mean an
attenuation), i.e., the loss quickly gets larger with distance and frequency.
For longer separations, other effects start to play a more important role:

    - Tropospheric scatter loss
    - Ducting and anomalous layer refraction loss
    - Diffraction loss
    - Clutter loss

Especially, for diffraction loss, information on the terrain heights is
needed. Therefore, `pycraf` implements functions to query
`SRTM data <https://www2.jpl.nasa.gov/srtm/>`, which was obtained from
a space shuttle mission. SRTM data has high spatial resolution of 3" (90 m),
which makes it more than appropriate for path propagation calculations.

In order to calculate the overall path attenuation, pycraf follows the
algorithm proposed in `ITU-R Recommendation P.452
<https://www.itu.int/rec/R-REC-P.452-16-201507-I/en>`_. We refer the user
to this document to learn more about the details of such calculations,
because the matter is quite complex and beyond the scope of this user manual.

Several helper routines are necessary to collect the necessary parameters for
the P.452 calculations, that fall in the following categories:

    - Height profile (terrain map) construction.
    - Determination of Geodesics (the Great-circle path equivalent on the
      Earth's ellipsoid, the so-called Geoid).
    - Finding radiometerological data for the path's midpoint, which is also
      necessary to determine the effective Earth radii.

.. note::

    For most of the functionality in this module, you will need to download
    SRTM tile data; see :ref:`working_with_srtm`.


Getting Started
===============

.. _pathprof-getting-started-height-profile:

Height profiles
---------------
Let's start with querying `SRTM data` and plot a height profile.

.. plot::
    :include-source:

    import matplotlib.pyplot as plt
    from astropy import units as u
    from pycraf import pathprof

    # allow download of missing SRTM data:
    pathprof.SrtmConf.set(download='missing')

    lon_t, lat_t = 6.8836 * u.deg, 50.525 * u.deg
    lon_r, lat_r = 7.3334 * u.deg, 50.635 * u.deg
    hprof_step = 100 * u.m

    (
        lons, lats, distance,
        distances, heights,
        bearing, back_bearing, back_bearings
        ) = pathprof.srtm_height_profile(
            lon_t, lat_t, lon_r, lat_r, hprof_step
            )

    _distances = distances.to(u.km).value
    _heights = heights.to(u.m).value

    plt.figure(figsize=(10, 5))
    plt.plot(_distances, _heights, 'k-')
    plt.xlabel('Distance [km]')
    plt.ylabel('Height [m]')
    plt.grid()


.. note::

    If the profile resolution, `hprof_step` is made large,
    Gaussian smoothing is applied to avoid aliasing effects.


.. _pathprof-getting-started-path-attenuation:

Path attenuation
----------------

The `~pycraf.pathprof` package implements `ITU-R Recommendation P.452-16
<https://www.itu.int/rec/R-REC-P.452-16-201507-I/en>`_ for path propagation
loss calculations. In `~pycraf.pathprof` this is a two-step procedure.
First a helper object, a `~pycraf.pathprof.PathProf` object, has to be
instantiated. It contains all kinds of parameters that define the path
geometry and hold other necessary quantities (there are many!). Second, one
feeds this object into one of the functions that calculate attenuation

    - Line-of-sight (LoS) or free-space loss: `~pycraf.pathprof.loss_freespace`
    - Tropospheric scatter loss: `~pycraf.pathprof.loss_troposcatter`
    - Ducting and anomalous layer refraction loss:
      `~pycraf.pathprof.loss_ducting`
    - Diffraction loss: `~pycraf.pathprof.loss_diffraction`

Of course, there is also a function (`~pycraf.pathprof.loss_complete`) to
calculate the total loss (a non-trivial combination of the above). As the
pathprof.complete_loss function also returns the most important constituents,
it is usually sufficient to use that. The individual functions are only
provided for reasons of computing speed, if one is really only interested in
one component::

    >>> from astropy import units as u
    >>> from pycraf import pathprof, conversions as cnv

    >>> pathprof.SrtmConf.set(download='missing')  # doctest: +IGNORE_OUTPUT

    >>> freq = 1. * u.GHz

    >>> lon_tx, lat_tx = 6.8836 * u.deg, 50.525 * u.deg
    >>> lon_rx, lat_rx = 7.3334 * u.deg, 50.635 * u.deg
    >>> hprof_step = 100 * u.m  # resolution of height profile

    >>> omega = 0. * u.percent  # fraction of path over sea
    >>> temperature = 290. * u.K
    >>> pressure = 1013. * u.hPa
    >>> time_percent = 2 * u.percent  # see P.452 for explanation
    >>> h_tg, h_rg = 5 * u.m, 50 * u.m
    >>> G_t, G_r = 0 * cnv.dBi, 15 * cnv.dBi

    # clutter zones
    >>> zone_t, zone_r = pathprof.CLUTTER.URBAN, pathprof.CLUTTER.SUBURBAN

    >>> pprop = pathprof.PathProp(  # doctest: +REMOTE_DATA +IGNORE_OUTPUT
    ...     freq,
    ...     temperature, pressure,
    ...     lon_tx, lat_tx,
    ...     lon_rx, lat_rx,
    ...     h_tg, h_rg,
    ...     hprof_step,
    ...     time_percent,
    ...     zone_t=zone_t, zone_r=zone_r,
    ...     )

The PathProp object is immutable, so if you want to change something, you have
to create a new instance. This is, because, many member attributes are
dependent on each other and by just changing one value one could easily create
inconsistencies. It is easily possible to access the parameters::

    >>> print(repr(pprop))  # doctest: +REMOTE_DATA
    PathProp<Freq: 1.000 GHz>

    >>> print(pprop)  # doctest: +REMOTE_DATA
    version        :           16 (P.452 version; 14 or 16)
    freq           :     1.000000 GHz
    wavelen        :     0.299792 m
    polarization   :            0 (0 - horizontal, 1 - vertical)
    temperature    :   290.000000 K
    pressure       :  1013.000000 hPa
    time_percent   :     2.000000 percent
    beta0          :     1.954410 percent
    omega          :     0.000000 percent
    lon_t          :     6.883600 deg
    lat_t          :    50.525000 deg
    lon_r          :     7.333400 deg
    lat_r          :    50.635000 deg
    lon_mid        :     7.108712 deg
    lat_mid        :    50.580333 deg
    delta_N        :    38.080852 dimless / km
    N0             :   324.427961 dimless
    distance       :    34.128124 km
    bearing        :    68.815620 deg
    back_bearing   :  -110.836903 deg
    hprof_step     :   100.000000 m
    ...

    # path elevation angles as seen from Tx, Rx
    >>> print('{:.3f} {:.3f}'.format(pprop.eps_pt, pprop.eps_pr))  # doctest: +REMOTE_DATA
    4.530 deg 1.883 deg

With the PathProp object, it is now just one function call to get the path
loss::

    >>> tot_loss = pathprof.loss_complete(pprop, G_t, G_r)  # doctest: +REMOTE_DATA
    >>> print('L_bfsg:   {:5.2f} - Free-space loss\n'  # doctest: +REMOTE_DATA
    ...       'L_bd:     {:5.2f} - Basic transmission loss associated '
    ...       'with diffraction\n'
    ...       'L_bs:     {:5.2f} - Tropospheric scatter loss\n'
    ...       'L_ba:     {:5.2f} - Ducting/layer reflection loss\n'
    ...       'L_b:      {:5.2f} - Complete path propagation loss\n'
    ...       'L_b_corr: {:5.2f} - As L_b but with clutter correction\n'
    ...       'L:        {:5.2f} - As L_b_corr but with gain '
    ...       'correction'.format(*tot_loss)
    ...      )
    L_bfsg:   123.34 dB - Free-space loss
    L_bd:     173.73 dB - Basic transmission loss associated with diffraction
    L_bs:     225.76 dB - Tropospheric scatter loss
    L_ba:     212.81 dB - Ducting/layer reflection loss
    L_b:      173.73 dB - Complete path propagation loss
    L_b_corr: 192.94 dB - As L_b but with clutter correction
    L:        177.94 dB - As L_b_corr but with gain correction

Using `pycraf.pathprof`
=======================

.. _pathprof-using-terrain:

Geodesics, height profiles and terrain maps
-------------------------------------------
In the :ref:`pathprof-getting-started-height-profile` section, we have already
seen, how one can query a height profile from `SRTM data`. It is also easy
to produce terrain maps of a region:

.. plot::
    :include-source:

    import matplotlib.pyplot as plt
    from astropy import units as u
    from pycraf import pathprof

    pathprof.SrtmConf.set(download='missing')

    lon_t, lat_t = 9.943 * u.deg, 54.773 * u.deg  # Northern Germany
    map_size_lon, map_size_lat = 1.5 * u.deg, 1.5 * u.deg
    map_resolution = 3. * u.arcsec

    lons, lats, heightmap = pathprof.srtm_height_map(
        lon_t, lat_t,
        map_size_lon, map_size_lat,
        map_resolution=map_resolution,
        )

    _lons = lons.to(u.deg).value
    _lats = lats.to(u.deg).value
    _heightmap = heightmap.to(u.m).value

    vmin, vmax = -20, 170
    terrain_cmap, terrain_norm = pathprof.terrain_cmap_factory(
        sealevel=0.5, vmax=vmax
        )
    _heightmap[_heightmap < 0] = 0.51  # fix for coastal region

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes((0., 0., 1.0, 1.0))
    cbax = fig.add_axes((0., 0., 1.0, .02))
    cim = ax.imshow(
        _heightmap,
        origin='lower', interpolation='nearest',
        cmap=terrain_cmap, norm=terrain_norm,
        vmin=vmin, vmax=vmax,
        extent=(_lons[0], _lons[-1], _lats[0], _lats[-1]),
        )
    cbar = fig.colorbar(
        cim, cax=cbax, orientation='horizontal'
        )
    ax.set_aspect(abs(_lons[-1] - _lons[0]) / abs(_lats[-1] - _lats[0]))
    cbar.set_label(r'Height (amsl)', color='k')
    cbax.xaxis.set_label_position('top')
    for t in cbax.xaxis.get_major_ticks():
        t.tick1On = True
        t.tick2On = True
        t.label1On = False
        t.label2On = True
    ctics = np.arange(0, 1150, 50)
    cbar.set_ticks(ctics)
    cbar.ax.set_xticklabels(map('{:.0f} m'.format, ctics), color='k')
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

Here, we made use of a special `~matplotlib` colormap, which can be produced
using `~pycraf.pathprof.terrain_cmap_factory`. It returns a `cmap` and a
`norm`, which make it so that blue colors always start below a height of 0 m
(i.e., the sea level).

.. note::

    Internally, the height-profile generation with
    `~pycraf.pathprof.srtm_height_profile` calls two Geodesics helper
    functions provided in pycraf: `~pycraf.pathprof.geoid_inverse` and
    `~pycraf.pathprof.geoid_direct`. They solve the "Geodesics problem" using
    `Vincenty’s formulae
    <https://en.wikipedia.org/wiki/Vincenty's_formulae>`_.
    The underlying method is based on an iterative approach
    and is used to find the distance and relative bearings between two points
    (P1, and P2) on the Geoid (Earth ellipsoid), or - if P1, the bearing, and
    distance are given -, it finds P2. The geodesics are the equivalent of
    great-circle paths on a sphere, but on the Geoid.

Another useful feature of the Geodesics functionality is that one can query
the Geographical coordinates of points at a certain distance from a central
point. On the sphere, they would all be located on a circle, but on Earth
it's different:

.. plot::
    :include-source:

    import numpy as np
    import matplotlib.pyplot as plt
    from astropy import units as u
    from pycraf import pathprof, geometry

    lon, lat = 0 * u.deg, 70 * u.deg
    bearings = np.linspace(-180, 180, 721) * u.deg
    distance = 1000 * u.km

    lons, lats, _ = pathprof.geoid_direct(lon, lat, bearings, distance)
    ang_dist = geometry.true_angular_distance(lon, lat, lons, lats)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    cax = fig.add_axes((0.9, 0.1, 0.02, 0.8))
    sc = ax.scatter(
        lons.to(u.deg), lats.to(u.deg),
        c=ang_dist.to(u.deg), cmap='viridis'
        )
    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label('Angular distance [deg]')
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.set_aspect(1. / np.cos(np.radians(70)))
    ax.grid()

.. note::

    Make no mistake, the apparent distortion mostly comes from projecting the
    "sphere" to a flat projection. However, if one calculates the angular
    distances with the formula for the sphere, there is some deviation
    (from North to South). For many cases where only rough estimates are
    needed it will be sufficient to treat the Geoid as a normal sphere.

.. _pathprof-using-attenmaps:

Producing maps of path propagation loss
---------------------------------------

Often it is desired to generate maps of path attenuation values, e.g., to
quickly determine regions where the necessary separations between a
potential interferer and the victim terminal would be too small, potentially
leading to radio frequency interference (RFI).

The simple approach would be to create a `~pycraf.pathprof.PathProp` instance
for each pixel in the desired region (with the Tx being in the center of the map, and the Rx located at the other map pixels) and run the `~pycraf.pathprof.loss_complete` function accordingly. This is relatively slow.
Therefore, we added a faster alternative, `~pycraf.pathprof.atten_map_fast`. The idea is to generate the full height profiles only for the pixels on the map edges and re-use the arrays for the inner pixels with a clever hashing algorithm. The details of this are encapsulated in the `~pycraf.pathprof.height_map_data` function, such that the user doesn't need to understand what's going on under the hood:

.. plot::
   :include-source:

    from astropy import units as u
    from pycraf import pathprof, conversions as cnv


    def plot_atten_map(lons, lats, total_atten):

        import matplotlib.pyplot as plt

        vmin, vmax = -5, 195
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_axes((0., 0., 1.0, 1.0))
        cbax = fig.add_axes((0., 0., 1.0, .02))
        cim = ax.imshow(
            total_atten,
            origin='lower', interpolation='nearest', cmap='inferno_r',
            vmin=vmin, vmax=vmax,
            extent=(lons[0], lons[-1], lats[0], lats[-1]),
            )
        cbar = fig.colorbar(
            cim, cax=cbax, orientation='horizontal'
            )
        ax.set_aspect(abs(lons[-1] - lons[0]) / abs(lats[-1] - lats[0]))
        cbar.set_label(r'Path propagation loss')
        ctics = np.arange(30, 200, 30)
        cbar.set_ticks(ctics)
        cbar.ax.set_xticklabels(map('{:.0f} dB'.format, ctics))
        ax.set_xlabel('Longitude [deg]')
        ax.set_ylabel('Latitude [deg]')

        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.show()


    pathprof.SrtmConf.set(download='missing')

    lon_tx, lat_tx = 6.88361 * u.deg, 50.52483 * u.deg
    map_size_lon, map_size_lat = 0.5 * u.deg, 0.5 * u.deg
    map_resolution = 10. * u.arcsec

    freq = 1. * u.GHz
    omega = 0. * u.percent  # fraction of path over sea
    temperature = 290. * u.K
    pressure = 1013. * u.hPa
    timepercent = 2 * u.percent  # see P.452 for explanation
    h_tg, h_rg = 50 * u.m, 10 * u.m
    G_t, G_r = 0 * cnv.dBi, 0 * cnv.dBi
    zone_t, zone_r = pathprof.CLUTTER.UNKNOWN, pathprof.CLUTTER.UNKNOWN
    hprof_step = 100 * u.m

    hprof_cache = pathprof.height_map_data(
        lon_tx, lat_tx,
        map_size_lon, map_size_lat,
        map_resolution=map_resolution,
        zone_t=zone_t, zone_r=zone_r,
        )  # dict-like

    atten_maps, eps_pt_map, eps_pr_map = pathprof.atten_map_fast(
        freq,
        temperature,
        pressure,
        h_tg, h_rg,
        timepercent,
        hprof_cache,
        )

    lons = hprof_cache['xcoords']
    lats = hprof_cache['ycoords']
    # index 4 is total loss without clutter/gain included:
    total_atten = atten_maps[4]

    plot_atten_map(lons, lats, total_atten)

For a more illustrative example, have a look at the Jupyter `tutorial notebook
<https://github.com/bwinkel/pycraf/tree/master/notebooks/03c_attenuation_maps.ipynb>`_
on this topic.

See Also
========

- `Astropy Units and Quantities package <http://docs.astropy.org/en/stable/
  units/index.html>`_, which is used extensively in pycraf.
- `ITU-R Recommendation P.452-16 <https://www.itu.int/rec/
  R-REC-P.452-16-201507-I/en>`_

Reference/API
=============

.. automodapi:: pycraf.pathprof
    :no-inheritance-diagram:
    :include-all-objects:
    :skip: CLUTTER_DATA
    :skip: PARAMETERS_BASIC
    :skip: PARAMETERS_V14
    :skip: PARAMETERS_V16

Available clutter types
-----------------------

+-------+---------------------------+------+------+
| Value | Alias                     | |ha| | |dk| |
+=======+===========================+======+======+
| -1    | CLUTTER.UNKNOWN           | 0    | 0    |
+-------+---------------------------+------+------+
| 0     | CLUTTER.SPARSE            | 4    | 100  |
+-------+---------------------------+------+------+
| 1     | CLUTTER.VILLAGE           | 5    | 70   |
+-------+---------------------------+------+------+
| 2     | CLUTTER.DECIDIOUS_TREES   | 15   | 50   |
+-------+---------------------------+------+------+
| 3     | CLUTTER.CONIFEROUS_TREES  | 20   | 50   |
+-------+---------------------------+------+------+
| 4     | CLUTTER.TROPICAL_FOREST   | 20   | 30   |
+-------+---------------------------+------+------+
| 5     | CLUTTER.SUBURBAN          | 9    | 25   |
+-------+---------------------------+------+------+
| 6     | CLUTTER.DENSE_SUBURBAN    | 12   | 20   |
+-------+---------------------------+------+------+
| 7     | CLUTTER.URBAN             | 20   | 20   |
+-------+---------------------------+------+------+
| 8     | CLUTTER.DENSE_URBAN       | 25   | 20   |
+-------+---------------------------+------+------+
| 9     | CLUTTER.HIGH_URBAN        | 35   | 20   |
+-------+---------------------------+------+------+
| 10    | CLUTTER.INDUSTRIAL_ZONE   | 20   | 50   |
+-------+---------------------------+------+------+

.. |ha| replace:: :math:`h_\mathrm{a}~[\mathrm{m}]`
.. |dk| replace:: :math:`d_\mathrm{k}~[\mathrm{m}]`
