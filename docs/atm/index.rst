.. pycraf-atm:

**************************************
Atmospheric models (`pycraf.atm`)
**************************************

.. currentmodule:: pycraf.atm

Introduction
============

The `~pycraf.atm` sub-package can be used to calculate the atmospheric
attenuation, which is relevant for various purposes. Examples would be to
calculate the propagation loss associated with a terrestrial path, or a
so-called slant path between a terminal at large height (e.g., a satellite)
and an Earth terminal. The latter case is also of interest for observations
of deep-space objects, as Earth's atmosphere dampens the signal of interest.
Especially, in astronomy one needs to correct for such attenuation effects to
determine the original intensity of an object.

The relevant
`ITU-R Rec. P.676-10 <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_
contains two models. The Annex-1 model is more accurate and is valid for
frequencies between 1 and 1000 GHz. It constructs the attenuation spectrum
from the Debye continuum absorption and water and oxygen resonance lines. In
contrast, Annex-2 is based on a simpler empirical method, that works for
frequencies up to 350 GHz only. It doesn't work with layers but assumes just
a one-layer slab having an effective temperature, which would produce the
same overall loss as the multi-layered model. Of course, this effective
temperature has nothing to do with the true physical temperature (profile)
of the atmosphere. Although the Annex-2 model is simpler and thus somewhat
faster in principle, most cases users of `~pycraf` will want to use Annex-1
functions. These are mostly implemented in `Cython <https://cython.org/>`_
and therefore reasonably fast.

Both Annexes provide a framework to calculate the specific attenuation (dB /
km) for an atmospheric slab. To compute the full path loss caused by the
atmosphere over a longer path, one has to integrate over the distance. For
ground-to-ground stations this is trivial (assuming the specific attenuation
is constant - which would not be true, if one station is at a significantly
different altitude). For slant paths, Annex 1 and 2 methods differ
substanially. The former determines the specific attenuation for potentially
hundreds of different layers and as such height profiles for temperature and
pressures are necessary. These are provided in `ITU-R Rec. P.835-5
<https://www.itu.int/rec/R-REC-P.835-5-201202-I/en>`_. The `~pycraf.atm` sub-
package contains the "Standard profile" (`~pycraf.atm.profile_standard`) and
five more specialized profiles, associated with the geographic latitude and
season (`~pycraf.atm.profile_highlat_summer`,
`~pycraf.atm.profile_highlat_winter`, `~pycraf.atm.profile_midlat_summer`,
`~pycraf.atm.profile_midlat_winter`, and `~pycraf.atm.profile_lowlat`).



Using `pycraf.atm`
==================

Height profiles
---------------

Let's start by having a look at the atmospheric standard height profile:

.. plot::
    :context: reset
    :include-source:

    import numpy as np
    import matplotlib.pyplot as plt
    from astropy import units as u
    from pycraf import atm
    from pycraf import conversions as cnv

    # define height grid
    height_grid = np.arange(0, 85, 0.1) * u.km

    # query profile_standard function
    hprof = atm.profile_standard(height_grid)

    # Plot various quantities
    _heights = height_grid.to(u.km).value

    plt.close()
    fig, axes = plt.subplots(2, 3, figsize=(12, 10))
    axes[0, 0].plot(hprof.temperature.to(u.K).value, _heights, 'k-')
    axes[0, 0].set_xlabel('Temperature [K]')
    axes[0, 0].set_xlim((160, 300))
    axes[0, 1].plot(hprof.pressure.to(u.hPa).value, _heights, 'b-', label='Total')
    axes[0, 1].plot(hprof.pressure_water.to(u.hPa).value, _heights, 'r-', label='Wet')
    axes[0, 1].legend(
        *axes[0, 1].get_legend_handles_labels(),
        loc='upper right', fontsize=10
        )
    axes[0, 1].set_xlabel('Pressure [hPa]')
    axes[0, 1].semilogx()
    axes[0, 1].set_xlim((1.e-6, 1100))
    axes[0, 2].plot(hprof.rho_water.to(u.g / u.cm ** 3).value, _heights, 'k-')
    axes[0, 2].set_xlabel('Water density [g / cm^3]')
    axes[0, 2].semilogx()
    #ax3.set_xlim((1.e-3, 1100))
    axes[1, 0].plot(hprof.ref_index.to(cnv.dimless).value - 1., _heights, 'k-')
    axes[1, 0].set_xlabel('Refractive index - 1')
    axes[1, 0].semilogx()
    #ax3.set_xlim((1.e-3, 1100))
    axes[1, 1].plot(hprof.humidity_water.to(u.percent).value, _heights, 'k-')
    axes[1, 1].set_xlabel('Relative humidity, water [%]')
    axes[1, 2].plot(hprof.humidity_ice.to(u.percent).value, _heights, 'k-')
    axes[1, 2].set_xlabel('Relative humidity, ice [%]')
    for idx, ax in enumerate(axes.flat):
        ax.set_ylim((0, 86))
        if idx % 3 == 0:
            ax.set_ylabel('Height [km]')
        ax.grid()


    fig.suptitle(
        'Atmospheric standard profile after ITU R-P.835-5, Annex 1',
        fontsize=16
        )
    plt.show()

Here the function `~pycraf.atm.profile_standard` returns a
`~collections.namedtuple` with the plotted quantities. This works in the same
way for the other special height profiles, provided in pycraf. Likewise, the
user could define their own profiles, which will seamlessly work with all
other functions in this package, as long as they use a compatible function
signature (including return type) and extend to at least 85 km height.

Specific attenuation
--------------------

Given certain physical conditions such as temperature, pressure and water
content, one can calculate the specific attenuation in the following manner.
As the `ITU-R Rec. P.676-10`_ methods work with the partial dry and wet air
pressure, one usually has to derive these, e.g., from water content or
humidity.

.. plot::
    :context:
    :include-source:

    _freqs = np.arange(1, 1000, 1)
    freq_grid = _freqs * u.GHz
    # weather parameter at ground-level
    total_pressure = 1013 * u.hPa
    temperature = 290 * u.K
    # water content is from the integral over the full height
    rho_water = 7.5 * u.g / u.m ** 3

    # alternatively, one can use atm.pressure_water_from_humidity
    pressure_water = atm.pressure_water_from_rho_water(temperature, rho_water)
    pressure_dry = total_pressure - pressure_water

    print(
        'Oxygen pressure: {0.value:.2f} {0.unit}, '
        'Water vapor partial pressure: {1.value:.2f} {1.unit}'.format(
            pressure_dry, pressure_water
        ))

    atten_dry, atten_wet = atm.atten_specific_annex1(
        freq_grid, pressure_dry, pressure_water, temperature
        )

    plt.close()
    plt.figure(figsize=(12, 7))
    plt.plot(
        _freqs, atten_dry.to(cnv.dB / u.km).value,
        'r-', label='Dry air'
        )
    plt.plot(
        _freqs, atten_wet.to(cnv.dB / u.km).value,
        'b-', label='Wet air'
        )
    plt.plot(
        _freqs, (atten_dry + atten_wet).to(cnv.dB / u.km).value,
        'k-', label='Total'
        )
    plt.semilogy()
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Specific Attenuation [dB / km]')
    plt.xlim((1, 999))
    plt.ylim((5.e-3, 0.9e5))
    plt.grid()
    plt.legend(*plt.gca().get_legend_handles_labels(), loc='upper left')
    plt.title(
        'Specific attenuation for standard conditions, '
        'according to ITU-R P.676 (10), annex 1',
        fontsize=16
        )
    plt.show()

Total attenuation
-----------------
With the specific attenuation, one can infer the total attenuation along
a terrestrial or slant path. For the latter scenario, one of course needs
to calculate the specific attenuation for each atmospheric layer if working
with Annex-1 formulas. If many calculations are to be done with the same
atmospheric profile, the calculation of the physical parameters, including
the frequency-dependent specific attenuation, would consume a lot of
computing time. Therefore, `~pycraf.atm` Annex-1 functions use a dictionary
to cache the height profiles. It has to be created once for a given
atmospheric profile, by calling `~pycraf.atm.atm_layers` and can then be used
for all subsequent calculations, e.g., by the `~pycraf.atm.atten_slant_annex1`
function, which will do ray-tracing through the atmosphere and determine
the overall atmospheric attenuation along the path.

.. plot::
    :context:
    :include-source:

    obs_alt = 300 * u.m
    _freqs = np.arange(0.25, 100, 0.5)
    freq_grid = _freqs * u.GHz

    cases = [
        # elevation, profile, label, linestyle
        (90 * u.deg, atm.profile_highlat_winter, 'Winter, Elevation: 90 deg', 'b-'),
        (90 * u.deg, atm.profile_highlat_summer, 'Summer, Elevation: 90 deg', 'r-'),
        (15 * u.deg, atm.profile_highlat_winter, 'Winter, Elevation: 15 deg', 'b--'),
        (15 * u.deg, atm.profile_highlat_summer, 'Summer, Elevation: 15 deg', 'r--'),
        ]

    plt.close()
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for elev, profile, label, linestyle in cases:

        atm_layers_cache = atm.atm_layers(freq_grid, profile)
        total_atten, refraction, tebb = atm.atten_slant_annex1(
            elev, obs_alt, atm_layers_cache, t_bg=2.73 * u.K
            )
        opacity = atm.opacity_from_atten(total_atten, elev)

        print('Refraction for {}: {:.1f}'.format(label, refraction.to(u.arcsec)))

        axes[0, 0].plot(_freqs, total_atten.to(cnv.dB).value, linestyle, label=label)
        axes[0, 1].plot(_freqs, (-total_atten).to(cnv.dimless).value, linestyle, label=label)
        axes[1, 0].plot(_freqs, opacity.to(cnv.dimless).value, linestyle, label=label)
        axes[1, 1].plot(_freqs, tebb.to(u.K).value, linestyle, label=label)

    axes[0, 0].semilogy()
    axes[1, 0].semilogy()
    axes[0, 0].legend(*axes[0, 0].get_legend_handles_labels(), loc='upper left', fontsize=8)
    axes[0, 0].set_ylabel('Total attenuation [dB]')
    axes[0, 1].set_ylabel('Total gain')
    axes[1, 0].set_ylabel('Zenith opacity')
    axes[1, 1].set_ylabel('Tebb [K]')
    axes[0, 0].set_ylim((2e-2, 9e2))
    axes[0, 1].set_ylim((0, 1))
    axes[1, 0].set_ylim((3e-3, 9e1))
    axes[1, 1].set_ylim((0, 310))

    for idx, ax in enumerate(axes.flat):
        ax.grid()
        ax.set_xlim((1, 99))
        if idx >= 2:
            ax.set_xlabel('Frequency [GHz]')

    plt.show()

.. note::

    Note, to convert between total attenuation, :math:`\gamma`, and (zenith!)
    opacity, :math:`\tau` (used in astronomy), use

    .. math::

        \gamma [\mathrm{dB}] = 10\log_{10} \gamma

        \gamma = 10^{\gamma [\mathrm{dB}] / 10}

        \gamma = e^{-\tau\cdot \mathrm{AM}},~ \mathrm{AM}\approx\frac{1}{\sin\delta}

        \tau = -\frac{1}{\mathrm{AM}}\ln\gamma


It is clear, the for very realistic results, one should work with an actual
height profile valid for the day of observation, such as measured with a radio
sonde for example.


Ray-tracing, refraction, and path-finding
-----------------------------------------
The function `~pycraf.atm.atten_slant_annex1`, which was demonstrated above,
internally calls a helper routine `~pycraf.atm.raytrace_path` to work out
the geometry of a path starting at a given altitude (above sea level) and
with a certain elevation. As the refractive index of each atmospheric layer
is slightly distinct, `Snell's law <https://en.wikipedia.org/wiki/Snell%27s_law>`_ applies and bends the ray somewhat as it travels from layer
to layer. The overall bending angle is called "refraction" and thus a
sources appears at a slightly different elevation angle for an observer.

If you are interested in how such a path looks like, the following
demonstrates how to plot it.

.. plot::
    :context:
    :include-source:

    # first, we need to create the atmospheric layer cache
    atm_layers_cache = atm.atm_layers([1] * u.GHz, atm.profile_highlat_winter)

    plt.close()
    fig = plt.figure(figsize=(12, 6))

    # to plot the atmospheric layers, we need to access the layers_cache:
    a_e = atm.EARTH_RADIUS
    layer_angles = np.arange(0, 0.1, 1e-3)
    layer_radii = atm_layers_cache['radii']
    bottom, top = layer_radii[[0, 900]]
    plt.plot(bottom * np.sin(layer_angles), bottom * np.cos(layer_angles), 'k-')
    plt.plot(top * np.sin(layer_angles), top * np.cos(layer_angles), 'k-')
    # we only plot some layers
    for r in layer_radii[[200, 500, 600, 700, 800, 850]]:
        plt.plot(r * np.sin(layer_angles), r * np.cos(layer_angles), 'k--', alpha=0.5)

    # now create four different example paths (of different type)
    for path_num, elevation, obs_alt, max_path_length in zip(
            [1, 2, 3, 4],
            [10, 20, -5, -45] * u.deg,
            [300, 300, 25000, 50000] * u.m,
            [1000, 230, 300, 1000] * u.km,
            ):

        path_params, _, refraction = atm.raytrace_path(
            elevation, obs_alt, atm_layers_cache,
            max_path_length=max_path_length,
            )

        print('total path length {:d}: {:5.1f}'.format(
            path_num, np.sum(path_params.a_n)
            ))

        radii = path_params.r_n
        angle = path_params.delta_n
        x, y = radii * np.sin(angle), radii * np.cos(angle)
        plt.plot(x, y, '-', label='Path {:d}'.format(path_num))

    plt.legend(*plt.gca().get_legend_handles_labels())
    plt.xlim((0, 290))
    plt.ylim((a_e - 5, 6453))
    plt.title('Path propagation through layered atmosphere')
    plt.xlabel('Projected distance (km)')
    plt.ylabel('Distance to Earth center (km)')
    plt.gca().set_aspect('equal')
    plt.show()

As you can see, `~pycraf.atm.raytrace_path` allows to specify a maximal path
length (as well as a maximal separation angle, see API docs), which can be
useful for terrestrial paths.

.. note::

    A Jupyter tutorial notebook about the `~pycraf.atm` package is
    provided in the `pycraf repository <https://github.com/bwinkel/pycraf/blob/master/notebooks/02_atmospheric_attenuation.ipynb>`_ on GitHub. It also
    contains plots of the specialized height profiles, as well as an example
    for the Annex-2 method.


See Also
========

- `Astropy Units and Quantities package <http://docs.astropy.org/en/stable/
  units/index.html>`_, which is used extensively in pycraf.
- `ITU-R Rec. P.453-12 <https://www.itu.int/rec/R-REC-P.453-12-201609-I/en>`_
- `ITU-R Rec. P.835-5 <https://www.itu.int/rec/R-REC-P.835-5-201202-I/en>`_
- `ITU-R Rec. P.676-10 <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_

Reference/API
=============

.. automodapi:: pycraf.atm
