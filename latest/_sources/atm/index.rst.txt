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
of the atmosphere.

Both Annexes provide a framework to calculate the specific attenuation
(dB / km) for an atmospheric slab. To compute the full path loss caused by
the atmosphere over a longer path, one has to integrate over the distance.
For ground-to-ground stations this is trivial (assuming the specific
attenuation is constant). For slant paths, Annex 1 and 2 differ
significantly. The former determines the specific attenuation for hundreds of
different layers and as such height profiles for temperature and pressures
are necessary. These are provided in
`ITU-R Rec. P.835-5 <https://www.itu.int/rec/R-REC-P.835-5-201202-I/en>`_.
The `~pycraf.atm` sub-package contains the "Standard profile"
(`~pycraf.atm.profile_standard`) and five more specialized profiles,
associated with the geographic latitude and season
(`~pycraf.atm.profile_highlat_summer`, `~pycraf.atm.profile_highlat_winter`,
`~pycraf.atm.profile_midlat_summer`, `~pycraf.atm.profile_midlat_winter`, and
`~pycraf.atm.profile_lowlat`).



Using `pycraf.atm`
==================

Height profiles
---------------

Let's start with having a look at the atmospheric standard height profile:

.. plot::
   :include-source:

    import numpy as np
    import matplotlib.pyplot as plt
    from astropy import units as u
    from pycraf import atm
    from pycraf import conversions as cnv

    # define height grid
    height_grid = np.arange(0, 85, 0.1) * u.km

    # query profile_standard function
    (
        temperatures,
        pressures,
        rho_water,
        pressures_water,
        ref_indices,
        humidities_water,
        humidities_ice
        ) = atm.profile_standard(height_grid)

    # Plot various quantities
    _heights = height_grid.to(u.km).value

    fig = plt.figure(figsize=(15, 13))
    axes = [fig.add_subplot(2, 3, i) for i in range(1, 7)]
    axes[0].plot(temperatures.to(u.K).value, _heights, 'k-')
    axes[0].set_xlabel('Temperature [K]')
    axes[0].set_xlim((160, 300))
    axes[1].plot(pressures.to(u.hPa), _heights, 'b-', label='Total')
    axes[1].plot(pressures_water.to(u.hPa), _heights, 'r-', label='Wet')
    axes[1].legend(
        *axes[1].get_legend_handles_labels(),
        loc='upper right', fontsize=10
        )
    axes[1].set_xlabel('Pressure [hPa]')
    axes[1].semilogx()
    axes[1].set_xlim((1.e-6, 1100))
    axes[2].plot(rho_water.to(u.g / u.cm ** 3).value, _heights, 'k-')
    axes[2].set_xlabel('Water density [g / cm^3]')
    axes[2].semilogx()
    #ax3.set_xlim((1.e-3, 1100))
    axes[3].plot(ref_indices.to(cnv.dimless).value - 1., _heights, 'k-')
    axes[3].set_xlabel('Refractive index - 1')
    axes[3].semilogx()
    #ax3.set_xlim((1.e-3, 1100))
    axes[4].plot(humidities_water.to(u.percent).value, _heights, 'k-')
    axes[4].set_xlabel('Relative humidity, water [%]')
    axes[5].plot(humidities_ice.to(u.percent).value, _heights, 'k-')
    axes[5].set_xlabel('Relative humidity, ice [%]')
    for idx, ax in enumerate(axes):
        ax.set_ylim((0, 86))
        if idx % 3 == 0:
            ax.set_ylabel('Height [km]')
        ax.grid()

    fig.suptitle(
        'Atmospheric standard profile after ITU R-P.835-5, Annex 1',
        fontsize=16
        )


Specific attenuation
--------------------

Given certain physical conditions such as temperature, pressure and water
content, one can calculate the specific attenuation:

.. plot::
   :include-source:

    import numpy as np
    import matplotlib.pyplot as plt
    from astropy import units as u
    from pycraf import atm
    from pycraf import conversions as cnv

    _freqs = np.arange(1, 1000, 1)
    freq_grid = _freqs * u.GHz
    total_pressure = 1013 * u.hPa
    temperature = 290 * u.K
    rho_water = 7.5 * u.g / u.m ** 3

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

    plt.figure(figsize=(15, 9))
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

Total attenuation
-----------------
With the specific attenuation, one can infer the total attenuation along
a terrestrial or slant path. For the latter scenario, one of course needs
to calculate the specific attenuation for each atmospheric layer if working
with Annex-1 formulas:

.. plot::
   :include-source:

    import numpy as np
    import matplotlib.pyplot as plt
    from astropy import units as u
    from pycraf import atm
    from pycraf import conversions as cnv

    elevation = 5 * u.deg
    obs_alt = 300 * u.m
    _freqs = np.arange(0.1, 100, 1)
    freq_grid = _freqs * u.GHz

    cases = [
        # elevation, profile, label, linestyle
        (90 * u.deg, atm.profile_highlat_winter, 'Winter, Elevation: 90 deg', 'b-'),
        (90 * u.deg, atm.profile_highlat_summer, 'Summer, Elevation: 90 deg', 'r-'),
        (15 * u.deg, atm.profile_highlat_winter, 'Winter, Elevation: 15 deg', 'b--'),
        (15 * u.deg, atm.profile_highlat_summer, 'Summer, Elevation: 15 deg', 'r--'),
        ]

    fig = plt.figure(figsize=(15, 16))
    axes = [fig.add_subplot(2, 2, i) for i in range(1, 5)]
    for elevation, profile, label,linestyle in cases:

        total_atten, refraction, tebb = atm.atten_slant_annex1(
            freq_grid, elevation, obs_alt, profile, t_bg=2.73 * u.K
            )
        opacity = atm.opacity_from_atten(total_atten, elevation)

        print(
            'Refraction for {}: {:.1f} arcsec'.format(
                label, refraction * 3600
            ))

        axes[0].plot(
            _freqs, total_atten.to(cnv.dB).value,
            linestyle, label=label
            )
        axes[1].plot(
            _freqs, (-total_atten).to(cnv.dimless).value,
            linestyle, label=label
            )
        axes[2].plot(
            _freqs, opacity.to(cnv.dimless).value,
            linestyle, label=label
            )
        axes[3].plot(
            _freqs, tebb.to(u.K).value,
            linestyle, label=label
            )

    axes[0].semilogy()
    axes[2].semilogy()
    axes[0].legend(
        *axes[0].get_legend_handles_labels(),
        loc='upper left', fontsize=8
        )
    axes[0].set_ylabel('Total attenuation [dB]')
    axes[1].set_ylabel('Total gain')
    axes[2].set_ylabel('Opacity')
    axes[3].set_ylabel('Tebb [K]')
    axes[0].set_ylim((2e-2, 9e2))
    axes[1].set_ylim((0, 1))
    axes[2].set_ylim((3e-3, 9e1))
    axes[3].set_ylim((0, 310))

    for idx, ax in enumerate(axes):
        ax.grid()
        ax.set_xlim((1, 99))
        if idx >= 2:
            ax.set_xlabel('Frequency [GHz]')

.. note::

    Note, to convert between total attenuation, :math:`\gamma`, and (zenith!)
    opacity, :math:`\tau` (used in astronomy), use

    .. math::

        \gamma [\mathrm{dB}] = 10\log_{10} \gamma

        \gamma = 10^{\gamma [\mathrm{dB}] / 10}

        \gamma = e^{-\tau\cdot \mathrm{AM}},~ \mathrm{AM}=\frac{1}{\sin\delta}

        \tau = -\frac{1}{\mathrm{AM}}\ln\gamma


It is clear, the for very realistic results, one should work with an actual
height profile valid for the day of observation, such as measured with a radio
sonde for example.

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
