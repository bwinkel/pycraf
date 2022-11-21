.. pycraf-conversions:

**************************************
Conversions (`pycraf.conversions`)
**************************************

.. currentmodule:: pycraf.conversions

Introduction
============

The `~pycraf.conversions` sub-package contains various functions to convert
and calculate power flux densities, electrical field strengths, transmitted
and received powers from each other. Routines that link antenna areas and
gains are also present. With the `~pycraf.conversions.free_space_loss` function
one can determine the free-space loss for a given frequency and distance
between transmitter (Tx) and receiver (Rx). Furthermore, several useful
Decibel units are defined (see `Reference/API`_).

Getting Started
===============

Using the `~pycraf.conversions` package is really simple::

    >>> from pycraf import conversions as cnv
    >>> from astropy import units as u

    >>> A_eff = 10 * u.m ** 2  # effective area
    >>> A_geom = 20 * u.m ** 2  # geometric area
    >>> eta_a = 50 * u.percent  # antenna efficiency

    >>> print('A_eff = {0.value:.1f} {0.unit}'.format(
    ...    cnv.eff_from_geom_area(A_geom, eta_a))
    ...    )
    A_eff = 10.0 m2

    >>> print('A_geom = {0.value:.1f} {0.unit}'.format(
    ...    cnv.geom_from_eff_area(A_eff, eta_a))
    ...    )
    A_geom = 20.0 m2

Because all function parameters and return values are Astropy Quantities
(see `~astropy.units.Quantity`), unit conversion is automatically performed::

    >>> cnv.eff_from_geom_area(10 * u.m ** 2, 50 * u.percent)  # doctest: +FLOAT_CMP
    <Quantity 5.0 m2>

    >>> cnv.eff_from_geom_area(10 * u.m ** 2, 0.5 * cnv.dimless)  # doctest: +FLOAT_CMP
    <Quantity 5.0 m2>

    >>> cnv.eff_from_geom_area(1 * u.km ** 2, 10 * u.percent)  # doctest: +FLOAT_CMP
    <Quantity 100000.0 m2>


.. warning::

    It is not possible to omit the unit, even if a quantity is dimensionless!

    pycraf would raise an exception if one tried::

        >>> cnv.eff_from_geom_area(10 * u.m ** 2, 0.5)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
          ...
        TypeError: Argument 'eta_a' to function 'eff_from_geom_area' has no
        'unit' attribute. You may want to pass in an astropy Quantity instead.


.. note::

    A Jupyter tutorial notebook about the `~pycraf.conversions` package is
    provided in the `pycraf repository <https://github.com/bwinkel/pycraf/blob/master/notebooks/01_conversions.ipynb>`_ on GitHub.

Using `pycraf.conversions`
==========================

Decibel units
-------------
In the `~pycraf.conversions` module, several Decibel units are defined
that often occur in spectrum-management tasks. Examples are the
:math:`\mathrm{dB}` unit or :math:`\mathrm{dBm}\equiv\mathrm{dB[mW]}`. In
Python these are defined with the help of the `~astropy.units` package, e.g.::

    >>> from astropy import units as u

    >>> dB_W = u.dB(u.W)
    >>> dB_W_m2_Hz = u.dB(u.W / u.m ** 2 / u.Hz)
    >>> dB_uV_m = u.dB(u.uV ** 2 / u.m ** 2)

.. note::

    The :math:`\mathrm{dB}[\mu\mathrm{V}^2 / \mathrm{m}^2]` unit is a bit
    special. Engineers will usually call this symbol
    :math:`\mathrm{dB}[\mu\mathrm{V} / \mathrm{m}]`, but strictly speaking
    it is the power of the electrical field which is referred to,
    which means that the amplitude of the E-field has to be squared.

    In fact, if one would omit the squaring from the definition, the unit
    would not work in the equations, because the `~astropy.units` framework
    would notice an inconsistency of the units.


Working with the Decibel units is easy::

    >>> from pycraf import conversions as cnv

    >>> power = 1 * cnv.dB_W
    >>> power  # doctest: +FLOAT_CMP
    <Decibel 1.0 dB(W)>

    >>> power.to(u.W)  # doctest: +FLOAT_CMP
    <Quantity 1.2589254117941673 W>
    >>> print(power.to(u.W))  # doctest: +FLOAT_CMP
    1.2589254117941673 W

Often, one wants to do some formatting in the print function, which works as
usual::

    >>> print('{:.1f}'.format(power.to(u.W)))
    1.3 W

Astropy `~astropy.units` provide even some additional formatting options::

    >>> print('{0.value:.1f} {0.unit}'.format(power.to(u.W)))
    1.3 W
    >>> print('{0.value:.1f} {0.unit:FITS}'.format(1 * u.m / u.s))
    1.0 m s-1

.. warning::

    Under some circumstances, the automatic conversion from the Decibel
    to the associated linear unit `can fail <https://github.com/astropy/astropy/issues/6319>`_. This may get fixed in a future
    version of Astropy. It helps to construct the Quantities via
    multiplication, avoiding the `~astropy.units.Quantity` constructor, i.e.,
    :code:`1 * cnv.dB` instead of :code:`u.Quantity(1, cnv.dB)`.


Conversion formulae
-------------------

For a complete list of the functions provided in the `~pycraf.conversions`
package, see the `Reference/API`_ section below. All functions expect
the parameters to have appropriate units, and will return a
`~astropy.units.Quantity`, i.e. a value with a unit. This makes it sometimes
a bit verbose to write code, but has the invaluable advantage that one
avoids mistakes in the calculations that come from using the wrong units.

For example, in the various ITU-R Recommendations, the described algorithms
often assume the quantities to be in a certain unit. In pycraf all functions
are defined in a way, that input parameters are first converted to the
correct unit before feeding the values into the ITU-R algorithms.

The following quantities are of interest in compatibility studies:

- Distance between transmitter and receiver: :math:`d`
- Frequency of radiation: :math:`f`
- Wavelength of radiation: :math:`\lambda = \frac{c}{f}`
- Geometric antenna area: :math:`A_\mathrm{geom}`
- Effective antenna area: :math:`A_\mathrm{eff} = \eta_\mathrm{A} A_\mathrm{geom}`; isotropic loss-less antenna: :math:`A_\mathrm{eff} = \frac{\lambda^2}{4\pi}`
- Antenna efficiency: :math:`\eta_\mathrm{A}`
- Antenna temperature: :math:`T_\mathrm{A}=\Gamma S_\nu`
- Antenna sensitivity: :math:`\Gamma=\frac{A_\mathrm{eff}}{2k_\mathrm{B}}`
- Transmitter/receiver gain, :math:`G_\mathrm{tx}`, :math:`G_\mathrm{rx}`: :math:`A_\mathrm{eff} = G\frac{\lambda^2}{4\pi}`
- Pointing flux: :math:`\vec S = \vec E \times \vec H`
- Power flux density: :math:`S\equiv\vert \vec S \vert = \sqrt{\frac{\varepsilon_0}{\mu_0}} \vert \vec E \vert^2 = \frac{E^2}{R_0}`
  with :math:`R_0 \equiv \sqrt{\frac{\varepsilon_0}{\mu_0}} = 376.73~\Omega`
- Spectral power flux density, :math:`S_\nu`, with: :math:`S=\int \mathrm{d}\nu\,S_\nu`
- Transmitted power, :math:`P_\mathrm{tx}`, with :math:`S = G_\mathrm{tx}\frac{P_\mathrm{tx}}{4\pi d^2}`
- Received power, :math:`P_\mathrm{rx} = S\cdot A_\mathrm{eff}^\mathrm{rx} = S\cdot G_\mathrm{rx}\frac{\lambda^2}{4\pi} = G_\mathrm{tx}\frac{P_\mathrm{tx}}{4\pi d^2} \cdot G_\mathrm{rx}\frac{\lambda^2}{4\pi} = G_\mathrm{tx} G_\mathrm{rx} P_\mathrm{tx} L_\mathrm{fs}`
- Received spectral power, :math:`P_\mathrm{rx,\nu} = \frac{P_\mathrm{rx}}{\Delta\nu}=2 k_\mathrm{B}T_\mathrm{A}`
- Signal bandwidth, :math:`\Delta \nu`
- Free-space loss: :math:`L_\mathrm{fs} = \frac{c^2}{16\pi^2}\frac{1}{d^2f^2} = \frac{\lambda^2}{16\pi^2 d^2}`

A few examples::

    >>> from pycraf import conversions as cnv
    >>> from astropy import units as u

    >>> frequency = 10 * u.GHz
    >>> S = 10 * u.Jy * u.MHz
    >>> E_rx = -30 * cnv.dB_uV_m
    >>> distance = 10 * u.km
    >>> G_tx = 20 * cnv.dBi
    >>> G_rx = 10 * cnv.dBi
    >>> P_rx = -10 * cnv.dBm
    >>> P_tx = 20 * cnv.dBm

    >>> cnv.efield_from_powerflux(S).to(cnv.dB_uV_m)  # doctest: +FLOAT_CMP
    <Decibel -44.23969433034762 dB(uV2 / m2)>
    >>> cnv.powerflux_from_efield(E_rx).to(cnv.dB_W_m2)  # doctest: +FLOAT_CMP
    <Decibel -175.7603056696524 dB(W / m2)>

    >>> cnv.ptx_from_powerflux(S, distance, G_tx).to(cnv.dB_W)  # doctest: +FLOAT_CMP
    <Decibel -119.00790135977903 dB(W)>
    >>> cnv.powerflux_from_ptx(P_tx, distance, G_tx).to(cnv.dB_W_m2)  # doctest: +FLOAT_CMP
    <Decibel -80.99209864022097 dB(W / m2)>

    >>> cnv.prx_from_powerflux(S, frequency, G_rx).to(cnv.dB_W)  # doctest: +FLOAT_CMP
    <Decibel -221.4556845816624 dB(W)>
    >>> cnv.powerflux_from_prx(P_rx, frequency, G_rx).to(cnv.dB_W_m2)  # doctest: +FLOAT_CMP
    <Decibel -8.544315418337588 dB(W / m2)>

    >>> cnv.free_space_loss(distance, frequency)  # doctest: +FLOAT_CMP
    <Decibel -132.44778322188336 dB>

    >>> cnv.prx_from_ptx(P_tx, G_tx, G_rx, distance, frequency).to(cnv.dB_W)  # doctest: +FLOAT_CMP
    <Decibel -112.44778322188337 dB(W)>
    >>> cnv.ptx_from_prx(P_rx, G_tx, G_rx, distance, frequency).to(cnv.dB_W)  # doctest: +FLOAT_CMP
    <Decibel 62.44778322188338 dB(W)>

See Also
========

- `Astropy Units and Quantities package <http://docs.astropy.org/en/stable/
  units/index.html>`_, which is used extensively in pycraf.

Reference/API
=============

.. automodapi:: pycraf.conversions
