.. pycraf-conversions:

**************************************
Conversions (`pycraf.conversions`)
**************************************

.. |apu| replace:: `astropy.units <http://docs.astropy.org/en/stable/units/index.html>`_

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

    >>> from astropy import units as u
    >>> from pycraf import conversions as cnv

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

    >> cnv.eff_from_geom_area(10 * u.m ** 2, 50 * u.percent)
    <Quantity 5.0 m2>

    >> cnv.eff_from_geom_area(10 * u.m ** 2, 0.5 * cnv.dimless)
    <Quantity 5.0 m2>

    >> cnv.eff_from_geom_area(1 * u.km ** 2, 10 * u.percent)
    <Quantity 100000.0 m2>


.. warning::

    It is not possible to omit the unit, even if a quantity is dimensionless

pycraf would raise an Exception if one tried::

    >> cnv.eff_from_geom_area(10 * u.m ** 2, 0.5)
    TypeError: Argument 'eta_a' to function 'eff_from_geom_area' has no
    'unit' attribute. You may want to pass in an astropy Quantity instead.


.. note::

    A Jupyter tutorial notebook about the `~pycraf.conversions` package is
    provided in the `pycraf repository <https://github.com/bwinkel/pycraf/blob/master/notebooks/01_conversions.ipynb>`_ on GitHub.

Using `pycraf.conversions`
==========================

.. toctree::
   :maxdepth: 2

   .. decibel_defs
   .. conversion_formulae

See Also
========

- `Astropy Units and Quantities package <http://docs.astropy.org/en/stable/
  units/index.html>`_, which is used extensively in pycraf.

Reference/API
=============

.. automodapi:: pycraf.conversions
