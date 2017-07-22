.. pycraf-protection:

****************************************
Protection levels (`pycraf.protection`)
****************************************

.. currentmodule:: pycraf.protection

Introduction
============

The `~pycraf.protection` sub-package is a small convenience module, which
contains protection levels (the bare numbers) for various scenarios.
At the moment, the radio-astronomy service (RAS) protection thresholds
as given in `ITU-R Rec. RA.769
<https://www.itu.int/rec/R-REC-RA.769-2-200305-I/en>`_ (Table 1 and 2),
as well as industry-level device emission limits according to
`CISPR-11 (EN 55011) <http://rfemcdevelopment.eu/index.php/en/emc-emi-standards/en-55011-2009>`_ and
`CISPR-22 (EN 55022) <http://www.rfemcdevelopment.eu/en/en-55022-2010>`_
are provided.


Using `pycraf.protection`
=========================

The function `~pycraf.protection.ra769_limits` returns a
`~astropy.table.Table` object that resembles (Table 1 and 2) from
`ITU-R Rec. RA.769 <https://www.itu.int/rec/R-REC-RA.769-2-200305-I/en>`_::

    >>> from pycraf import protection  # doctest: +SKIP
    >>> protection.ra769_limits()  # doctest: +SKIP
    <Table length=21>
    frequency bandwidth   T_A   ...     Slim_nu        Efield    Efield_norm
       MHz       MHz       K    ... dB(W / (Hz m2)) dB(uV2 / m2) dB(uV2 / m2)
     float64   float64  float64 ...     float64       float64      float64
    --------- --------- ------- ... --------------- ------------ ------------
           13         0   50000 ...          -247.6        -54.9        -41.9
           26         0   15000 ...          -249.1        -52.5        -43.3
           74         2     750 ...          -258.2        -50.4        -52.5
          152         3     150 ...          -259.2        -48.7        -53.4
          325         7      40 ...          -257.5        -43.5        -51.7
          408         4      25 ...          -255.1        -43.4        -49.3
          ...       ...     ... ...             ...          ...          ...
        31550       500      18 ...          -228.0          4.8        -22.2
        43000      1000      25 ...          -226.4          9.3        -20.7
        89000      8000      12 ...          -227.9         16.8        -22.2
       150000      8000      14 ...          -223.2         21.6        -17.4
       224000      8000      20 ...          -218.2         26.6        -12.4
       270000      8000      25 ...          -215.8         29.0        -10.0

The two functions `~pycraf.protection.cispr11_limits` and
`~pycraf.protection.cispr22_limits` have a different call signature (see
API reference), but are similarly easy to use::

    >>> from pycraf import protection  # doctest: +SKIP
    >>> import astropy.units as u  # doctest: +SKIP
    >>> protection.cispr11_limits(  # doctest: +SKIP
    ...     500 * u.MHz, detector_type='QP', detector_dist=100 * u.m
    ...     )
    (<Decibel [ 26.54242509] dB(uV2 / m2)>, <Quantity 120.0 kHz>)

These return a tuple with the electrical field limit and the detector
bandwidth (currently always 120 kHz) for a frequency and distance. Two
detector types are available: Quasi-Peak (QP) and RMS.

See Also
========

- `Astropy Units and Quantities package <http://docs.astropy.org/en/stable/
  units/index.html>`_, which is used extensively in pycraf.
- `ITU-R Rec. RA.769 <https://www.itu.int/rec/R-REC-RA.769-2-200305-I/en>`_
- `CISPR-11 (EN 55011) <http://rfemcdevelopment.eu/index.php/en/emc-emi-standards/en-55011-2009>`_
- `CISPR-22 (EN 55022) <http://www.rfemcdevelopment.eu/en/en-55022-2010>`_

Reference/API
=============

.. automodapi:: pycraf.protection
