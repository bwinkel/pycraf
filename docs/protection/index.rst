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

RA.769
------

The function `~pycraf.protection.ra769_limits` returns a
`~astropy.table.QTable` object that resembles (Table 1 to 3) from
`ITU-R Rec. RA.769 <https://www.itu.int/rec/R-REC-RA.769-2-200305-I/en>`_::

    >>> from pycraf import protection
    >>> import astropy.units as u

    >>> protection.ra769_limits(mode='continuum')
    <QTable length=21>
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

For VLBI the returned table object looks a bit different::

    >>> protection.ra769_limits(mode='vlbi')
    <QTable length=21>
    frequency   T_A     T_rx      Slim_nu
       MHz       K       K    dB(W / (Hz m2))
     float64  float64 float64     float64
    --------- ------- ------- ---------------
           13   50000      60          -217.6
           26   15000      60          -217.2
           74     750      60          -220.7
          152     150      60          -220.3
          325      40      60          -216.9
          408      25      60          -215.6
          ...     ...     ...             ...
        31550      18      65          -178.0
        43000      25      65          -174.9
        89000      12      30          -171.9
       150000      14      30          -167.2
       224000      20      43          -162.1
       270000      25      50          -159.8

This is because the VLBI thresholds are not based on the RMS (after a certain
integration time and with a certain bandwidth), but on the system temperature
(typical receiver noise values and antenna temperatures). The thresholds are
defined as 1% of Tsys.

For continuum and spectral modes, it is possible to use a different
integration time::

  >>> plim_2000s = protection.ra769_limits(mode='continuum')[7]['Plim']
  >>> for itime in [15 * u.min, 1 * u.h, 2 * u.h, 5 * u.h, 10 * u.h]:
  ...     tab = protection.ra769_limits(mode='continuum', integ_time=itime)
  ...     # print values of footnote (1) of tables 1 and 2 in RA.769
  ...     print('{:.1f}'.format(tab[7]['Plim'] - plim_2000s))
  1.7 dB
  -1.3 dB
  -2.8 dB
  -4.8 dB
  -6.3 dB


CISPR limits
------------

The two functions `~pycraf.protection.cispr11_limits` and
`~pycraf.protection.cispr22_limits` have a different call signature (see
API reference), but are similarly easy to use::

    >>> protection.cispr11_limits(  # doctest: +FLOAT_CMP
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
