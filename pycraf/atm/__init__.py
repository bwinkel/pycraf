#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
The atm subpackage provides an implementation of the atmospheric models of
`ITU-R Rec. P.676-11 <https://www.itu.int/rec/R-REC-P.676-11-201609-I/en>`_.
For this, various other algorithms from the following two ITU-R
Recommendations are
necessary:

    - `ITU-R Rec. P.835-5 <https://www.itu.int/rec/R-REC-P.835-5-201202-I/en>`_,
      which contains the Standard and several special atmospheric profiles.
    - `ITU-R Rec. P.453-12 <https://www.itu.int/rec/R-REC-P.453-12-201609-I/en>`_,
      which is used to calculate the refractive index from temperature and
      water/total air pressure. Furthermore, P.453 has formulae to derive the
      saturation water pressure from temperature and total air pressure,
      as well as the water pressure from temperature, pressure and humidity,
      or alternatively from temperature and wator vapor density.

Notes
-----
The new version of P.676,
`ITU-R Rec. P.676-11 <https://www.itu.int/rec/R-REC-P.676-11-201609-I/en>`_, has updated the algorithms for Annex 2 (compare with https://www.itu.int/rec/R-REC-P.676-11-201609-I/en>). As the Annex 1 methods are more accurate (and
sufficiently fast), users should use these for their work. `~pycraf`
continues to provide the Annex 2 solutions of the P.676-10 for historical
reasons, but this should be considered as deprecated.
'''

from .atm import *
