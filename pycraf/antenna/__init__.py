#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This sub-package provides various (simplified) antenna patterns for use in
compatibility studies. At the moment, the following are implemented:

- IMT phased array antenna patterns for base station and mobile devices to
  be used for AI 1.13 IMT2020 studies. These are defined in the so-called
  IMT.MODEL document (`ITU-R TG 5/1 document 5-1/36 <https://www.itu.int/md/R15-TG5.1-C-0036>`_).
- A pattern that can be used for radio telescopes (`ITU-R Rec. RA.1631-0
  <https://www.itu.int/rec/R-REC-RA.1631-0-200305-I/en>`_).
- Pattern for fixed wireless systems ("fixed-links"), as defined in
  `ITU-R Rec. F.699-7 <https://www.itu.int/rec/R-REC-F.699-7-200604-I/en>`_.
'''

from .imt import *
from .ras import *
from .fixedlink import *
