#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Contains functions to convert power flux densities, field strengths,
transmitted and received powers at certain distances and frequencies into each
other. Furthermore, some convenience decibel-Scales are defined.

The following Decibel units are available:


================ ============== ======================================
      Unit             Alias         Definition
================ ============== ======================================
|dimless|        dimless        |u|.dimensionless_unscaled
|dB_ic|          dB, dBi, dBc   |u|.dB(dimless)
|dBW|            dB_W           |u|.dB(|u|.W)
|dBW_Hz|         dB_W_Hz        |u|.dB(|u|.W / |u|.Hz)
|dBW_m2|         dB_W_m2        |u|.dB(|u|.W / |u|.m ** 2)
|dBW_m2_Hz|      dB_W_m2_Hz     |u|.dB(|u|.W / |u|.m ** 2 / |u|.Hz)
|dBW_Jy_Hz|      dB_Jy_Hz       |u|.dB(|u|.Jy * |u|.Hz)
|dBm|            dBm = dB_mW    |u|.dB(|u|.mW)
|dBm_MHz|        dBm_MHz        |u|.dB(|u|.mW / |u|.MHz)
|dB_uV_m|        dB_uV_m        |u|.dB(|u|.uV ** 2 / |u|.m ** 2)
|dB_1_m|         dB_1_m         |u|.dB(1 / |u|.m)
================ ============== ======================================

u = `astropy.units <http://docs.astropy.org/en/stable/units/index.html>`_

.. |u| replace:: u
.. |dimless| replace:: :math:`1`
.. |dB_ic| replace:: :math:`\mathrm{dB},~\mathrm{dBi},~\mathrm{dBc}`
.. |dBW| replace:: :math:`\mathrm{dB}[\mathrm{W}]`
.. |dBW_Hz| replace:: :math:`\mathrm{dB}[\mathrm{W} / \mathrm{Hz}]`
.. |dBW_m2| replace:: :math:`\mathrm{dB}[\mathrm{W} / \mathrm{m}^2]`
.. |dBW_m2_Hz| replace:: :math:`\mathrm{dB}[\mathrm{W} / \mathrm{m}^2 / \mathrm{Hz}]`
.. |dBW_Jy_Hz| replace:: :math:`\mathrm{dB}[\mathrm{Jy} \cdot \mathrm{Hz}]`
.. |dBm| replace:: :math:`\mathrm{dB}[\mathrm{mW}]`
.. |dBm_MHz| replace:: :math:`\mathrm{dB}[\mathrm{mW} / \mathrm{MHz}]`
.. |dB_uV_m| replace:: :math:`\mathrm{dB}[\mu\mathrm{V}^2 / \mathrm{m}^2]`
.. |dB_1_m| replace:: :math:`\mathrm{dB}[1 / \mathrm{m}]`
'''

from .conversions import *
