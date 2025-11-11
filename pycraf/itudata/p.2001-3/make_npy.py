#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os


BASEPATH = 'R-REC-P.2001-3-201908'


def main():

    lons, lats = np.meshgrid(
        np.linspace(0, 360, 241), np.linspace(90, -90, 121),
        )

    dn_median = np.genfromtxt(os.path.join(BASEPATH, 'DN_Median.txt'))
    dn_supslope = np.genfromtxt(os.path.join(BASEPATH, 'DN_SupSlope.txt'))
    dn_subslope = np.genfromtxt(os.path.join(BASEPATH, 'DN_SubSlope.txt'))
    dn_dz = np.genfromtxt(os.path.join(BASEPATH, 'dndz_01.txt'))

    surfwv_50 = np.genfromtxt(os.path.join(BASEPATH, 'surfwv_50_fixed.txt'))

    foes_50 = np.genfromtxt(os.path.join(BASEPATH, 'FoEs50.txt'))
    foes_10 = np.genfromtxt(os.path.join(BASEPATH, 'FoEs10.txt'))
    foes_1 = np.genfromtxt(os.path.join(BASEPATH, 'FoEs01.txt'))
    foes_01 = np.genfromtxt(os.path.join(BASEPATH, 'FoEs0.1.txt'))

    h0 = np.genfromtxt(os.path.join(BASEPATH, 'h0.txt'))

    # import matplotlib.pyplot as plt

    # plt.contourf(lons, lats, dn_dz, 128)
    # plt.show()

    save_kwargs = {
        'lons': lons.astype(np.float32),
        'lats': lats.astype(np.float32),
        'dn_median': dn_median.astype(np.float32),
        'dn_supslope': dn_supslope.astype(np.float32),
        'dn_subslope': dn_subslope.astype(np.float32),
        'dn_dz': dn_dz.astype(np.float32),
        }
    np.savez('refract_map', **save_kwargs)

    save_kwargs = {
        'lons': lons.astype(np.float64),
        'lats': lats.astype(np.float64),
        'surfwv_50': surfwv_50.astype(np.float64),
        }
    np.savez('wv_map', **save_kwargs)

    save_kwargs = {
        'lons': lons.astype(np.float32),
        'lats': lats.astype(np.float32),
        'h0': h0.astype(np.float32),
        }
    np.savez('h0_map', **save_kwargs)

    save_kwargs = {
        'lons': lons.astype(np.float32),
        'lats': lats.astype(np.float32),
        'foes_50': foes_50.astype(np.float32),
        'foes_10': foes_10.astype(np.float32),
        'foes_1': foes_1.astype(np.float32),
        'foes_01': foes_01.astype(np.float32),
        }
    np.savez('sporadic_e_map', **save_kwargs)

    # read using:
    # dat = np.load('refract_map.npz')

    lons, lats = np.meshgrid(
        np.linspace(0, 360, 321), np.linspace(90, -90, 161),
        )

    pr6 = np.genfromtxt(os.path.join(BASEPATH, 'Esarain_Pr6_v5.txt'))
    mt = np.genfromtxt(os.path.join(BASEPATH, 'Esarain_Mt_v5.txt'))
    beta = np.genfromtxt(os.path.join(BASEPATH, 'Esarain_Beta_v5.txt'))

    # import matplotlib.pyplot as plt

    # plt.contourf(lons, lats, pr6, 128)
    # plt.show()

    save_kwargs = {
        'lons': lons.astype(np.float64),
        'lats': lats.astype(np.float64),
        'pr6': pr6.astype(np.float64),
        'mt': mt.astype(np.float64),
        'beta': beta.astype(np.float64),
        }
    np.savez('rain_map', **save_kwargs)

    tropoclim = np.genfromtxt(
        os.path.join(BASEPATH, 'TropoClim.txt'), dtype=np.int8
        )

    lons, lats = np.meshgrid(
        np.linspace(-179.75, 179.75, 720), np.linspace(89.75, -89.75, 360),
        )

    save_kwargs = {
        'lons': lons.astype(np.float32),
        'lats': lats.astype(np.float32),
        'tropoclim': tropoclim,  # use "nearest neighbor" for interpolation!
        }
    np.savez('tropoclim_map', **save_kwargs)


if __name__ == '__main__':
    main()
