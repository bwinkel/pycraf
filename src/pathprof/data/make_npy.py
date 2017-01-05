#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os


BASEPATH = 'R-REC-P.452-16-201507'


def main():

    lons = np.genfromtxt(os.path.join(BASEPATH, 'LON.TXT'))
    lats = np.genfromtxt(os.path.join(BASEPATH, 'LAT.TXT'))
    dn50 = np.genfromtxt(os.path.join(BASEPATH, 'DN50.TXT'))
    n050 = np.genfromtxt(os.path.join(BASEPATH, 'N050.TXT'))


    # import matplotlib.pyplot as plt

    # plt.contourf(lons, lats, dn50, 128)
    # plt.show()

    # plt.contourf(lons, lats, n050, 128)
    # plt.show()

    save_kwargs = {
        'lons': lons.astype(np.float32),
        'lats': lats.astype(np.float32),
        'dn50': dn50.astype(np.float32),
        'n050': n050.astype(np.float32),
        }
    np.savez('refract_map', **save_kwargs)

    # read using:
    # dat = np.load('refract_map.npz')


if __name__ == '__main__':
    main()
