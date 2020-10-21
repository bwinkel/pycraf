import numpy as np
import matplotlib.pyplot as plt
from pycraf import pathprof, conversions as cnv
from astropy import units as u


lon_tx, lat_tx = 6.8836 * u.deg, 50.525 * u.deg
lon_rx, lat_rx = 7.3334 * u.deg, 50.635 * u.deg
hprof_step = 100 * u.m  # resolution of height profile
omega = 0. * u.percent
temperature = 290. * u.K
pressure = 1013. * u.hPa
h_tg, h_rg = 5 * u.m, 50 * u.m
G_t, G_r = 0 * cnv.dBi, 15 * cnv.dBi
zone_t, zone_r = pathprof.CLUTTER.URBAN, pathprof.CLUTTER.SUBURBAN

frequency = np.array([0.1, 0.5, 1, 2, 5, 10, 20, 50, 100])
time_percent = np.logspace(-3, np.log10(50), 100)

# as frequency and time_percent are arrays, we need to add
# new axes to allow proper broadcasting
results = pathprof.losses_complete(
    frequency[:, np.newaxis] * u.GHz,
    temperature,
    pressure,
    lon_tx, lat_tx,
    lon_rx, lat_rx,
    h_tg, h_rg,
    hprof_step,
    time_percent[np.newaxis] * u.percent,
    zone_t=zone_t, zone_r=zone_r,
    )

fig, ax = plt.subplots(1, figsize=(8, 8))
L_b_corr = results['L_b_corr'].value
t = time_percent.squeeze()
lidx = np.argmin(np.abs(t - 2e-3))
for idx, f in enumerate(frequency.squeeze()):
    p = ax.semilogx(t, L_b_corr[idx], '-')
    ax.text(
        2e-3, L_b_corr[idx][lidx] - 1,
        '{:.1f} GHz'.format(f),
        ha='left', va='top', color=p[0].get_color(),
        )

ax.grid()
ax.set_xlim((time_percent[0], time_percent[-1]))
ax.set_xlabel('Time percent [%]')
ax.set_ylabel('L_b_corr [dB]')