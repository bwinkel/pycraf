import numpy as np
import matplotlib.pyplot as plt
import pycraf.conversions as cnv
from pycraf.antenna import *
from astropy import units as u

azims = np.arange(-180, 180.1, 0.5) * u.deg
elevs = np.arange(-90, 90.1, 0.5) * u.deg

# BS (outdoor) according to IMT.PARAMETER table 10 (multipage!)
G_Emax = 5 * cnv.dB
A_m, SLA_nu = 30. * cnv.dB, 30. * cnv.dB
azim_3db, elev_3db = 65. * u.deg, 65. * u.deg

gains_single = imt2020_single_element_pattern(
    azims[np.newaxis], elevs[:, np.newaxis],
    G_Emax,
    A_m, SLA_nu,
    azim_3db, elev_3db
    ).to(cnv.dB).value

fig = plt.figure(figsize=(12, 6))
ax = fig.add_axes((0.15, 0.1, 0.7, 0.7))
cax = fig.add_axes((0.85, 0.1, 0.02, 0.7))

im = ax.imshow(
    gains_single,
    extent=(azims[0].value, azims[-1].value, elevs[0].value, elevs[-1].value),
    vmin=-25, vmax=np.max(np.ceil(gains_single)),
    origin='lower', cmap='viridis',
    )
plt.colorbar(im, cax=cax)
cax.set_ylabel('Gain [dB]')
ax.set_xlabel('Azimuth [deg]')
ax.set_ylabel('Elevation [deg]')
ax.set_title('Single element pattern')
plt.show()

d_H, d_V = 0.5 * cnv.dimless, 0.5 * cnv.dimless
N_H, N_V = 8, 8

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.8, 0.1, 0.02, 0.8])

for i, azim_i in enumerate(u.Quantity([0, 30], u.deg)):
    for j, elev_j in enumerate(u.Quantity([0, -15], u.deg)):

        ax = axes[i, j]
        gains_array = imt2020_composite_pattern(
            azims[np.newaxis], elevs[:, np.newaxis],
            azim_i, elev_j,
            G_Emax,
            A_m, SLA_nu,
            azim_3db, elev_3db,
            d_H, d_V,
            N_H, N_V,
            ).to(cnv.dB).value

        gains_array[gains_array < -100] = -100  # fix blanks (-infty)
        im = ax.imshow(
            gains_array, extent=(
                azims[0].value, azims[-1].value,
                elevs[0].value, elevs[-1].value
                ),
            vmin=-26, vmax=26,
            origin='lower', cmap='viridis',
            )
        plt.colorbar(im, cax=cax)
        cax.set_ylabel('Gain [dB]')
        ax.set_xlabel('Azimuth [deg]')
        ax.set_ylabel('Elevation [deg]')
        ax.set_xlim((-90, 90))
        ax.set_ylim((-90, 90))
        ax.set_title(
            'Escan: {:.1f}d, Tilt: {:.1f}d'.format(
                azim_i.value, -elev_j.value
            ))
plt.show()