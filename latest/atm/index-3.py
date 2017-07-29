import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from pycraf import atm
from pycraf import conversions as cnv

elevation = 5 * u.deg
obs_alt = 300 * u.m
_freqs = np.arange(0.1, 100, 1)
freq_grid = _freqs * u.GHz

cases = [
    # elevation, profile, label, linestyle
    (90 * u.deg, atm.profile_highlat_winter, 'Winter, Elevation: 90 deg', 'b-'),
    (90 * u.deg, atm.profile_highlat_summer, 'Summer, Elevation: 90 deg', 'r-'),
    (15 * u.deg, atm.profile_highlat_winter, 'Winter, Elevation: 15 deg', 'b--'),
    (15 * u.deg, atm.profile_highlat_summer, 'Summer, Elevation: 15 deg', 'r--'),
    ]

fig = plt.figure(figsize=(15, 16))
axes = [fig.add_subplot(2, 2, i) for i in range(1, 5)]
for elevation, profile, label,linestyle in cases:

    total_atten, refraction, tebb = atm.atten_slant_annex1(
        freq_grid, elevation, obs_alt, profile, t_bg=2.73 * u.K
        )
    opacity = atm.opacity_from_atten(total_atten, elevation)

    print(
        'Refraction for {}: {:.1f} arcsec'.format(
            label, refraction * 3600
        ))

    axes[0].plot(
        _freqs, total_atten.to(cnv.dB).value,
        linestyle, label=label
        )
    axes[1].plot(
        _freqs, (-total_atten).to(cnv.dimless).value,
        linestyle, label=label
        )
    axes[2].plot(
        _freqs, opacity.to(cnv.dimless).value,
        linestyle, label=label
        )
    axes[3].plot(
        _freqs, tebb.to(u.K).value,
        linestyle, label=label
        )

axes[0].semilogy()
axes[2].semilogy()
axes[0].legend(
    *axes[0].get_legend_handles_labels(),
    loc='upper left', fontsize=8
    )
axes[0].set_ylabel('Total attenuation [dB]')
axes[1].set_ylabel('Total gain')
axes[2].set_ylabel('Opacity')
axes[3].set_ylabel('Tebb [K]')
axes[0].set_ylim((2e-2, 9e2))
axes[1].set_ylim((0, 1))
axes[2].set_ylim((3e-3, 9e1))
axes[3].set_ylim((0, 310))

for idx, ax in enumerate(axes):
    ax.grid()
    ax.set_xlim((1, 99))
    if idx >= 2:
        ax.set_xlabel('Frequency [GHz]')