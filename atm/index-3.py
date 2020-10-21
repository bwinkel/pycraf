obs_alt = 300 * u.m
_freqs = np.arange(0.25, 100, 0.5)
freq_grid = _freqs * u.GHz
# >>>
cases = [
    # elevation, profile, label, linestyle
    (90 * u.deg, atm.profile_highlat_winter, 'Winter, Elevation: 90 deg', 'b-'),
    (90 * u.deg, atm.profile_highlat_summer, 'Summer, Elevation: 90 deg', 'r-'),
    (15 * u.deg, atm.profile_highlat_winter, 'Winter, Elevation: 15 deg', 'b--'),
    (15 * u.deg, atm.profile_highlat_summer, 'Summer, Elevation: 15 deg', 'r--'),
    ]
# ...
plt.close()
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for elev, profile, label, linestyle in cases:
# ...
    atm_layers_cache = atm.atm_layers(freq_grid, profile)
    total_atten, refraction, tebb = atm.atten_slant_annex1(
        elev, obs_alt, atm_layers_cache, t_bg=2.73 * u.K
        )
    opacity = atm.opacity_from_atten(total_atten, elev)
# ...
    print('Refraction for {}: {:.1f}'.format(label, refraction.to(u.arcsec)))
# ...
    _ = axes[0, 0].plot(_freqs, total_atten.to(cnv.dB).value, linestyle, label=label)
    _ = axes[0, 1].plot(_freqs, 1 / total_atten.to(cnv.dimless).value, linestyle, label=label)
    _ = axes[1, 0].plot(_freqs, opacity.to(cnv.dimless).value, linestyle, label=label)
    _ = axes[1, 1].plot(_freqs, tebb.to(u.K).value, linestyle, label=label)
# Refraction for Winter, Elevation: 90 deg: -0.0 arcsec
# Refraction for Summer, Elevation: 90 deg: -0.0 arcsec
# Refraction for Winter, Elevation: 15 deg: -228.6 arcsec
# Refraction for Summer, Elevation: 15 deg: -237.9 arcsec
axes[0, 0].semilogy()  # doctest: +IGNORE_OUTPUT
axes[1, 0].semilogy()  # doctest: +IGNORE_OUTPUT
axes[0, 0].legend(*axes[0, 0].get_legend_handles_labels(), loc='upper left', fontsize=8)  # doctest: +IGNORE_OUTPUT
axes[0, 0].set_ylabel('Total attenuation [dB]')  # doctest: +IGNORE_OUTPUT
axes[0, 1].set_ylabel('Total gain')  # doctest: +IGNORE_OUTPUT
axes[1, 0].set_ylabel('Zenith opacity')  # doctest: +IGNORE_OUTPUT
axes[1, 1].set_ylabel('Tebb [K]')  # doctest: +IGNORE_OUTPUT
axes[0, 0].set_ylim((2e-2, 9e2))  # doctest: +IGNORE_OUTPUT
axes[0, 1].set_ylim((0, 1))  # doctest: +IGNORE_OUTPUT
axes[1, 0].set_ylim((3e-3, 9e1))  # doctest: +IGNORE_OUTPUT
axes[1, 1].set_ylim((0, 310))  # doctest: +IGNORE_OUTPUT
# >>>
for idx, ax in enumerate(axes.flat):
    ax.grid()  # doctest: +IGNORE_OUTPUT
    ax.set_xlim((1, 99))  # doctest: +IGNORE_OUTPUT
    if idx >= 2:
        ax.set_xlabel('Frequency [GHz]')  # doctest: +IGNORE_OUTPUT
# ...
