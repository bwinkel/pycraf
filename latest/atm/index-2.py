_freqs = np.arange(1, 1000, 1)
freq_grid = _freqs * u.GHz
# weather parameter at ground-level
total_pressure = 1013 * u.hPa
temperature = 290 * u.K
# water content is from the integral over the full height
rho_water = 7.5 * u.g / u.m ** 3
# >>>
# alternatively, one can use atm.pressure_water_from_humidity
pressure_water = atm.pressure_water_from_rho_water(temperature, rho_water)
pressure_dry = total_pressure - pressure_water
# >>>
print(
    'Oxygen pressure: {0.value:.2f} {0.unit}, '
    'Water vapor partial pressure: {1.value:.2f} {1.unit}'.format(
        pressure_dry, pressure_water
    ))
# Oxygen pressure: 1002.96 hPa, Water vapor partial pressure: 10.04 hPa
atten_dry, atten_wet = atm.atten_specific_annex1(
    freq_grid, pressure_dry, pressure_water, temperature
    )
# ...
plt.close()
plt.figure(figsize=(12, 7))  # doctest: +IGNORE_OUTPUT
plt.plot(
    _freqs, atten_dry.to(cnv.dB / u.km).value,
    'r-', label='Dry air'
    )  # doctest: +IGNORE_OUTPUT
plt.plot(
    _freqs, atten_wet.to(cnv.dB / u.km).value,
    'b-', label='Wet air'
    )  # doctest: +IGNORE_OUTPUT
plt.plot(
    _freqs, (atten_dry + atten_wet).to(cnv.dB / u.km).value,
    'k-', label='Total'
    )  # doctest: +IGNORE_OUTPUT
plt.semilogy()  # doctest: +IGNORE_OUTPUT
plt.xlabel('Frequency [GHz]')  # doctest: +IGNORE_OUTPUT
plt.ylabel('Specific Attenuation [dB / km]')  # doctest: +IGNORE_OUTPUT
plt.xlim((1, 999))  # doctest: +IGNORE_OUTPUT
plt.ylim((5.e-3, 0.9e5))  # doctest: +IGNORE_OUTPUT
plt.grid()  # doctest: +IGNORE_OUTPUT
plt.legend(*plt.gca().get_legend_handles_labels(), loc='upper left')  # doctest: +IGNORE_OUTPUT
plt.title(
    'Specific attenuation for standard conditions, '
    'according to ITU-R P.676 (10), annex 1',
    fontsize=16
    )  # doctest: +IGNORE_OUTPUT
