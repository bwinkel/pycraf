import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from pycraf import atm
from pycraf import conversions as cnv

_freqs = np.arange(1, 1000, 1)
freq_grid = _freqs * u.GHz
total_pressure = 1013 * u.hPa
temperature = 290 * u.K
rho_water = 7.5 * u.g / u.m ** 3

pressure_water = atm.pressure_water_from_rho_water(temperature, rho_water)
pressure_dry = total_pressure - pressure_water

print(
    'Oxygen pressure: {0.value:.2f} {0.unit}, '
    'Water vapor partial pressure: {1.value:.2f} {1.unit}'.format(
        pressure_dry, pressure_water
    ))

atten_dry, atten_wet = atm.atten_specific_annex1(
    freq_grid, pressure_dry, pressure_water, temperature
    )

plt.figure(figsize=(15, 9))
plt.plot(
    _freqs, atten_dry.to(cnv.dB / u.km).value,
    'r-', label='Dry air'
    )
plt.plot(
    _freqs, atten_wet.to(cnv.dB / u.km).value,
    'b-', label='Wet air'
    )
plt.plot(
    _freqs, (atten_dry + atten_wet).to(cnv.dB / u.km).value,
    'k-', label='Total'
    )
plt.semilogy()
plt.xlabel('Frequency [GHz]')
plt.ylabel('Specific Attenuation [dB / km]')
plt.xlim((1, 999))
plt.ylim((5.e-3, 0.9e5))
plt.grid()
plt.legend(*plt.gca().get_legend_handles_labels(), loc='upper left')
plt.title(
    'Specific attenuation for standard conditions, '
    'according to ITU-R P.676 (10), annex 1',
    fontsize=16
    )