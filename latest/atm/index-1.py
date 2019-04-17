import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from pycraf import atm
from pycraf import conversions as cnv

# define height grid
height_grid = np.arange(0, 85, 0.1) * u.km

# query profile_standard function
(
    temperatures,
    pressures,
    rho_water,
    pressures_water,
    ref_indices,
    humidities_water,
    humidities_ice
    ) = atm.profile_standard(height_grid)

# Plot various quantities
_heights = height_grid.to(u.km).value

fig = plt.figure(figsize=(15, 13))
axes = [fig.add_subplot(2, 3, i) for i in range(1, 7)]
axes[0].plot(temperatures.to(u.K).value, _heights, 'k-')
axes[0].set_xlabel('Temperature [K]')
axes[0].set_xlim((160, 300))
axes[1].plot(pressures.to(u.hPa), _heights, 'b-', label='Total')
axes[1].plot(pressures_water.to(u.hPa), _heights, 'r-', label='Wet')
axes[1].legend(
    *axes[1].get_legend_handles_labels(),
    loc='upper right', fontsize=10
    )
axes[1].set_xlabel('Pressure [hPa]')
axes[1].semilogx()
axes[1].set_xlim((1.e-6, 1100))
axes[2].plot(rho_water.to(u.g / u.cm ** 3).value, _heights, 'k-')
axes[2].set_xlabel('Water density [g / cm^3]')
axes[2].semilogx()
#ax3.set_xlim((1.e-3, 1100))
axes[3].plot(ref_indices.to(cnv.dimless).value - 1., _heights, 'k-')
axes[3].set_xlabel('Refractive index - 1')
axes[3].semilogx()
#ax3.set_xlim((1.e-3, 1100))
axes[4].plot(humidities_water.to(u.percent).value, _heights, 'k-')
axes[4].set_xlabel('Relative humidity, water [%]')
axes[5].plot(humidities_ice.to(u.percent).value, _heights, 'k-')
axes[5].set_xlabel('Relative humidity, ice [%]')
for idx, ax in enumerate(axes):
    ax.set_ylim((0, 86))
    if idx % 3 == 0:
        ax.set_ylabel('Height [km]')
    ax.grid()

fig.suptitle(
    'Atmospheric standard profile after ITU R-P.835-5, Annex 1',
    fontsize=16
    )