import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from pycraf import atm
from pycraf import conversions as cnv
# >>>
# define height grid
height_grid = np.arange(0, 85, 0.1) * u.km
# >>>
# query profile_standard function
hprof = atm.profile_standard(height_grid)
# >>>
# Plot various quantities
_heights = height_grid.to(u.km).value
# >>>
plt.close()
fig, axes = plt.subplots(2, 3, figsize=(12, 10))
axes[0, 0].plot(hprof.temperature.to(u.K).value, _heights, 'k-')  # doctest: +SKIP
axes[0, 0].set_xlabel('Temperature [K]')  # doctest: +SKIP
axes[0, 0].set_xlim((160, 300))  # doctest: +SKIP
axes[0, 1].plot(hprof.pressure.to(u.hPa).value, _heights, 'b-', label='Total')  # doctest: +SKIP
axes[0, 1].plot(hprof.pressure_water.to(u.hPa).value, _heights, 'r-', label='Wet')  # doctest: +SKIP
axes[0, 1].legend(
    *axes[0, 1].get_legend_handles_labels(),
    loc='upper right', fontsize=10
    )  # doctest: +SKIP
axes[0, 1].set_xlabel('Pressure [hPa]')  # doctest: +SKIP
axes[0, 1].semilogx()  # doctest: +SKIP
axes[0, 1].set_xlim((1.e-6, 1100))  # doctest: +SKIP
axes[0, 2].plot(hprof.rho_water.to(u.g / u.cm ** 3).value, _heights, 'k-')  # doctest: +SKIP
axes[0, 2].set_xlabel('Water density [g / cm^3]')  # doctest: +SKIP
axes[0, 2].semilogx()  # doctest: +SKIP
axes[1, 0].plot(hprof.ref_index.to(cnv.dimless).value - 1., _heights, 'k-')  # doctest: +SKIP
axes[1, 0].set_xlabel('Refractive index - 1')  # doctest: +SKIP
axes[1, 0].semilogx()  # doctest: +SKIP
axes[1, 1].plot(hprof.humidity_water.to(u.percent).value, _heights, 'k-')  # doctest: +SKIP
axes[1, 1].set_xlabel('Relative humidity, water [%]')  # doctest: +SKIP
axes[1, 2].plot(hprof.humidity_ice.to(u.percent).value, _heights, 'k-')  # doctest: +SKIP
axes[1, 2].set_xlabel('Relative humidity, ice [%]')  # doctest: +SKIP
for idx, ax in enumerate(axes.flat):
    ax.set_ylim((0, 86))  # doctest: +SKIP
    if idx % 3 == 0:
        ax.set_ylabel('Height [km]')  # doctest: +SKIP
    ax.grid()  # doctest: +SKIP
# ...
fig.suptitle(
    'Atmospheric standard profile after ITU R-P.835-5, Annex 1',
    fontsize=16
    )  # doctest: +SKIP
plt.show()  # doctest: +SKIP
