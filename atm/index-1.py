import numpy as np
import matplotlib.pyplot as plt
from pycraf import atm
from pycraf import conversions as cnv
from astropy import units as u
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
axes[0, 0].plot(hprof.temperature.to(u.K).value, _heights, 'k-')  # doctest: +IGNORE_OUTPUT
axes[0, 0].set_xlabel('Temperature [K]')  # doctest: +IGNORE_OUTPUT
axes[0, 0].set_xlim((160, 300))  # doctest: +IGNORE_OUTPUT
axes[0, 1].plot(hprof.pressure.to(u.hPa).value, _heights, 'b-', label='Total')  # doctest: +IGNORE_OUTPUT
axes[0, 1].plot(hprof.pressure_water.to(u.hPa).value, _heights, 'r-', label='Wet')  # doctest: +IGNORE_OUTPUT
axes[0, 1].legend(
    *axes[0, 1].get_legend_handles_labels(),
    loc='upper right', fontsize=10
    )  # doctest: +IGNORE_OUTPUT
axes[0, 1].set_xlabel('Pressure [hPa]')  # doctest: +IGNORE_OUTPUT
axes[0, 1].semilogx()  # doctest: +IGNORE_OUTPUT
axes[0, 1].set_xlim((1.e-6, 1100))  # doctest: +IGNORE_OUTPUT
axes[0, 2].plot(hprof.rho_water.to(u.g / u.cm ** 3).value, _heights, 'k-')  # doctest: +IGNORE_OUTPUT
axes[0, 2].set_xlabel('Water density [g / cm^3]')  # doctest: +IGNORE_OUTPUT
axes[0, 2].semilogx()  # doctest: +IGNORE_OUTPUT
axes[1, 0].plot(hprof.ref_index.to(cnv.dimless).value - 1., _heights, 'k-')  # doctest: +IGNORE_OUTPUT
axes[1, 0].set_xlabel('Refractive index - 1')  # doctest: +IGNORE_OUTPUT
axes[1, 0].semilogx()  # doctest: +IGNORE_OUTPUT
axes[1, 1].plot(hprof.humidity_water.to(u.percent).value, _heights, 'k-')  # doctest: +IGNORE_OUTPUT
axes[1, 1].set_xlabel('Relative humidity, water [%]')  # doctest: +IGNORE_OUTPUT
axes[1, 2].plot(hprof.humidity_ice.to(u.percent).value, _heights, 'k-')  # doctest: +IGNORE_OUTPUT
axes[1, 2].set_xlabel('Relative humidity, ice [%]')  # doctest: +IGNORE_OUTPUT
for idx, ax in enumerate(axes.flat):
    ax.set_ylim((0, 86))  # doctest: +IGNORE_OUTPUT
    if idx % 3 == 0:
        ax.set_ylabel('Height [km]')  # doctest: +IGNORE_OUTPUT
    ax.grid()  # doctest: +IGNORE_OUTPUT
# ...
fig.suptitle(
    'Atmospheric standard profile after ITU R-P.835-5, Annex 1',
    fontsize=16
    )  # doctest: +IGNORE_OUTPUT
