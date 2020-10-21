import numpy as np
import matplotlib.pyplot as plt
import pycraf.conversions as cnv
from pycraf.antenna import *
from astropy import units as u

azims = np.arange(-180, 180.1, 0.5) * u.deg
elevs = np.arange(-90, 90.1, 0.5) * u.deg

G0 = 18. * cnv.dB
phi_3db = 65. * u.deg
# theta_3db can be inferred in the following way:
theta_3db = 31000 / G0.to(cnv.dimless) / phi_3db.value * u.deg
k_p, k_h, k_v = (0.7, 0.7, 0.3) * cnv.dimless
tilt_m, tilt_e = (0, 0) * u.deg

bs_gains = imt_advanced_sectoral_peak_sidelobe_pattern_400_to_6000_mhz(
    azims[np.newaxis], elevs[:, np.newaxis],
    G0, phi_3db, theta_3db,
    k_p, k_h, k_v,
    tilt_m=tilt_m, tilt_e=tilt_e,
    ).to(cnv.dB).value

fig = plt.figure(figsize=(12, 6))
ax = fig.add_axes((0.15, 0.1, 0.7, 0.7))
cax = fig.add_axes((0.85, 0.1, 0.02, 0.7))

im = ax.imshow(
    bs_gains,
    extent=(azims[0].value, azims[-1].value, elevs[0].value, elevs[-1].value),
    origin='lower', cmap='viridis',
    )
plt.colorbar(im, cax=cax)
cax.set_ylabel('Gain [dB]')
ax.set_xlabel('Azimuth [deg]')
ax.set_ylabel('Elevation [deg]')
plt.show()