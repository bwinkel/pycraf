import numpy as np
import matplotlib.pyplot as plt
from pycraf import pathprof, geometry
from astropy import units as u

lon, lat = 0 * u.deg, 70 * u.deg
bearings = np.linspace(-180, 180, 721) * u.deg
distance = 1000 * u.km

lons, lats, _ = pathprof.geoid_direct(lon, lat, bearings, distance)
ang_dist = geometry.true_angular_distance(lon, lat, lons, lats)

fig = plt.figure(figsize=(7, 6))
ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
cax = fig.add_axes((0.9, 0.1, 0.02, 0.8))
sc = ax.scatter(
    lons.to(u.deg), lats.to(u.deg),
    c=ang_dist.to(u.deg), cmap='viridis'
    )
cbar = plt.colorbar(sc, cax=cax)
cbar.set_label('Angular distance [deg]')
ax.set_xlabel('Longitude [deg]')
ax.set_ylabel('Latitude [deg]')
ax.set_aspect(1. / np.cos(np.radians(70)))
ax.grid()