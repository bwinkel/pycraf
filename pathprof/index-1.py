import matplotlib.pyplot as plt
from pycraf import pathprof
from astropy import units as u

# allow download of missing SRTM data:
pathprof.SrtmConf.set(download='missing')

lon_t, lat_t = 6.8836 * u.deg, 50.525 * u.deg
lon_r, lat_r = 7.3334 * u.deg, 50.635 * u.deg
hprof_step = 100 * u.m

(
    lons, lats, distance,
    distances, heights,
    bearing, back_bearing, back_bearings
    ) = pathprof.srtm_height_profile(
        lon_t, lat_t, lon_r, lat_r, hprof_step
        )

_distances = distances.to(u.km).value
_heights = heights.to(u.m).value

plt.figure(figsize=(10, 5))
plt.plot(_distances, _heights, 'k-')
plt.xlabel('Distance [km]')
plt.ylabel('Height [m]')
plt.grid()