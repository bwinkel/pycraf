import matplotlib.pyplot as plt
from pycraf import pathprof
from astropy import units as u

pathprof.SrtmConf.set(download='missing')

lon_t, lat_t = 9.943 * u.deg, 54.773 * u.deg  # Northern Germany
map_size_lon, map_size_lat = 1.5 * u.deg, 1.5 * u.deg
map_resolution = 3. * u.arcsec

lons, lats, heightmap = pathprof.srtm_height_map(
    lon_t, lat_t,
    map_size_lon, map_size_lat,
    map_resolution=map_resolution,
    )

_lons = lons.to(u.deg).value
_lats = lats.to(u.deg).value
_heightmap = heightmap.to(u.m).value

vmin, vmax = -20, 170
terrain_cmap, terrain_norm = pathprof.terrain_cmap_factory(
    sealevel=0.5, vmax=vmax
    )
_heightmap[_heightmap < 0] = 0.51  # fix for coastal region

fig = plt.figure(figsize=(10, 10))
ax = fig.add_axes((0., 0., 1.0, 1.0))
cbax = fig.add_axes((0., 0., 1.0, .02))
cim = ax.imshow(
    _heightmap,
    origin='lower', interpolation='nearest',
    cmap=terrain_cmap, norm=terrain_norm,
    vmin=vmin, vmax=vmax,
    extent=(_lons[0], _lons[-1], _lats[0], _lats[-1]),
    )
cbar = fig.colorbar(
    cim, cax=cbax, orientation='horizontal'
    )
ax.set_aspect(abs(_lons[-1] - _lons[0]) / abs(_lats[-1] - _lats[0]))
cbar.set_label(r'Height (amsl)', color='k')
cbax.xaxis.set_label_position('top')
for t in cbax.xaxis.get_major_ticks():
    t.tick1line.set_visible(False)
    t.tick2line.set_visible(True)
    t.label1.set_visible(False)
    t.label2.set_visible(True)
ctics = np.arange(0, 1150, 50)
cbar.set_ticks(ctics)
cbar.ax.set_xticklabels(map('{:.0f} m'.format, ctics), color='k')
ax.set_xlabel('Longitude [deg]')
ax.set_ylabel('Latitude [deg]')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')