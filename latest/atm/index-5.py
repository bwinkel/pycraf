elevations = np.arange(0.5, 90, 1)
obs_alt = 100 * u.m
refractions = np.array([
    atm.path_endpoint(
        elev * u.deg, obs_alt, atm_layers_cache,
        ).refraction.to(u.arcsec).value
    for elev in elevations
    ])
# ...
plt.close()  # doctest: +SKIP
fig = plt.figure(figsize=(8, 4))  # doctest: +SKIP
plt.plot(elevations, refractions, '-')  # doctest: +SKIP
plt.xlabel('Elevation (deg)')  # doctest: +SKIP
plt.ylabel('Refraction (arcsec)')  # doctest: +SKIP
plt.grid()  # doctest: +SKIP
plt.show()  # doctest: +SKIP
