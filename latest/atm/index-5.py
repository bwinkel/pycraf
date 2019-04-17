elevations = np.arange(0.5, 90, 1)
obs_alt = 100 * u.m
refractions = np.array([
    atm.path_endpoint(
        elev * u.deg, obs_alt, atm_layers_cache,
        ).refraction.to(u.arcsec).value
    for elev in elevations
    ])
# ...
plt.close()  # doctest: +IGNORE_OUTPUT
fig = plt.figure(figsize=(8, 4))  # doctest: +IGNORE_OUTPUT
plt.plot(elevations, refractions, '-')  # doctest: +IGNORE_OUTPUT
plt.xlabel('Elevation (deg)')  # doctest: +IGNORE_OUTPUT
plt.ylabel('Refraction (arcsec)')  # doctest: +IGNORE_OUTPUT
plt.grid()  # doctest: +IGNORE_OUTPUT
