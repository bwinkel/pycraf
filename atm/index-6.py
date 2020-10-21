obs_alt = 0.1001 * u.km
e_a = atm.EARTH_RADIUS
# >>>
layer_angles = np.linspace(-0.001, 0.02, 400)
radii = atm_layers_cache['radii']
# >>>
plt.close()
fig = plt.figure(figsize=(14, 6))
for r in radii:
    plt.plot(
        r * np.sin(layer_angles),
        r * np.cos(layer_angles) - e_a,
        'k--', alpha=0.5
        )  # doctest: +IGNORE_OUTPUT
# ...
for elev in np.linspace(-0.04, -0.01, 21):
    path_params, _, _ = atm.raytrace_path(
        elev * u.deg, obs_alt, atm_layers_cache,
        max_path_length=20 * u.km,
        )
    plt.plot(path_params.x_n, path_params.y_n - e_a, '-')  # doctest: +IGNORE_OUTPUT
# ...
plt.xlim((-0.1, 15.1))  # doctest: +IGNORE_OUTPUT
plt.ylim((obs_alt.value - 0.012, obs_alt.value + 0.001))  # doctest: +IGNORE_OUTPUT
plt.title('Path propagation through layered atmosphere')  # doctest: +IGNORE_OUTPUT
plt.xlabel('Projected distance (km)')  # doctest: +IGNORE_OUTPUT
plt.ylabel('Height above ground (km)')  # doctest: +IGNORE_OUTPUT
