# first, we need to create the atmospheric layer cache
atm_layers_cache = atm.atm_layers([1] * u.GHz, atm.profile_highlat_winter)
# >>>
plt.close()
fig = plt.figure(figsize=(12, 6))
# >>>
# to plot the atmospheric layers, we need to access the layers_cache:
a_e = atm.EARTH_RADIUS
layer_angles = np.arange(0, 0.1, 1e-3)
layer_radii = atm_layers_cache['radii']
bottom, top = layer_radii[[0, 900]]
plt.plot(bottom * np.sin(layer_angles), bottom * np.cos(layer_angles), 'k-')  # doctest: +IGNORE_OUTPUT
plt.plot(top * np.sin(layer_angles), top * np.cos(layer_angles), 'k-')  # doctest: +IGNORE_OUTPUT
# we only plot some layers
for r in layer_radii[[200, 500, 600, 700, 800, 850]]:
    plt.plot(r * np.sin(layer_angles), r * np.cos(layer_angles), 'k--', alpha=0.5)  # doctest: +IGNORE_OUTPUT
# ...
# now create four different example paths (of different type)
for path_num, elevation, obs_alt, max_path_length in zip(
        [1, 2, 3, 4],
        [10, 20, -5, -45] * u.deg,
        [300, 300, 25000, 50000] * u.m,
        [1000, 230, 300, 1000] * u.km,
        ):
# ...
    path_params, _, refraction = atm.raytrace_path(
        elevation, obs_alt, atm_layers_cache,
        max_path_length=max_path_length,
        )
# ...
    print('total path length {:d}: {:5.1f}'.format(
        path_num, np.sum(path_params.a_n)
        ))
# ...
    radii = path_params.r_n
    angle = path_params.delta_n
    x, y = radii * np.sin(angle), radii * np.cos(angle)
    _ = plt.plot(x, y, '-', label='Path {:d}'.format(path_num))
# total path length 1: 1000.0
# total path length 2: 230.0
# total path length 3: 300.0
# total path length 4:  71.0
plt.legend(*plt.gca().get_legend_handles_labels())  # doctest: +IGNORE_OUTPUT
plt.xlim((0, 290))  # doctest: +IGNORE_OUTPUT
plt.ylim((a_e - 5, 6453))  # doctest: +IGNORE_OUTPUT
plt.title('Path propagation through layered atmosphere')  # doctest: +IGNORE_OUTPUT
plt.xlabel('Projected distance (km)')  # doctest: +IGNORE_OUTPUT
plt.ylabel('Distance to Earth center (km)')  # doctest: +IGNORE_OUTPUT
plt.gca().set_aspect('equal')  # doctest: +IGNORE_OUTPUT
