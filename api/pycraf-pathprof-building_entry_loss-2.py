import matplotlib.pyplot as plt
from pycraf import conversions as cnv
from pycraf import pathprof
from astropy import units as u

freqs = [0.1, 1, 10, 100] * u.GHz
colors = ['k', 'r', 'g', 'b']
theta = 0 * u.deg
prob = np.linspace(1e-6, 1 - 1e-6, 200) * cnv.dimless

plt.figure(figsize=(8, 6))

for freq, color in zip(freqs, colors):

    for btype, ls in zip([
            pathprof.BuildingType.TRADITIONAL,
            pathprof.BuildingType.THERM_EFF
            ], ['--', ':']):

        L_bel = pathprof.building_entry_loss(freq, theta, prob, btype)

        plt.plot(L_bel, prob, ls, color=color)

    # labels
    plt.plot([], [], '-', color=color, label=str(freq))

plt.xlabel('BEL [dB]')
plt.ylabel('Probability')
plt.xlim((-20, 140))
plt.ylim((0, 1))
plt.legend(*plt.gca().get_legend_handles_labels())
plt.title('BEL (dashed: Traditional, dotted: Thermally inefficient')
plt.grid()