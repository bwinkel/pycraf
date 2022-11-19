import matplotlib.pyplot as plt
from pycraf import conversions as cnv
from pycraf import pathprof
from astropy import units as u

freq = np.logspace(-1, 2, 100) * u.GHz
theta = 0 * u.deg
prob = 0.5 * cnv.dimless

plt.figure(figsize=(8, 6))

for btype in [
        pathprof.BuildingType.TRADITIONAL,
        pathprof.BuildingType.THERM_EFF
        ]:

    L_bel = pathprof.building_entry_loss(freq, theta, prob, btype)

    plt.semilogx(freq, L_bel, '-', label=str(btype))

plt.xlabel('Frequency [GHz]')
plt.ylabel('BEL [dB]')
plt.xlim((0.1, 100))
plt.ylim((10, 60))
plt.legend(*plt.gca().get_legend_handles_labels())
plt.title('Median BEL at horizontal incidence')
plt.grid()