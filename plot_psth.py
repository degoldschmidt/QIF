from __future__ import division  ## true division in Python 2 (NOT NEEDED FOR PYTHON3)
import numpy as np               ## package for scientific computing with Python
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

spikes = np.loadtxt("./data/spiking/spikes.dat")
params = np.loadtxt("./data/spiking/net.cfg")
N      = params[0]
T      = params[1] 
dt     = params[2]

# the histogram of the data
binsize=10
hist, bins = np.histogram(spikes[:,0], bins=T/binsize)
hist = hist/binsize

plt.bar(bins[:-1], hist, width = binsize)
plt.xlim(min(bins), max(bins))

plt.xlabel('Time [ms]')
plt.ylabel('Firing rate estimate [Hz]')
plt.title(r'PSTH')
plt.axis([0, T, 0, 1.2*max(hist)])
plt.savefig("./figs/psth.pdf", dpi=900)