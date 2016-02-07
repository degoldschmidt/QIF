import numpy as np               ## package for scientific computing with Python
import matplotlib.pyplot as plt  ## package for plotting

spikes = np.loadtxt("./data/spiking/spikes.dat")
params = np.loadtxt("./data/spiking/net.cfg")
N = params[0]
T = params[1] 

## Plotting raster plot and population activity
print("Plot raster plot...")
plt.title('Spike Raster Plot')
plt.xlim([0,T])
plt.ylim([0.75,N+0.25])
plt.ylabel('Neuron index')
plt.xlabel('Time (msec)')
plt.plot(spikes[:,0], spikes[:,1], 'k.', markersize=1)
plt.savefig("./figs/spikes.pdf", dpi=900)