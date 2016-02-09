import numpy as np               ## package for scientific computing with Python
import matplotlib.pyplot as plt  ## package for plotting

rates = np.loadtxt("./data/rate/rates.dat")
params = np.loadtxt("./data/rate/net.cfg")
N = params[0]
T = params[1]
dt = params[2]
time = np.arange(0,T+dt,dt) 

## Plotting raster plot and population activity
print("Plotting rates...")
plt.title('Firing rates')
plt.xlim([0,T])
plt.ylabel('Firing rate [Hz]')
plt.xlabel('Time (msec)')
plt.plot(rates[0,:], 'k-', rates[1,:], 'r-', rates[2,:], 'b-', rates[2,:], 'g-', rates[3,:], 'c-', rates[4,:], 'y-')
plt.savefig("./figs/rates.pdf", dpi=900)