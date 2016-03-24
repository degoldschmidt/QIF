import numpy as np               ## package for scientific computing with Python
import matplotlib.pyplot as plt  ## package for plotting

input = np.loadtxt("./data/rate/input.dat")
params = np.loadtxt("./data/rate/net.cfg")
N = params[0]
T = params[1]
dt = params[2]
time = np.arange(0,T+dt,dt) 

## Plotting raster plot and population activity
print("Plotting input...")
plt.title('Firing rates')
plt.xlim([-5,T])
#plt.ylim([0,0.01])
plt.ylabel('Firing rate [Hz]')
plt.xlabel('Time (msec)') 
plt.plot(input[0,:], 'k-')
plt.savefig("./figs/input.pdf", dpi=900)