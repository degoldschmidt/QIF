from __future__ import division  ## true division in Python 2 (NOT NEEDED FOR PYTHON3)
import numpy as np               ## package for scientific computing with Python
from scipy import sparse as sp   ## package for sparse matrices
from scipy import stats as st    ## package for stats
import matplotlib.pyplot as plt  ## package for plotting


## setup parameters and state variables
N    = 1000                    # number of neurons
T    = 1000                    # total time to simulate (msec)
dt   = 0.5                     # simulation time step (msec)
time = np.arange(0, T+dt, dt)  # time array

## QIF/LIF properties
Vrest    = -65                     # resting potential [mV]
Vth      = -50                     # spike threshold [mV]
deltaV   = Vth - Vrest             # difference between threshold and resting
Vm       = np.ones([N,len(time)])  # membrane potential [mV] trace over time
Vm *= Vrest                        # initial condition of membrane potential
tau_m    = 10                      # time constant [msec]
tau_ref  = 4                       # refractory period [msec]
tau_psc  = 5                       # post synaptic current filter time constant [msec]
tau_esc  = 19                      # escape rate time constant [msec]
beta_esc = 1/4                     # escape rate function slope
Rm       = 1.                      # membrane resistance

## Input currents
print("Generate input...")
I    = np.zeros((N,len(time)))  # net input
Iext = np.ones(N)               # externally applied stimulus
Iconst = 0.                     # constant external input current
Iext = Iconst * Iext            # constant inputs to all neurons

# Rates
popAct    = np.zeros(len(time))  # population activity (instantaneous rate)
psthdt    = 400                  # PSTH time duration [msec]

## Synapse weight matrix
print("Generate weight matrix...")
g        = 1.                                               # recurrent gain parameter
mu_w     = 0                                                # zero mean
sigma_w  = g*(1/N)                                          # variance 1/N for balance
w_conn = 0.01                                               # connectivity in the weight matrix
rands = st.norm(loc=mu_w,scale=sigma_w).rvs                 # samples from a Gaussian random distr.
synapses = sp.random(N, N, density=w_conn, data_rvs=rands)  # generates sparse matrix using the Gaussian samples
#print synapses.nnz                                         # prints number of nonzero elements in synapses
#np.random.normal(mu_w, sigma_w, (N,N))                     # Gaussian distributed weights (full network)

## LIF neurons
def f_LIF(i):
    return -(Vm[active,i-1] - Vrest)

## QIF neurons
def f_QIF(i):
    return (Vm[active,i-1] - Vrest)*(Vm[active,i-1] - Vth)/deltaV

## Synaptic current model (exponential kernel)
def Isyn(t):
    ## t is an array of times since each neuron's last spike event
    t[np.nonzero(t < 0)] = 0
    return t*np.exp(-t/tau_psc)
last_spike = np.zeros(N) - tau_ref       # initialize last spike time vector
samples = np.zeros(N)                    # array of N samples

## Escape rate function
def esc_rate(i):
    return 1-np.exp( -dt*(1/tau_esc)*np.exp(beta_esc*(Vm[:,i]-Vth))  )

## Simulate network (i = time steps; t = simulation time)
print("Simulate...")
raster = np.zeros([N,len(time)])*np.nan
for i, t in enumerate(time):

    # array of active units (units not in refractory period)
    active = np.nonzero(t > last_spike + tau_ref)

    # Euler integration of membrane potential
    Vm[active,i] = Vm[active,i-1] + dt * ( f_QIF(i) + Rm * I[active, i-1]) / tau_m

    # array of all spikes at time step i
    #spiked = np.nonzero(Vm[:,i] > Vth)                                         # deterministic spike generation (not used)
    samples = np.random.binomial(1, esc_rate(i))                                # stochastic spike generation based on Bernoulli distribution (binomial with n=1)
    spiked = np.nonzero(samples)                                                # indices of neurons with spike events

    # array spike times of last spike for refractory period and synaptic current
    last_spike[spiked] = t

    # log spikes in raster (plus 1 because neuron index starts with 1)
    raster[spiked,i] = spiked[0]+1

    # net current equals external input + dot product of synaptic weights and synaptic currents
    I[:,i] = Iext + synapses.dot(Isyn(t - last_spike))                          # add "np.sin(0.01*i)*" for oscillating input

## calculate mean escape rate

print("Print number of spikes per neuron per second")
countmat = np.sign(np.nan_to_num(raster))
print(1000*np.sum(countmat)/(T*N))

## Plotting raster plot and population activity
#print "Plot raster plot..."
#outfile = "spikes.dat"
#np.savetxt(outfile, raster, fmt='%.f', delimiter=' ', newline='\n', header='', footer='', comments='# ')
#plt.title('Spike Raster Plot')
#py.xlim([0,T])
#py.ylim([0.75,N+0.25])
plt.plot(time, np.transpose(raster), 'k.')
##py.plot(time, 1000*esc_rates[0,:], 'b-')

## Plotting membrane potential
#print "Plot membrane potential..."
#py.title('Membrane Potential')
#py.pcolor(Vm)
#py.colorbar()

#py.ylabel('Rate [Hz]')
#py.ylabel('Neuron index')
#py.xlabel('Time (msec)')
plt.show()
