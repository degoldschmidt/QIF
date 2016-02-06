from __future__ import division
import numpy as np
from scipy import sparse
from scipy import stats
import matplotlib.pyplot as plt
import sys
import time
from progressbar import AnimatedMarker, Percentage, RotatingMarker, Counter, ETA, Bar, ProgressBar


## setup parameters and state variables
N    = 1000                    # number of neurons
T    = 1000                    # total time to simulate (msec)
dt   = 0.5                     # simulation time step (msec)
time = np.arange(0, T+dt, dt)  # time array

## progress bar for long simulations
use_progbar = False
if use_progbar:
    widgets = ['Progress: ', Percentage(), ' ', Bar(marker=RotatingMarker()),
           ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=len(time)).start()

## QIF/LIF properties
Vrest    = -65                           # resting potential [mV]
Vth      = -50                           # spike threshold [mV]
deltaV   = Vth - Vrest                   # difference between threshold and resting
Vm       = Vrest*np.ones([N,len(time)])  # potential [V] trace over time
tau_m    = 10                            # time constant [msec]
tau_ref  = 4                             # refractory period [msec]
tau_psc  = 5                             # post synaptic current filter time constant [msec]
tau_esc  = 19                            # escape rate time constant [msec]
beta_esc = 1/4                           # escape rate function slope
Rm       = 1.                            # membrane resistance

## Input currents
I    = np.zeros((N,len(time)))           # net input
Iext = np.ones(N)                        # externally applied stimulus
Iconst = 0.                              # constant external input

# Rates
popAct    = np.zeros((N,len(time)))          # population activity (instantaneous)
psthdt    = 400                          # PSTH time duration [msec]

## constant input to all neurons
print "Generate input..."
Iext = Iconst * Iext                        # external input set to 0.001 [A]

## Synapse weight matrix
print "Generate weight matrix..."
g        = 1.                                       # recurrent gain parameter
mu_w     = 0                                        # zero mean
sigma_w  = g*(1/N)                                  # variance 1/N for balance
syn_connectivity = 0.01
randweights = stats.norm(loc=mu_w,scale=sigma_w).rvs
synapses = sparse.random(N, N, density=syn_connectivity, data_rvs=randweights)
#print synapses.nnz                                 # prints number of nonzero elements in synapses
#np.random.normal(mu_w, sigma_w, (N,N))   # Gaussian distributed weights

## LIF neurons
def f_LIF(i):
    return -(Vm[:,i-1] - Vrest)

## QIF neurons
def f_QIF(i):
    return (Vm[:,i-1] - Vrest)*(Vm[:,i-1] - Vth)/deltaV

## Synaptic current model (exponential kernel)
def Isyn(t):
    '''t is an array of times since each neuron's last spike event'''
    t[np.nonzero(t < 0)] = 0
    return t*np.exp(-t/tau_psc)
last_spike = np.zeros(N) - tau_ref       # initialize last spike time vector
samples = np.zeros(N)                    # array of N samples

## Escape rate function
def esc_rate(i):
    return (1/tau_esc)*np.exp(beta_esc*(Vm[:,i]-Vth))

## Simulate network (i = time steps; t = simulation time)
print "Simulate..."
raster = np.zeros([N,len(time)])*np.nan
for i, t in enumerate(time):
    # update progress bar
    if use_progbar:
        pbar.update(i)

    # array of active units (units not in refractory period)
    #active = np.nonzero(t > last_spike + tau_ref)

    # Euler integration of membrane potential
    Vm[:,i] = Vm[:,i-1] + dt * ( f_QIF(i) + Rm * I[:, i-1]) / tau_m

    # array of all spikes at time step i
    #spiked = np.nonzero(Vm[:,i] > Vth)                                         # deterministic spike generation (not used)
    #samples = np.random.binomial(1, esc_rate(i))                                # stochastic spike generation based on Bernoulli distribution (binomial with n=1)
    #spiked = np.nonzero(samples)                                                # indices of neurons with spike events

    # array spike times of last spike for refractory period and synaptic current
    #last_spike[spiked] = t

    # log spikes in raster (plus 1 because neuron index starts with 1)
    #raster[spiked,i] = spiked[0]+1

    # net current equals external input + dot product of synaptic weights and synaptic currents
    popAct[:,i]=esc_rate(i)
    I[:,i] = Iext + synapses.dot(popAct[:,i])                          # add "np.sin(0.01*i)*" for oscillating input

## calculate mean escape rate


## finish progress bar
if use_progbar:
    pbar.finish()

print "Print average firing rate per neuron per second"
frate = 1000*np.sum(popAct)/(T*N)
print frate

## Plotting raster plot and population activity
#print "Plot raster plot..."
#outfile = "spikes.dat"
#np.savetxt(outfile, raster, fmt='%.f', delimiter=' ', newline='\n', header='', footer='', comments='# ')
#plt.title('Spike Raster Plot')
plt.xlim([0,T])
plt.ylim([0.75,N+0.25])
#plt.plot(time, np.transpose(raster), 'k.')
##py.plot(time, 1000*esc_rates[0,:], 'b-')

## Plotting membrane potential
#print "Plot membrane potential..."
#py.title('Membrane Potential')
plt.pcolor(Vm)
plt.colorbar()

#py.ylabel('Rate [Hz]')
#py.ylabel('Neuron index')
#py.xlabel('Time (msec)')
plt.show()
