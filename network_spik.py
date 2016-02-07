from __future__ import division  ## true division in Python 2 (NOT NEEDED FOR PYTHON3)
import numpy as np               ## package for scientific computing with Python
from scipy import sparse as sp   ## package for sparse matrices
from scipy import stats as st    ## package for stats
import matplotlib.pyplot as plt  ## package for plotting

## Setup parameters and state variables
N    = 3000                    # number of neurons
T    = 20000                   # total time to simulate (msec)
dt   = 0.5                     # simulation time step (msec)
time = np.arange(0, T+dt, dt)  # time array
print("Start simulation with",N, "neurons for", T/1000., "s (step size:", dt/1000.,"s)")

## QIF/LIF properties
Vrest    = -65                     # resting potential [mV]
Vth      = -50                     # spike threshold [mV]
deltaV   = Vth - Vrest             # difference between threshold and resting
Vm       = np.ones([N,len(time)])  # membrane potential [mV] trace over time
Vm      *= Vrest                   # initial condition of membrane potential
tau_m    = 10                      # time constant [msec]
tau_ref  = 4                       # refractory period [msec]
tau_psc  = 5                       # post synaptic current filter time constant [msec]
tau_esc  = 19                      # escape rate time constant [msec]
beta_esc = 1/4                     # escape rate function slope
Rm       = 1.                      # membrane resistance
a        = 1.-np.exp(-dt/tau_psc)  # smoothing factor for exponential kernel

## Input currents & spike times containers
print("Generate input...")
I      = np.zeros((N,len(time)))  # net input
Iext   = np.ones(N)               # externally applied stimulus
Iconst = 10.                      # constant external input current
Iext   = Iconst * Iext            # constant inputs to all neurons
Isyn   = np.zeros(N)              # synaptic current is now lowpass equation
spikes = []                       # spike times and index
last_spike = np.zeros(N)-tau_ref  # initialize last spike time vector
samples = np.zeros(N)             # array of N samples of spikes

## Synapse weight matrix
print("Generate weight matrix...")
g        = 1.                                               # recurrent gain parameter
mu_w     = 0                                                # zero mean
sigma_w  = g*(1/N)                                          # variance 1/N for balance
w_conn   = 0.01                                             # connectivity in the weight matrix
rands    = st.norm(loc=mu_w,scale=sigma_w).rvs              # samples from a Gaussian random distr.
w_rec    = sp.random(N, N, density=w_conn, data_rvs=rands)  # generates sparse matrix
#print synapses.nnz                                         # prints number of nonzero elements
#np.random.normal(mu_w, sigma_w, (N,N))                     # Gaussian distributed weights (full network)

## LIF neurons
def f_LIF(i):
    return -(Vm[active,i-1] - Vrest)

## QIF neurons
def f_QIF(i):
    return (Vm[active,i-1] - Vrest)*(Vm[active,i-1] - Vth)/deltaV

## Cumulative exponential distribution
def cum_exp(x):
    return 1-np.exp(-dt*x)

## Escape rate function
def esc_rate(V):
    return (1/tau_esc)*np.exp(beta_esc*(V-Vth))

## Simulate network (i = time steps; t = simulation time)
print("Simulate...")
for i, t in enumerate(time):

    ## Membrane voltage dynamics
    active = np.nonzero(t > last_spike + tau_ref)                                   # array of active units (units not in refractory period)
    Vm[active,i] = Vm[active,i-1] + dt * ( f_QIF(i) + Rm * I[active, i-1]) / tau_m  # Euler integration of membrane potential

    ## Spike generation
    #spiked = np.nonzero(Vm[:,i] > Vth)                          # deterministic spike generation (not used)
    samples = np.random.binomial(1, cum_exp(esc_rate(Vm[:,i])))  # stochastic spike generation based on Bernoulli distribution (binomial with n=1)
    spiked = np.nonzero(samples)                                 # indices of neurons with spike events
    last_spike[spiked] = t                                       # array spike times of last spike for refractory period and synaptic current
    for i in spiked[0]:
        spikes.append([t, i])                                    # append spike time and spiking neuron index
    
    ## Net synaptic current    
    Isyn = (1.-a)*Isyn + a*samples                    # lowpass equation for expontial kernel
    I[:,i] = -np.sin(0.001*i)*Iext + w_rec.dot(Isyn)  # add "np.sin(0.01*i)*" for oscillating input

## calculate mean number of spikes (per neuron and sec) during the whole trial
spikes = np.asarray(spikes)  # need to convert to numpy array to use size and savetxt 
print("Print number of spikes per neuron per second...")
print(1000*np.sum(spikes.size/2)/(T*N))

## write params and spike events into file
params = np.array([[N, T, dt]])
outfile = "./data/spiking/net.cfg"
np.savetxt(outfile, params, fmt='%u %u %.2f', delimiter=' ', newline='\n', header='#N #T [ms] #dt [ms]')
outfile = "./data/spiking/spikes.dat"
np.savetxt(outfile, spikes, fmt='%5.1f\t%4u', delimiter=' ', newline='\n', header='#t_sp [ms] #i_sp')