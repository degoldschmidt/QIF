from __future__ import division  ## true division in Python 2 (NOT NEEDED FOR PYTHON3)
import numpy as np               ## package for scientific computing with Python
from scipy import sparse as sp   ## package for sparse matrices
from scipy import stats as st    ## package for stats
import matplotlib.pyplot as plt  ## package for plotting

## Setup parameters and state variables
N    = 1000                    # number of neurons
T    = 1000                    # total time to simulate (msec)
dt   = 0.5                     # simulation time step (msec)
time = np.arange(0, T+dt, dt)  # time array
print("Start simulation with",N, "neurons for", T/1000., "s (step size:", dt/1000.,"s)")

## QIF/LIF properties
Vrest    = -65                     # resting potential [mV]
Vth      = -50                     # spike threshold [mV]
deltaV   = Vth - Vrest             # difference between threshold and resting
Vm       = np.ones([N,len(time)])  # membrane potential [mV] trace over time
Vm      *= Vrest                   # initial condition of membrane potential
Vreset   = 30
tau_m    = 10                      # time constant [msec]
tau_ref  = 4                       # refractory period [msec]
tau_psc  = 5                       # post synaptic current filter time constant [msec]
tau_esc  = 19                      # escape rate time constant [msec]
beta_esc = 1/4                     # escape rate function slope
Rm       = 1.                      # membrane resistance
a        = 1.-np.exp(-dt/tau_psc)  # smoothing factor for exponential kernel

## Input currents & firing rate containers
print("Generate input...")
I          = np.zeros((N,len(time)))  # net input
Iconst     = 10.                       # constant external input
Iext       = np.ones(N)               # externally applied stimulus
Iext       = Iconst * Iext            # external input set to 0.001 [A]
rate       = np.zeros((N,len(time)))  # population activity (instantaneous)
popAct     = np.zeros(len(time))      # population activity (instantaneous)
psthdt     = 400                      # PSTH time duration [msec]

## Synapse weight matrix
print("Generate weight matrix...")
g        = 1.                                               # recurrent gain parameter
mu_w     = 0                                                # zero mean
sigma_w  = g*(1/N)                                          # variance 1/N for balance
w_conn   = 0.1                                             # connectivity in the weight matrix
rands    = st.norm(loc=mu_w,scale=sigma_w).rvs              # samples from a Gaussian random distr.
w_rec    = sp.random(N, N, density=w_conn, data_rvs=rands)  # generates sparse matrix
#print synapses.nnz                                         # prints number of nonzero elements
#np.random.normal(mu_w, sigma_w, (N,N))                     # Gaussian distributed weights (full network)

## LIF neurons
def f_LIF(i):
    return -(Vm[:,i-1] - Vrest)

## QIF neurons
def f_QIF(i):
    return (Vm[:,i-1] - Vrest)*(Vm[:,i-1] - Vth)/deltaV

## Cumulative exponential distribution
def cum_exp(x):
    return 1-np.exp(-dt*x)

## Escape rate function
def esc_rate(V):
    return (1/tau_esc)*np.exp(beta_esc*(V-Vth))

## Simulate network (i = time steps; t = simulation time)
print("Simulate...")
raster = np.zeros([N,len(time)])*np.nan
for i, t in enumerate(time):

    # Reset mechanism
    Vm[(Vm[:,i-1]>Vth),i-1] = Vrest

    # Euler integration of membrane potential
    Vm[:,i] = Vm[:,i-1] + dt * ( f_QIF(i) + Rm * I[:, i-1]) / tau_m

    # net current equals external input + dot product of synaptic weights and synaptic currents
    rate[:,i]=esc_rate(Vm[:,i])
    popAct[i] = np.sum(rate[:,i])/N
    I[:,i] = Iext + w_rec.dot(rate[:,i])                          # add "np.sin(0.01*i)*" for oscillating input

## calculate mean escape rate
print("Print average firing rate per neuron per second")
frate = 1000*np.sum(popAct)/T
print(frate)
print(rate.shape)
## write parameters and rates into file
params = np.array([[N, T, dt]])
outfile = "./data/rate/net.cfg"
np.savetxt(outfile, params, fmt='%u %u %.2f', delimiter=' ', newline='\n', header='#N #T [ms] #dt [ms]')
outfile = "./data/rate/rates.dat"
np.savetxt(outfile, rate, delimiter=' ', newline='\n')