from __future__ import division
import numpy as np
import pylab as py
import sys
import time
from time import sleep

from progressbar import AnimatedMarker, Percentage, RotatingMarker, Counter, ETA, Bar, ProgressBar

## setup parameters and state variables
N    = 100                     # number of neurons
T    = 15000                   # total time to simulate (msec)
dt   = 0.125                   # simulation time step (msec)
time = np.arange(0, T+dt, dt)  # time array
activeT = int(50./dt)          # duration of input pulse

## progress bar for long simulations
use_progbar = True
if use_progbar:
    widgets = ['Progress: ', Percentage(), ' ', Bar(marker=RotatingMarker()),
           ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=len(time)).start()

## QIF/LIF properties
Vrest    = -65                           # resting potential [V]
Vth      = -50                           # spike threshold [V]
deltaV   = Vth - Vrest                   # difference between threshold and resting
Vm       = Vrest*np.ones([N,len(time)])  # potential [V] trace over time
tau_m    = 10                            # time constant [msec]
tau_ref  = 4                             # refractory period [msec]
tau_psc  = 5                             # post synaptic current filter time constant [msec]
tau_esc  = 19                            # escape rate time constant [msec]
beta_esc = 1/4                           # escape rate function slope
Rm       = 1.                            # membrane resistance
pinput   = 0.05                          # input connectivity

## Input currents
I    = np.zeros((N,len(time)))           # net input
Iexp    = np.zeros((N,len(time)))        # net input
Iext = np.ones((N, len(time)))           # externally applied stimulus
#Ninput = int(pinput*N)                  # number of neurons that receive external input
#print "Input to ",Ninput," neurons."

# Rates
popAct    = np.zeros(len(time))           # population activity (instantaneous)
psthdt    = 400                           # PSTH time duration [msec]
esc_rates = np.zeros((N,len(time)))           # escape rates of each neuron

## constant input to all neurons
print "Generate input..."
Iext = 0.001 * Iext                         # external input set to 0.001 [A]

## time-varying, random input (not used)
#if T >= activeT*dt:
#    for index in np.arange(0,Ninput):
#        #index = np.random.randint(N, size=Ninput)                            # if random indices
#        Iext[index,:] = np.random.uniform(0.,2.5, activeT)                    # uniform random numbers as input
#    Iext = np.concatenate((Iext, np.zeros((N, len(time)-activeT))), axis=1)   # after half of total time no input

## Synapse weight matrix
print "Generate weight matrix..."
g        = 1.                                       # recurrent gain parameter
mu_w     = 0                                        # zero mean
sigma_w  = g*(1/N)                                  # variance 1/N for balance
synapses = np.random.normal(mu_w, sigma_w, (N,N))   # Gaussian distributed weights

## LIF neurons
def f_LIF(i):
    return -(Vm[active,i-1] - Vrest)

## QIF neurons
def f_QIF(i):
    return (Vm[active,i-1] - Vrest)*(Vm[active,i-1] - Vrest)/deltaV

## Synaptic current model (exponential kernel)
def Isyn(t):
    '''t is an array of times since each neuron's last spike event'''
    t[np.nonzero(t < 0)] = 0
    return t*np.exp(-t/tau_psc)
last_spike = np.zeros(N) - tau_ref                  # initialize last spike time vector

## Escape rate function
def esc_rate(i):
    return (1/tau_esc)*np.exp(beta_esc*(Vm[:,i]-Vth))

## Simulate network (i = time steps; t = simulation time)
print "Simulate..."
raster = np.zeros([N,len(time)])*np.nan
for i, t in enumerate(time[1:],1):
    # update progress bar
    if use_progbar:
        pbar.update(i)

    # array of active units (units not in refractory period)
    active = np.nonzero(t > last_spike + tau_ref)

    # Euler integration of membrane potential
    Vm[active,i] = Vm[active,i-1] + ( f_QIF(i) + Rm * I[active,i-1]) / tau_m * dt

    # array of all spikes at time step i
    #spiked = np.nonzero(Vm[:,i] > Vth)                                         # deterministic spike generation (not used)
    esc_rates[:,i] = esc_rate(i)                                                # store escape rates over time
    samples = np.zeros((N,1))                                                   # array of N samples
    samples = np.random.binomial(1, esc_rate(i))                                # stochastic spike generation based on Bernoulli distribution (binomial with n=1)
    spiked = np.nonzero(samples)                                                # indices of neurons with spike events

    # array spike times of last spike for refractory period and synaptic current
    last_spike[spiked] = t

    # log spikes in raster (plus 1 because neuron index starts with 1)
    raster[spiked,i] = spiked[0]+1

    # update population activity [1/sec]
    popAct[i] = (np.nansum(raster[:,i-int(psthdt/dt):i]))/(psthdt*N)

    # net current equals external input + dot product of synaptic weights and synaptic currents
    I[:,i] = Iext[:,i] + synapses.dot(Isyn(t - last_spike))
    Iexp[:,i] = Isyn(t - last_spike)

## calculate mean escape rate


## finish progress bar
if use_progbar:
    pbar.finish()

## Plotting raster plot and population activity
print "Plot raster plot..."
py.title('Spike Raster Plot')
py.xlim([0,T])
py.ylim([0.75,1000+N+0.25])
py.plot(time, np.transpose(raster), 'k.')
py.plot(time, 1000.*popAct, 'r-')
#py.plot(time, 1000*esc_rates[0,:], 'b-')
#py.plot(time, synapses.dot(Iexp)[0,:], 'b-')

## Plotting membrane potential
#print "Plot membrane potential..."
#py.title('Membrane Potential')
#py.pcolor(Vm)
#py.colorbar()

py.ylabel('Rate [Hz]')
py.ylabel('Neuron index')
py.xlabel('Time (msec)')
py.show()
