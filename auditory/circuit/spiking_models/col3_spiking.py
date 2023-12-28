"""
3 unit spiking model of SSA, FS, and functional connectivity


"""

from __future__ import division

import matplotlib.gridspec as gridspec

import numpy as np
import matplotlib.pyplot as plt
#import scipy as sp

import argparse
#import brian2 as b2
from brian2 import *

def eq_pyr(unit):

    if (unit == 1):
        thal = '(thal1 + thal2)'
        synvar = 'iSynE1E2'
        syn_external = '''
        iSynE1E2 = -ge1e2*(v-vE) : amp
        '''
        
        syn_decay = '''
        dge1e2/dt = -ge1e2/ms : siemens
        '''
        
    elif (unit == 2):
        thal = '(al*thal1 + thal2 + al*thal3)'
        synvar = 'iSynE2E1 + iSynE2E3'
        syn_external = '''
        iSynE2E1 = -ge2e1*(v-vE) : amp
        iSynE2E3 = -ge2e3*(v-vE) : amp
        '''

        syn_decay = '''
        dge2e1/dt = -ge2e1/ms : siemens
        dge2e3/dt = -ge2e3/ms : siemens
        '''
        
    elif (unit == 3):
        thal = '(thal2 + thal3)'
        synvar = 'iSynE3E2'
        syn_external = '''
        iSynE3E2 = -ge3e2*(v-vE) : amp        
        '''

        syn_decay = '''
        dge3e2/dt = -ge3e2/ms : siemens
        '''
                
    else:
        raise ValueError('Invalid unit choice',str(unit))
            
    eqs_e = '''
    # E soma
    dv/dt=( -w + Itot + gL*dT*exp((v-vT)/dT) + sigma*xi*(nS*Cm)**.5 )/Cm : volt (unless refractory)
    
    iL = -gL*(v-EL) : amp
    iDend = -gsd*(v-vD)/(0.3) : amp
    
    thal1 = q*IFast1*nA*DSlow1*DFast1 : amp
    thal2 = q*IFast2*nA*DSlow2*DFast2 : amp
    thal3 = q*IFast3*nA*DSlow3*DFast3 : amp
    
    # total current
    Itot = iDend +'''+synvar+'''+ iSynEE'''+str(unit)+''' + iSynEP'''+str(unit)+''' + iL + I + '''+thal+''' : amp
    
    # soma synapses
    
    '''+syn_external+'''

    iSynEE'''+str(unit)+''' = -gee*(v-vE) : amp
    #iSynEP'''+str(unit)+''' = -gep*(1-2.5*(1-DSlow'''+str(unit)+'''))*(v-vI) : amp
    iSynEP'''+str(unit)+''' = -gep*(1-1.7*(1-DSlow'''+str(unit)+'''))*(v-vI) : amp
    #iSynEP'''+str(unit)+''' = -gep*(1-0.*(1-DSlow'''+str(unit)+'''))*(v-vI) : amp

    
    # synapse decay
    '''+syn_decay+'''
    dgee/dt = -gee/ms : siemens
    dgep/dt = -gep/ms : siemens
    dges/dt = -ges/ms : siemens
    
    # adaptation
    dw/dt = (a*(v-EL) - w)/tauw : amp
    
    # dendrite
    dvD/dt = (iDendL + iSoma + iDendSyn)/Cm : volt
    iDendL = -gL*(vD-EL) : amp
    iSoma = -gsd*(vD-v)/(0.7) : amp
    #iDendSyn = -ges*(1+3*(1-DSlow'''+str(unit)+'''))*(vD-vI) : amp
    iDendSyn = -ges*(1+3*(1-DSlow'''+str(unit)+'''))*(vD-vI) : amp
    
    # synaptic depression (thalamus, slow)
    dDSlow1/dt = (1-DSlow1)/tauD_slow1 - DSlow1*stimulus1(t)/tauD_slow2 : 1
    # synaptic depression (thalamus, fast)
    dDFast1/dt = (1-DFast1 - stimulus1(t))/tauD_fast : 1
    # fast input facilitation
    dIFast1/dt = (-IFast1 + stimulus1(t))/tauI_fast : 1
    
    # synaptic depression (thalamus, slow)
    dDSlow2/dt = (1-DSlow2)/tauD_slow1 - DSlow2*stimulus2(t)/tauD_slow2 : 1
    # synaptic depression (thalamus, fast)
    dDFast2/dt = (1-DFast2 - stimulus2(t))/tauD_fast : 1
    # fast input facilitation
    dIFast2/dt = (-IFast2 + stimulus2(t))/tauI_fast : 1
    
    # synaptic depression (thalamus, slow)
    dDSlow3/dt = (1-DSlow3)/tauD_slow1 - DSlow3*stimulus3(t)/tauD_slow2 : 1
    # synaptic depression (thalamus, fast)
    dDFast3/dt = (1-DFast3 - stimulus3(t))/tauD_fast : 1
    # fast input facilitation
    dIFast3/dt = (-IFast3 + stimulus3(t))/tauI_fast : 1
    
    
    gL : siemens
    vT : volt
    dT : volt
    a : siemens
    I : amp
    Cm : farad
    '''

    return eqs_e


def eq_pv(unit):
    
    if (unit == 1):
        thal = '(thal1 + thal2)'

        synvar = 'iSynP1E2'
        syn_external = '''
        iSynP1E2 = -gp1e2*(v-vE) : amp
        '''

        syn_decay = '''
        dgp1e2/dt = -gp1e2/ms : siemens
        '''
        
    elif (unit == 2):
        thal = '(al*thal1 + thal2 + al*thal3)'

        synvar = 'iSynP2E1 + iSynP2E3'
        syn_external = '''
        iSynP2E1 = -gp2e1*(v-vE) : amp
        iSynP2E3 = -gp2e3*(v-vE) : amp
        '''

        syn_decay = '''
        dgp2e1/dt = -gp2e1/ms : siemens
        dgp2e3/dt = -gp2e3/ms : siemens
        '''
        
    elif (unit == 3):
        thal = '(thal2 + thal3)'

        synvar = 'iSynP3E2'
        syn_external = '''
        iSynP3E2 = -gp3e2*(v-vE) : amp
        '''

        syn_decay = '''
        dgp3e2/dt = -gp3e2/ms : siemens
        '''
                
    else:
        raise ValueError('Invalid unit choice',str(unit))

    
    eqs_p = '''
    # PV
    dv/dt=(Itot - pv_opto - w + iL + gL*dT*exp((v-vT)/dT) + sigma*xi*(nS*Cm)**.5)/Cm : volt (unless refractory)
    iL = -gL*(v-EL) : amp

    Itot = '''+thal+''' + I*pv_opto2 +'''+synvar+''' + iSynPE'''+str(unit)+''' + iSynPP'''+str(unit)+''' + iSynPS'''+str(unit)+''': amp
    
    thal1 = q*IFast1*nA*DSlow1*DFast1 : amp
    thal2 = q*IFast2*nA*DSlow2*DFast2 : amp
    thal3 = q*IFast3*nA*DSlow3*DFast3 : amp
    
    # soma synapses
    '''+syn_external+'''
    iSynPE'''+str(unit)+''' = -gpe*(v-vE) : amp
    iSynPP'''+str(unit)+''' = -gpp*(v-vI) : amp
    iSynPS'''+str(unit)+''' = -gps*(v-vI) : amp
    
    # synapse decay
    '''+syn_decay+'''
    dgpe/dt = -gpe/(25*ms) : siemens
    dgpp/dt = -gpp/ms : siemens
    dgps/dt = -gps/ms : siemens
    
    # adaptation
    dw/dt = (a*(v-EL) - w)/tauw : amp
    
    # synaptic depression (thalamus, slow)
    dDSlow1/dt = (1-DSlow1)/tauD_slow1 - DSlow1*stimulus1(t)/tauD_slow2 : 1
    # synaptic depression (thalamus, fast)
    dDFast1/dt = (1-DFast1 - stimulus1(t))/tauD_fast : 1
    # fast input facilitation
    dIFast1/dt = (-IFast1 + stimulus1(t))/tauI_fast : 1
    
    # synaptic depression (thalamus, slow)
    dDSlow2/dt = (1-DSlow2)/tauD_slow1 - DSlow2*stimulus2(t)/tauD_slow2 : 1
    # synaptic depression (thalamus, fast)
    dDFast2/dt = (1-DFast2 - stimulus2(t))/tauD_fast : 1
    # fast input facilitation
    dIFast2/dt = (-IFast2 + stimulus2(t))/tauI_fast : 1
    
    # synaptic depression (thalamus, slow)
    dDSlow3/dt = (1-DSlow3)/tauD_slow1 - DSlow3*stimulus3(t)/tauD_slow2 : 1
    # synaptic depression (thalamus, fast)
    dDFast3/dt = (1-DFast3 - stimulus3(t))/tauD_fast : 1
    # fast input facilitation
    dIFast3/dt = (-IFast3 + stimulus3(t))/tauI_fast : 1
    
    gL : siemens
    vT : volt
    dT : volt
    a : siemens
    I : amp
    Cm : farad
    '''

    return eqs_p

def eq_som(unit):

    if unit == 1:
        thal = ''
        synvar = 'iSynS1E2'
        syn_external = '''
        iSynS1E2 = -gs1e2*(v-vE) : amp
        '''

        syn_decay = '''
        dgs1e2/dt = -gs1e2/ms : siemens
        '''

    elif unit == 2:
        thal = ''
        synvar = 'iSynS2E1 + iSynS2E3'
        syn_external = '''
        iSynS2E1 = -gs2e1*(v-vE) : amp
        iSynS2E3 = -gs2e3*(v-vE) : amp
        '''

        syn_decay = '''
        dgs2e1/dt = -gs2e1/ms : siemens
        dgs2e3/dt = -gs2e3/ms : siemens
        '''

    elif unit == 3:
        thal = ''
        synvar = 'iSynS3E2'
        syn_external = '''
        iSynS3E2 = -gs3e2*(v-vE) : amp
        '''

        syn_decay = '''
        dgs3e2/dt = -gs3e2/ms : siemens
        '''
    
        
    eqs_s = '''
    # SOM
    dv/dt=( gL*dT*exp((v-vT)/dT) + Itot - som_opto - w + iL  + sigma*xi*(nS*Cm)**.5)/Cm : volt (unless refractory)
    iL = -gL*(v-EL) : amp
    
    Itot = + iSynSE'''+str(unit)+''' + I: amp
    # soma synapses
    '''+syn_external+'''
    iSynSE'''+str(unit)+''' = -gse*(v-vE) : amp
    
    # synapse decay
    '''+syn_decay+'''
    dgse/dt = -gse/(15*ms) : siemens
    #dgpp/dt = -gpp/ms : siemens
    #dgps/dt = -gps/ms : siemens
    
    # adaptation
    dw/dt = (a*(v-EL) - w)/tauw : amp
    
    gL : siemens
    vT : volt
    dT : volt
    a : siemens
    I : amp
    Cm : farad
    '''

    return eqs_s



def get_spikes(start_time,end_time,sim_dt,spikemon_t):
    """
    get firing rate given spikemonitor and time interval.

    start_time: time of input start
    end_time: time of input end
    sim_dt: simulation time step
    spikemon_t: time array from spike monitor
    """

    start_idx = int(start_time/sim_dt)
    end_idx = int(end_time/sim_dt)

    # convert spikemon to ms time units
    ms_units = (spikemon_t/ms)

    # mark all times where there was a spike between the start and end times. add everything.
    total_spikes = np.sum((ms_units>start_time)*(ms_units<end_time))

    return total_spikes


def get_FR(start_time,end_time,sim_dt,spikemon_t,n):
    """
    get firing rate given spikemonitor and time interval.

    start_time: time of input start
    end_time: time of input end
    sim_dt: simulation time step
    spikemon_t: time array from spike monitor
    n: total number of neurons
    
    """

    start_idx = int(start_time/sim_dt)
    end_idx = int(end_time/sim_dt)

    # convert spikemon to ms time units
    ms_units = (spikemon_t/ms)

    # mark all times where there was a spike between the start and end times. add everything.
    total_spikes = np.sum((ms_units>start_time)*(ms_units<end_time))

    # convert ms time interval to sec
    time_interval = (end_time-start_time)/1000.


    # convert to Hz
    firing_rate = total_spikes/time_interval


    # normalize by total number of neurons
    firing_rate /= n
    
    return firing_rate


def collect_spikes(stim_array,stim_dt,sim_dt,spikemon_t):
    """
    get total spike count for a given spikemon
    
    stim_array: array of stimulation time/strengths, e.g., [0,0,0,0,1,0,0,0,1,0,0]
    stim_dt: time interval of stimulation
    sim_dt: time interval of simulation
    spikemon_t: time array from spike monitor
    
    """

    # get all stim start times (index position*stim_dt)
    stim_start_times = np.where(stim_array!=0)[0]*stim_dt
    
    # preallocate firing rate array
    spike_array = np.zeros(len(stim_start_times))
    
    for i in range(len(stim_start_times)):
        spike_array[i] = get_spikes(stim_start_times[i],stim_start_times[i]+stim_dt,sim_dt,spikemon_t)
    
    return spike_array


def collect_FR(stim_array,stim_dt,sim_dt,spikemon_t,n):
    """
    get all firing rates for a given spikemon
    
    stim_array: array of stimulation time/strengths, e.g., [0,0,0,0,1,0,0,0,1,0,0]
    stim_dt: time interval of stimulation
    sim_dt: time interval of simulation
    spikemon_t: time array from spike monitor
    
    """
    
    # get all stim start times (index position*stim_dt)
    stim_start_times = np.where(stim_array!=0)[0]*stim_dt
    
    # preallocate firing rate array
    FR_array = np.zeros(len(stim_start_times))
    
    for i in range(len(stim_start_times)):
        FR_array[i] = get_FR(stim_start_times[i],stim_start_times[i]+stim_dt,sim_dt,spikemon_t,n)
    
    return FR_array
    

def get_FR_dev(start_time,end_time,sim_dt,spikemon,n):
    """
    get firing rate given spikemonitor and time interval.
    quantify deviation as well.

    start_time: time of input start
    end_time: time of input end
    sim_dt: simulation time step
    spikemon_t: time array from spike monitor
    n: total number of neurons
    
    """

    spikemon_t = spikemon.t
    
    # convert spikemon to ms time units
    ms_units = (spikemon_t/ms)

    # spike count for each neuron
    spike_counts = np.zeros(n)
    
    # loop over indices and count spikes for each index.
    for i in range(n):
        # get boolean index where neuron i appears
        neuron_idx = spikemon.i == i

        # get boolean index where neurons spike in time interval [start_time, end_time]
        time_idx = (ms_units>start_time)*(ms_units<end_time)

        # get total count of neuron i spiking in this time interval
        spike_counts[i] = np.sum(neuron_idx*time_idx)

    # get std deviation
    dev = np.std(spike_counts)
    #print 'dev',dev,type(dev)
    
    
    # mark all times where there was a spike between the start and end times. add everything.
    total_spikes = np.sum(spike_counts)

    # convert ms time interval to sec
    time_interval = (end_time-start_time)/1000.

    # convert to Hz
    firing_rate = total_spikes/time_interval

    # normalize by total number of neurons
    firing_rate /= n

    #print 'firing_rate,dev',firing_rate,dev
    return firing_rate,dev,spike_counts


def collect_FR_dev(stim_array,stim_dt,sim_dt,spikemon,n,return_spikes=False):
    """
    get all firing rates for a given spikemon
    
    stim_array: array of stimulation time/strengths, e.g., [0,0,0,0,1,0,0,0,1,0,0]
    stim_dt: time interval of stimulation
    sim_dt: time interval of simulation
    spikemon_t: time array from spike monitor

    returns:
    spikelist: (n,len(stim_start_times)) matrix of spike counts.
    """

    #print 'spikemon.i min',np.amin(spikemon.i),np.amax(spikemon.i)
    spikemon_t = spikemon
    
    # get all stim start times (index position*stim_dt)
    stim_start_times = np.where(stim_array!=0)[0]*stim_dt
    
    # preallocate firing rate array and standard deviation
    FR_array = np.zeros(len(stim_start_times))
    dev_array = np.zeros(len(stim_start_times))
    spikelist = np.zeros((n,len(stim_start_times)))        
    
    for i in range(len(stim_start_times)):
        FR_array[i],dev_array[i],spikelist[:,i] = get_FR_dev(stim_start_times[i],stim_start_times[i]+stim_dt,sim_dt,spikemon,n)

    #print 'type',type(dev_array)

    return FR_array,dev_array,spikelist


def setup_and_run(pv_opto=0,som_opto=0,paradigm='ssa',seed=0,pv_opto2=1.,pars={'wee':1,'taud1':1000},dt=0.5):
    
    np.random.seed(seed)

    multiplier = 2
    
    n_pyr = 800*multiplier
    n_pv = 100*multiplier
    n_som = 100*multiplier

    vE = 0.*mV
    vI = -67*mV

    # matrix mult 20, default params seems to work okay with seed 3    
    # weight matrix
    multiplier = 20.#20.
    W = np.zeros((3,3))
    

    if (paradigm == 'ssa') or (paradigm == 'fs'):
        W[0,0] = pars['wee']#.5#0.9 # wee
        W[0,1] = 2. # wep
        W[0,2] = 1.#0.5 # wes
        
        W[1,0] = .1#.1#.1 # wpe
        W[1,1] = 2.#1.#2. # wpp
        W[1,2] = 2.#.6#0.6 # wps

        
    else:
        W[0,0] = .5#0.9 # wee
        W[0,1] = 2. #2 # wep
        W[0,2] = 1#0.5 # wes

        
        W[1,0] = .1#.1#.1 # wpe
        W[1,1] = 2.#1.#2. # wpp
        W[1,2] = 2.#.6#0.6 # wps


    W[2,0] = 6#6.#10.#8 # wse
    W[2,1] = 0.
    W[2,2] = 0.

    W *= multiplier

    b = 8*pA # weight increment

    #a = 4.*nS
    tauw = 150.*ms
    gsd = 18.75*nS
    
    gee_max = W[0,0]*nS#1.66*nS
    gep_max = W[0,1]*nS#136.4*nS
    ges_max = W[0,2]*nS#68.2*nS

    gpe_max = W[1,0]*nS#5.*nS
    gpp_max = W[1,1]*nS#(136.4*nS)/10
    gps_max = W[1,2]*nS#45.5*nS

    gse_max = W[2,0]*nS#5*(1.66*nS)
    #gsp_max = 136.4*nS
    #gss_max = 45.5*nS

    EL = -60*mV

    sigma = 20*mV#20*mV
    tau = 10*ms
    tauD_fast = 10*ms#11*ms

    tauD_slow1 = pars['taud1']*ms#1500*ms
    tauD_slow2 = 250*ms#200*ms

    tauI_fast = 1*ms

    # timed inputs

    al = .85#.9

    if paradigm == 'ssa':
        q = 1#.95#.85
        stim_dt = 100
        stim_arr1 = np.array([0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0])
        stim_arr2 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        stim_arr3 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])


    
    elif paradigm == 'ssa2':
        q = 1#.95#.85
        stim_dt = 100
        stim_arr1 = np.array([0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0])
        stim_arr2 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        stim_arr3 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        
    elif paradigm == 'fs1':
        q = .4#.95#.85
        stim_dt = 10
        stim_arr1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        stim_arr2 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0])
        stim_arr3 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])        
        
    elif paradigm == 'fs2':
        q = .4#.95#.85
        stim_dt = 10
        stim_arr1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        stim_arr2 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0])
        stim_arr3 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])        
        
    elif paradigm == 'fs3':
        q = .4#.95#.85
        stim_dt = 10
        stim_arr1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        stim_arr2 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0])
        stim_arr3 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        
    elif paradigm == 'pv':
        q = .7#.4#.95#.85
        stim_dt = 10
        stim_arr1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        stim_arr2 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0])
        stim_arr3 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

    elif paradigm == 'tuning1':
        q = 1#.95#.85
        stim_dt = 100
        stim_arr1 = np.array([0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0])
        stim_arr2 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        stim_arr3 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

    elif paradigm == 'tuning2':
        q = 1#.95#.85
        stim_dt = 100
        stim_arr1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        stim_arr2 = np.array([0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0])
        stim_arr3 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

    elif paradigm == 'tuning3':
        q = 1#.95#.85
        stim_dt = 100
        stim_arr1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        stim_arr2 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        stim_arr3 = np.array([0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0])

    else:
        raise ValueError('Invalid experimental paradigm =',paradigm)
        
        
    
    stimulus1 = TimedArray(
        stim_arr1,
        dt=stim_dt*ms)

    #stim_arr1[np.where(stim_arr1!=0)[0]+1]=1
    #stimulus1b = TimedArray(
    #    stim_arr1,
    #    dt=stim_dt*ms)

    

    stimulus2 = TimedArray(
        stim_arr2,
        dt=stim_dt*ms)

    #stim_arr2[np.where(stim_arr2!=0)[0]+1]=1
    #stimulus2b = TimedArray(
    #    stim_arr2,
    #    dt=stim_dt*ms)
    
    stimulus3 = TimedArray(
        stim_arr3,
        dt=stim_dt*ms)
    
    som_opto = som_opto*nA    
    pv_opto = pv_opto*nA
    pv_opto2 = pv_opto2
    
    #print('pv_opto =',pv_opto,'; som_opto =',som_opto,'pv_opto2 =',pv_opto2)
    

    ############################### Neuron groups and parameters
    G_PYR1 = NeuronGroup(n_pyr, eq_pyr(1), threshold='v>20*mV', reset='v=-60*mV;w+=b', method='Euler',refractory=2*ms) # pyr
    G_PYR1.Cm = 180*pF
    G_PYR1.gL = 6.25*nS
    G_PYR1.dT = 1.*mV # 0.25 for PV
    G_PYR1.vT = -40.*mV # randomly distributed and different for SOMs
    G_PYR1.a = 4.*nS

    G_PYR2 = NeuronGroup(n_pyr, eq_pyr(2), threshold='v>20*mV', reset='v=-60*mV;w+=b', method='Euler',refractory=2*ms) # pyr
    G_PYR2.Cm = 180*pF
    G_PYR2.gL = 6.25*nS
    G_PYR2.dT = 1.*mV # 0.25 for PV
    G_PYR2.vT = -40.*mV # randomly distributed and different for SOMs
    G_PYR2.a = 4.*nS

    G_PYR3 = NeuronGroup(n_pyr, eq_pyr(3), threshold='v>20*mV', reset='v=-60*mV;w+=b', method='Euler',refractory=2*ms) # pyr
    G_PYR3.Cm = 180*pF
    G_PYR3.gL = 6.25*nS
    G_PYR3.dT = 1.*mV # 0.25 for PV
    G_PYR3.vT = -40.*mV # randomly distributed and different for SOMs
    G_PYR3.a = 4.*nS

    
    G_PV1 = NeuronGroup(n_pv, eq_pv(1), threshold='v>20*mV', reset='v=-60*mV', method='Euler',refractory=2*ms)
    G_PV1.Cm = 80.*pF
    G_PV1.gL = 5.*nS
    G_PV1.dT = 0.25*mV
    G_PV1.vT = -40.*mV
    G_PV1.a = 0.*nS

    G_PV2 = NeuronGroup(n_pv, eq_pv(2), threshold='v>20*mV', reset='v=-60*mV', method='Euler',refractory=2*ms)
    G_PV2.Cm = 80.*pF
    G_PV2.gL = 5.*nS
    G_PV2.dT = 0.25*mV
    G_PV2.vT = -40.*mV
    G_PV2.a = 0.*nS

    
    G_PV3 = NeuronGroup(n_pv, eq_pv(3), threshold='v>20*mV', reset='v=-60*mV', method='Euler',refractory=2*ms)
    G_PV3.Cm = 80.*pF
    G_PV3.gL = 5.*nS
    G_PV3.dT = 0.25*mV
    G_PV3.vT = -40.*mV
    G_PV3.a = 0.*nS

    G_SOM1 = NeuronGroup(n_som, eq_som(1), threshold='v>20*mV', reset='v=-60*mV', method='Euler',refractory=2*ms)
    G_SOM1.Cm = 80.*pF
    G_SOM1.gL = 5.*nS
    G_SOM1.dT = 1.*mV
    G_SOM1.vT = -45.*mV
    G_SOM1.a = 4.*nS
    
    G_SOM2 = NeuronGroup(n_som, eq_som(2), threshold='v>20*mV', reset='v=-60*mV', method='Euler',refractory=2*ms)
    G_SOM2.Cm = 80.*pF
    G_SOM2.gL = 5.*nS
    G_SOM2.dT = 1.*mV
    G_SOM2.vT = -45.*mV
    G_SOM2.a = 4.*nS

    G_SOM3 = NeuronGroup(n_som, eq_som(3), threshold='v>20*mV', reset='v=-60*mV', method='Euler',refractory=2*ms)
    G_SOM3.Cm = 80.*pF
    G_SOM3.gL = 5.*nS
    G_SOM3.dT = 1.*mV
    G_SOM3.vT = -45.*mV
    G_SOM3.a = 4.*nS

    ############################### inputs
    # baseline
    G_PYR1.I[:] = .35*nA#.35*nA
    G_PV1.I[:] = .05*nA#.05*nA


    G_PYR2.I[:] = .35*nA#.35*nA
    G_PV2.I[:] = .05*nA#.05*nA


    G_PYR3.I[:] = .35*nA#.35*nA
    G_PV3.I[:] = .05*nA#.05*nA


    if (paradigm == 'ssa') or (paradigm == 'fs'):
        G_SOM1.I[:] = .1*nA#.025*nA#.025*nA
        G_SOM2.I[:] = .1*nA#.025*nA#.025*nA
        G_SOM3.I[:] = .1*nA#.025*nA#.025*nA
    else:
        G_SOM1.I[:] = .1*nA#.025*nA#.025*nA
        G_SOM2.I[:] = .1*nA#.025*nA#.025*nA
        G_SOM3.I[:] = .1*nA#.025*nA#.025*nA

    ############################### synapses unit 1
    # inhibitory conductances are incremented positively here, but they give rise to negative currents based on the inhibitory reversal potential in the equations above.
    SynEE1 = Synapses(G_PYR1,G_PYR1,on_pre='gee += gee_max/n_pyr')
    SynEE1.connect(p=.1)
    SynEP1 = Synapses(G_PV1,G_PYR1,on_pre='gep += gep_max/n_pv')
    SynEP1.connect(p=.6)
    SynES1 = Synapses(G_SOM1,G_PYR1,on_pre='ges += ges_max/n_som')
    SynES1.connect(p=.6)

    SynPE1 = Synapses(G_PYR1,G_PV1,on_pre='gpe += gpe_max/n_pyr')
    SynPE1.connect(p=.6)
    SynPP1 = Synapses(G_PV1,G_PV1,on_pre='gpp += gpp_max/n_pv')
    #SynPP1 = Synapses(G_PV1,G_PV1,on_pre='gpp -= 0*nS')
    SynPP1.connect(p=.6)
    SynPS1 = Synapses(G_SOM1,G_PV1,on_pre='gps += gps_max/n_som')
    #SynPS1 = Synapses(G_SOM1,G_PV1,on_pre='gps -= 0*nS')
    SynPS1.connect(p=.6)

    SynSE1 = Synapses(G_PYR1,G_SOM1,on_pre='gse += gse_max/n_pyr') 
    SynSE1.connect(p=.6)
    #SynSP1 = Synapses(G_PYR1,G_PV1,on_pre='gep -= gep_max')
    #SynSP1.connect(p=.6)
    #SynSS1 = Synapses(G_PYR1,G_SOM1,on_pre='gep -= gep_max')
    #SynSS1.connect(p=.6)

    ############################### synapses between units 1 and 2

    # E2->E1
    SynE1E2 = Synapses(G_PYR2,G_PYR1,on_pre='ge1e2 += gee_max/n_pyr')
    SynE1E2.connect(p=.1)

    # E1->E2
    SynE2E1 = Synapses(G_PYR1,G_PYR2,on_pre='ge2e1 += gee_max/n_pyr')
    SynE2E1.connect(p=.1)

    # E1->P2
    SynP2E1 = Synapses(G_PYR1,G_PV2,on_pre='gp2e1 += gpe_max/n_pyr')
    SynP2E1.connect(p=.1)

    # E2->P1
    SynP1E2 = Synapses(G_PYR2,G_PV1,on_pre='gp1e2 += gpe_max/n_pyr')
    SynP1E2.connect(p=.1)

    # E1->S2
    SynS2E1 = Synapses(G_PYR1,G_SOM2,on_pre='gs2e1 += gse_max/n_pyr')
    SynS2E1.connect(p=.1)

    # E2->S1
    SynS1E2 = Synapses(G_PYR2,G_SOM1,on_pre='gs1e2 += gse_max/n_pyr')
    SynS1E2.connect(p=.1)


    ############################### synapses unit 2
    # inhibitory conductances are incremented positively here, but they give rise to negative currents based on the inhibitory reversal potential in the equations above.
    SynEE2 = Synapses(G_PYR2,G_PYR2,on_pre='gee += gee_max/n_pyr')
    SynEE2.connect(p=.1)
    SynEP2 = Synapses(G_PV2,G_PYR2,on_pre='gep += gep_max/n_pv')
    SynEP2.connect(p=.6)
    SynES2 = Synapses(G_SOM2,G_PYR2,on_pre='ges += ges_max/n_som')
    SynES2.connect(p=.6)

    SynPE2 = Synapses(G_PYR2,G_PV2,on_pre='gpe += gpe_max/n_pyr')
    SynPE2.connect(p=.6)
    SynPP2 = Synapses(G_PV2,G_PV2,on_pre='gpp += gpp_max/n_pv')
    #SynPP2 = Synapses(G_PV2,G_PV2,on_pre='gpp -= 0*nS')
    SynPP2.connect(p=.6)
    SynPS2 = Synapses(G_SOM2,G_PV2,on_pre='gps += gps_max/n_som')
    #SynPS2 = Synapses(G_SOM2,G_PV2,on_pre='gps -= 0*nS')
    SynPS2.connect(p=.6)

    SynSE2 = Synapses(G_PYR2,G_SOM2,on_pre='gse += gse_max/n_pyr') 
    SynSE2.connect(p=.6)
    #SynSP2 = Synapses(G_PYR2,G_PV2,on_pre='gep -= gep_max')
    #SynSP2.connect(p=.6)
    #SynSS2 = Synapses(G_PYR2,G_SOM2,on_pre='gep -= gep_max')
    #SynSS2.connect(p=.6)    


    ############################### synapses between units 2 and 3
    SynE2E3 = Synapses(G_PYR3,G_PYR2,on_pre='ge2e3 += gee_max/n_pyr')
    SynE2E3.connect(p=.1)

    SynE3E2 = Synapses(G_PYR2,G_PYR3,on_pre='ge3e2 += gee_max/n_pyr')
    SynE3E2.connect(p=.1)
    
    # E2->P3
    SynP3E2 = Synapses(G_PYR2,G_PV3,on_pre='gp3e2 += gpe_max/n_pyr')
    SynP3E2.connect(p=.1)

    # E3->P2
    SynP2E3 = Synapses(G_PYR3,G_PV2,on_pre='gp2e3 += gpe_max/n_pyr')
    SynP2E3.connect(p=.1)

    # E2->S3
    SynS3E2 = Synapses(G_PYR2,G_SOM3,on_pre='gs3e2 += gse_max/n_pyr')
    SynS3E2.connect(p=.1)

    # E3->S2
    SynS2E3 = Synapses(G_PYR3,G_SOM2,on_pre='gs2e3 += gse_max/n_pyr')
    SynS2E3.connect(p=.1)

    
    ############################### synapses unit 3
    # inhibitory conductances are incremented positively here, but they give rise to negative currents based on the inhibitory reversal potential in the equations above.
    SynEE3 = Synapses(G_PYR3,G_PYR3,on_pre='gee += gee_max/n_pyr')
    SynEE3.connect(p=.1)
    SynEP3 = Synapses(G_PV3,G_PYR3,on_pre='gep += gep_max/n_pv')
    SynEP3.connect(p=.6)
    SynES3 = Synapses(G_SOM3,G_PYR3,on_pre='ges += ges_max/n_som')
    SynES3.connect(p=.6)

    SynPE3 = Synapses(G_PYR3,G_PV3,on_pre='gpe += gpe_max/n_pyr')
    SynPE3.connect(p=.6)
    SynPP3 = Synapses(G_PV3,G_PV3,on_pre='gpp += gpp_max/n_pv')
    #SynPP3 = Synapses(G_PV3,G_PV3,on_pre='gpp -= 0*nS')
    SynPP3.connect(p=.6)
    SynPS3 = Synapses(G_SOM3,G_PV3,on_pre='gps += gps_max/n_som')
    #SynPS3 = Synapses(G_SOM3,G_PV3,on_pre='gps -= 0*nS')
    SynPS3.connect(p=.6)

    SynSE3 = Synapses(G_PYR3,G_SOM3,on_pre='gse += gse_max/n_pyr') 
    SynSE3.connect(p=.6)
    #SynSP3 = Synapses(G_PYR3,G_PV3,on_pre='gep -= gep_max')
    #SynSP3.connect(p=.6)
    #SynSS3 = Synapses(G_PYR3,G_SOM3,on_pre='gep -= gep_max')
    #SynSS3.connect(p=.6)    

    ############################### run
    G_PYR1.v[:] = EL+10*mV
    G_PYR1.vD[:] = EL+10*mV

    G_PYR2.v[:] = EL+10*mV
    G_PYR2.vD[:] = EL+10*mV

    G_PYR3.v[:] = EL+10*mV
    G_PYR3.vD[:] = EL+10*mV

    
    G_PYR1.DSlow1[:] = 1
    G_PYR1.DSlow2[:] = 1
    G_PYR1.DSlow3[:] = 1
    G_PYR1.DFast1[:] = 1
    G_PYR1.DFast2[:] = 1
    G_PYR1.DFast3[:] = 1

    G_PYR2.DSlow1[:] = 1
    G_PYR2.DSlow2[:] = 1
    G_PYR2.DSlow3[:] = 1
    G_PYR2.DFast1[:] = 1
    G_PYR2.DFast2[:] = 1
    G_PYR2.DFast3[:] = 1

    G_PYR3.DSlow1[:] = 1
    G_PYR3.DSlow2[:] = 1
    G_PYR3.DSlow3[:] = 1    
    G_PYR3.DFast1[:] = 1
    G_PYR3.DFast2[:] = 1
    G_PYR3.DFast3[:] = 1


    G_PV1.DSlow1[:] = 1
    G_PV1.DSlow2[:] = 1
    G_PV1.DSlow3[:] = 1    
    G_PV1.DFast1[:] = 1
    G_PV1.DFast2[:] = 1
    G_PV1.DFast3[:] = 1

    G_PV2.DSlow1[:] = 1
    G_PV2.DSlow2[:] = 1
    G_PV2.DSlow3[:] = 1    
    G_PV2.DFast1[:] = 1
    G_PV2.DFast2[:] = 1
    G_PV2.DFast3[:] = 1
    
    G_PV3.DSlow1[:] = 1
    G_PV3.DSlow2[:] = 1
    G_PV3.DSlow3[:] = 1    
    G_PV3.DFast1[:] = 1
    G_PV3.DFast2[:] = 1
    G_PV3.DFast3[:] = 1

    G_PV1.v[:] = EL
    G_SOM1.v[:] = EL

    G_PV2.v[:] = EL
    G_SOM2.v[:] = EL

    G_PV3.v[:] = EL
    G_SOM3.v[:] = EL

    #M_PYR1 = StateMonitor(G_PYR1, ['v','vD','w','ges',
    #                               'DSlow1','DFast1','IFast1','DSlow3','DFast3','IFast3',
    #                               'thal1','thal2','thal3'], record=True)

    #    iSynEP'''+str(unit)+''' = -gep*(1-(1-DSlow'''+str(unit)+'''))*(v-vI) : amp
    M_PYR2 = StateMonitor(G_PYR2, ['thal2','DSlow2','v'], record=True)

    #M_PYR2 = StateMonitor(G_PYR2, ['v','vD','w','ges',
    #                               'DSlow1','DFast1','IFast1','DSlow3','DFast3','IFast3',
    #                               'thal1','thal2','thal3'], record=True)
    
    #M_PV1 = StateMonitor(G_PV1, ['v','DSlow1','DFast1','thal1','thal2','thal3'], record=True)
    #M_SOM1 = StateMonitor(G_SOM1, ['v'], record=True)
    
    spikemon_PYR1 = SpikeMonitor(G_PYR1)
    spikemon_PYR2 = SpikeMonitor(G_PYR2)
    spikemon_PYR3 = SpikeMonitor(G_PYR3)
    
    #spikemon_PV1 = SpikeMonitor(G_PV1)
    #spikemon_SOM1 = SpikeMonitor(G_SOM1)
    
    defaultclock.dt = dt*ms
    T = len(stim_arr1)*stim_dt*ms
    run(T)

    fulldict = {}

    fulldict['som_opto'] = som_opto
    fulldict['pv_opto'] = pv_opto
    fulldict['pv_opto2'] = pv_opto2
    
    #fulldict['M_PYR1'] = M_PYR1
    fulldict['M_PYR2'] = M_PYR2
    #fulldict['M_PV1'] = M_PV1
    #fulldict['M_SOM1'] = M_SOM1

    fulldict['spikemon_PYR1'] = spikemon_PYR1
    fulldict['spikemon_PYR2'] = spikemon_PYR2
    fulldict['spikemon_PYR3'] = spikemon_PYR3
    
    #fulldict['spikemon_PV1'] = spikemon_PV1
    #fulldict['spikemon_SOM1'] = spikemon_SOM1

    fulldict['stim_arr1'] = stim_arr1
    fulldict['stim_arr2'] = stim_arr2
    fulldict['stim_arr3'] = stim_arr3
    
    fulldict['stim_dt'] = stim_dt

    fulldict['T'] = T
    fulldict['defaultclock'] = defaultclock.dt
    fulldict['n_pyr'] = n_pyr

    return fulldict
    

def plot(fulldict,choice):

    #fulldict = setup_and_run()

    M_PYR1 = fulldict['M_PYR1']
    #M_PV1 = fulldict['M_PV1']
    #M_SOM1 = fulldict['M_SOM1']

    spikemon_PYR1 = fulldict['spikemon_PYR1']
    spikemon_PYR2 = fulldict['spikemon_PYR2']
    
    spikemon_PV1 = fulldict['spikemon_PV1']
    spikemon_SOM1 = fulldict['spikemon_SOM1']

    som_opto = fulldict['som_opto']
    pv_opto = fulldict['pv_opto']

    stim_arr1 = fulldict['stim_arr1']
    stim_dt = fulldict['stim_dt']

    T = fulldict['T']

    if choice == 'traces':
        fig = plt.figure()
        ax11 = fig.add_subplot(411)
        ax21 = fig.add_subplot(412)
        ax31 = fig.add_subplot(413)
        ax41 = fig.add_subplot(414)

        ax11.plot(M_PYR1.t/ms,M_PYR1.v[0]/mV,label='PYR')
        ax11.plot(M_PYR1.t/ms,M_PYR1.vD[0]/mV,label='Dend')
        ax11.plot(M_PYR1.t/ms,M_PYR1.w[0]/nA,label='Adap.')
        ax11.plot(M_PYR1.t/ms,M_PYR1.ges[0]/nS,label='SOM cond.')
        
        ax11.plot(M_PYR1.t/ms,M_PYR1.DSlow1[0],label='PYR1 Dep.')
        ax11.plot(M_PYR1.t/ms,M_PYR1.DFast1[0],label='PYR1 Dep. fast')
        ax11.plot(M_PYR1.t/ms,M_PYR1.IFast1[0],label='PYR1 I fast')

        #ax21.plot(M_PV1.t/ms,M_PV1.v[0]/mV,label='PV1')
        #ax21.plot(M_PV1.t/ms,M_PV1.DSlow1[0],label='PV1 DSlow')
        #ax21.plot(M_PV1.t/ms,M_PV1.DFast1[0],label='PV1 DFast')
        
        ax21.legend()

        #ax31.plot(M_PYR1.t/ms,M_PYR1.IFast1[0],label='ifast1')
        #ax31.plot(M_PYR1.t/ms,M_PYR1.DSlow1[0],label='dslow1')
        #ax31.plot(M_PYR1.t/ms,M_PYR1.DFast1[0],label='dfast1')
        
        ax31.plot(M_PYR1.t/ms,M_PYR1.thal1[0]/nA,label='thal1')
        ax31.plot(M_PYR1.t/ms,M_PYR1.thal2[0]/nA,label='thal2')
        ax31.plot(M_PYR1.t/ms,M_PYR1.thal3[0]/nA,label='thal3')


        ax41.plot(M_PYR1.t/ms,M_PYR1.DSlow3[0],label='PYR3 Dep.')
        ax41.plot(M_PYR1.t/ms,M_PYR1.DFast3[0],label='PYR3 Dep. fast')
        ax41.plot(M_PYR1.t/ms,M_PYR1.IFast3[0],label='PYR3 I fast')

        
        #ax31.plot(M_PYR1.t/ms,M_PYR1.IFast[0],label='I fast')

        #ax31.plot(M_SOM1.t/ms,M_SOM1.v[0]/mV,label='SOM')
        
        # get spike times of neuron 0
        idx1 = np.where(spikemon_PYR1.i==0)[0]
        idx2 = np.where(spikemon_PYR2.i==0)[0]
        
        ax11.scatter(spikemon_PYR1.t[idx1]/ms, spikemon_PYR1.i[idx1]+1,color='k')
        ax11.scatter(spikemon_PYR2.t[idx2]/ms, spikemon_PYR2.i[idx2]+1,color='k')

        ax31.legend()
        ax41.legend()
        ax11.legend()

    elif choice == 'rasters':

        # plot rasters
        fig2 = plt.figure()
        gs = gridspec.GridSpec(6,1)
        ax1 = plt.subplot(gs[:4,:])
        ax2 = plt.subplot(gs[4,:])
        ax3 = plt.subplot(gs[5,:])

        ax1.scatter(spikemon_PYR1.t/ms, spikemon_PYR1.i,color='k',s=.1)
        ax2.scatter(spikemon_PV1.t/ms, spikemon_PV1.i,color='k',s=.1)
        ax3.scatter(spikemon_SOM1.t/ms, spikemon_SOM1.i,color='k',s=.1)

        ax1.set_xlim(0,T/ms)
        ax2.set_xlim(0,T/ms)
        ax3.set_xlim(0,T/ms)

    elif choice == 'rasters_e':

        # plot rasters
        fig2 = plt.figure()
        gs = gridspec.GridSpec(6,1)
        ax1 = plt.subplot(gs[:4,:])
        ax2 = plt.subplot(gs[4,:])
        ax3 = plt.subplot(gs[5,:])

        ax1.scatter(spikemon_PYR1.t/ms, spikemon_PYR1.i,color='k',s=.1)
        ax2.scatter(spikemon_PYR2.t/ms, spikemon_PYR2.i,color='k',s=.1)
        
        #ax3.scatter(spikemon_SOM1.t/ms, spikemon_SOM1.i,color='k',s=.1)

        ax1.set_xlim(0,T/ms)
        ax2.set_xlim(0,T/ms)
        ax3.set_xlim(0,T/ms)

        
    elif choice == 'psth':

        # plot PSTH
        fig3 = plt.figure()
        gs = gridspec.GridSpec(3,1)
        ax31 = plt.subplot(gs[0,:])
        ax32 = plt.subplot(gs[1,:])
        ax33 = plt.subplot(gs[2,:])

        bin_factor = 1./2

        ax31.hist(spikemon_PYR1.t/ms, color='k',bins=int((T/ms)*bin_factor))
        ax32.hist(spikemon_PV1.t/ms, color='k',bins=int((T/ms)*bin_factor))
        ax33.hist(spikemon_SOM1.t/ms, color='k',bins=int((T/ms)*bin_factor))

        ax31.set_xlim(0,T/ms)
        ax32.set_xlim(0,T/ms)
        ax33.set_xlim(0,T/ms)

    elif choice == 'psth_e':

        # plot PSTH
        fig3 = plt.figure()
        gs = gridspec.GridSpec(3,1)
        ax31 = plt.subplot(gs[0,:])
        ax32 = plt.subplot(gs[1,:])
        ax33 = plt.subplot(gs[2,:])

        bin_factor = 1./2

        ax31.hist(spikemon_PYR1.t/ms, color='k',bins=int((T/ms)*bin_factor))
        ax31.hist(spikemon_PYR2.t/ms, color='k',bins=int((T/ms)*bin_factor))

        ax31.set_xlim(0,T/ms)
        ax32.set_xlim(0,T/ms)
        ax33.set_xlim(0,T/ms)


def main():

    paradigm = 'pv'
    fulldict1 = setup_and_run(paradigm=paradigm)

    if False:
        #plot(fulldict1,choice='rasters')
        plot(fulldict1,choice='traces')
        #plot(fulldict1,choice='psth')
        #plt.show()
    
    #fulldict2 = setup_and_run(pv_opto=.3)
    fulldict2 = setup_and_run(pv_opto=.2,paradigm=paradigm)#fulldict2 = setup_and_run(pv_opto=.3)
    fulldict3 = setup_and_run(som_opto=1.,paradigm=paradigm)

    if paradigm == 'ssa':
        FR_control_pyr1 = collect_FR(fulldict1['stim_arr1']+fulldict1['stim_arr3'],
                                     fulldict1['stim_dt'],
                                     fulldict1['defaultclock'],
                                     fulldict1['spikemon_PYR1'].t,
                                     fulldict1['n_pyr'])

        FR_pvoff_pyr1 = collect_FR(fulldict2['stim_arr1']+fulldict2['stim_arr3'],
                                   fulldict2['stim_dt'],
                                   fulldict2['defaultclock'],
                                   fulldict2['spikemon_PYR1'].t,
                                     fulldict2['n_pyr'])

        FR_somoff_pyr1 = collect_FR(fulldict3['stim_arr1']+fulldict3['stim_arr3'],
                                    fulldict3['stim_dt'],
                                    fulldict3['defaultclock'],
                                    fulldict3['spikemon_PYR1'].t,
                                     fulldict3['n_pyr'])

        FR_control_pyr2 = collect_FR(fulldict1['stim_arr1']+fulldict1['stim_arr3'],
                                     fulldict1['stim_dt'],
                                     fulldict1['defaultclock'],
                                     fulldict1['spikemon_PYR2'].t,
                                     fulldict1['n_pyr'])

        FR_pvoff_pyr2 = collect_FR(fulldict2['stim_arr1']+fulldict2['stim_arr3'],
                                   fulldict2['stim_dt'],
                                   fulldict2['defaultclock'],
                                   fulldict2['spikemon_PYR2'].t,
                                     fulldict2['n_pyr'])

        FR_somoff_pyr2 = collect_FR(fulldict3['stim_arr1']+fulldict3['stim_arr3'],
                                    fulldict3['stim_dt'],
                                    fulldict3['defaultclock'],
                                    fulldict3['spikemon_PYR2'].t,
                                     fulldict3['n_pyr'])

        adapted_fr = 1#FR_control_pyr1[-1]


        if False:
            
            fig = plt.figure()
            gs = gridspec.GridSpec(3,3)

            ax11 = plt.subplot(gs[0,0]) # psth control
            ax21 = plt.subplot(gs[1,0]) # psth pvoff
            ax31 = plt.subplot(gs[2,0]) # psth somoff

            ax12 = plt.subplot(gs[0,1]) # plot FR
            ax22 = plt.subplot(gs[1,1]) # plot FR

            ax13 = plt.subplot(gs[:,2]) # plot FR diff.


            bin_factor = 1./2

            ax11.hist(fulldict1['spikemon_PYR1'].t/ms, color='k',bins=int((fulldict1['T']/ms)*bin_factor))
            ax21.hist(fulldict2['spikemon_PYR1'].t/ms, color='k',bins=int((fulldict2['T']/ms)*bin_factor))
            ax31.hist(fulldict3['spikemon_PYR1'].t/ms, color='k',bins=int((fulldict3['T']/ms)*bin_factor))

            ax11.set_xlim(500,fulldict1['T']/ms)
            ax21.set_xlim(500,fulldict2['T']/ms)
            ax31.set_xlim(500,fulldict3['T']/ms)

            tone_number1 = np.arange(len(np.where((fulldict1['stim_arr1']+fulldict1['stim_arr3'])!=0)[0]))

            bar_width = 0.2
            ax12.set_title('Mean FR')
            ax12.bar(tone_number1,FR_control_pyr1/adapted_fr,width=bar_width,label='control1',color='blue')
            ax12.bar(tone_number1+bar_width,FR_pvoff_pyr1/adapted_fr,width=bar_width,label='pv_off1',color='green')
            ax12.bar(tone_number1+2*bar_width,FR_somoff_pyr1/adapted_fr,width=bar_width,label='som_off1',color='red')

            ax12.plot([0,4],[1,1],ls='--',color='gray')
            ax22.plot([0,4],[1,1],ls='--',color='gray')

            ax13.set_title('Diff from Control')
            ax13.bar(tone_number1,np.abs(FR_control_pyr1-FR_pvoff_pyr1)/adapted_fr,
                     width=bar_width,label='control-pv_off',color='green')
            ax13.bar(tone_number1+bar_width,np.abs(FR_control_pyr1-FR_somoff_pyr1)/adapted_fr,
                     width=bar_width,label='control-som_off',color='red')

        if False:
            fig = plt.figure()
            gs = gridspec.GridSpec(3,3)

            ax11 = plt.subplot(gs[0,0]) # psth control
            ax21 = plt.subplot(gs[1,0]) # psth pvoff
            ax31 = plt.subplot(gs[2,0]) # psth somoff

            ax12 = plt.subplot(gs[0,1]) # plot FR
            ax22 = plt.subplot(gs[1,1]) # plot FR

            ax13 = plt.subplot(gs[:,2]) # plot FR diff.


            bin_factor = 1./2

            ax11.hist(fulldict1['spikemon_PYR2'].t/ms, color='k',bins=int((fulldict1['T']/ms)*bin_factor))
            ax21.hist(fulldict2['spikemon_PYR2'].t/ms, color='k',bins=int((fulldict2['T']/ms)*bin_factor))
            ax31.hist(fulldict3['spikemon_PYR2'].t/ms, color='k',bins=int((fulldict3['T']/ms)*bin_factor))

            ax11.set_xlim(500,fulldict1['T']/ms)
            ax21.set_xlim(500,fulldict2['T']/ms)
            ax31.set_xlim(500,fulldict3['T']/ms)

            tone_number1 = np.arange(len(np.where((fulldict1['stim_arr1']+fulldict1['stim_arr3'])!=0)[0]))

            bar_width = 0.2
            ax12.set_title('Mean FR')
            ax12.bar(tone_number1,FR_control_pyr2/adapted_fr,width=bar_width,label='control1',color='blue')
            ax12.bar(tone_number1+bar_width,FR_pvoff_pyr2/adapted_fr,width=bar_width,label='pv_off1',color='green')
            ax12.bar(tone_number1+2*bar_width,FR_somoff_pyr2/adapted_fr,width=bar_width,label='som_off1',color='red')

            ax12.plot([0,4],[1,1],ls='--',color='gray')
            ax22.plot([0,4],[1,1],ls='--',color='gray')

            ax13.set_title('Diff from Control')
            ax13.bar(tone_number1,np.abs(FR_control_pyr2-FR_pvoff_pyr2)/adapted_fr,
                     width=bar_width,label='control-pv_off',color='green')
            ax13.bar(tone_number1+bar_width,np.abs(FR_control_pyr2-FR_somoff_pyr2)/adapted_fr,
                     width=bar_width,label='control-som_off',color='red')

        if True:
            fig = plt.figure()
            gs = gridspec.GridSpec(1,2)

            ax11 = plt.subplot(gs[0,0]) # psth control
            ax12 = plt.subplot(gs[0,1]) # plot FR

            bin_factor = 1./2
            

            tone_number1 = np.arange(len(np.where((fulldict1['stim_arr1']+fulldict1['stim_arr3'])!=0)[0]))

            bar_width = 0.2
            ax11.set_title('Mean FR')
            ax11.bar(tone_number1[4:],FR_control_pyr2[4:]/adapted_fr,width=bar_width,label='control1',color='blue')
            ax11.bar(tone_number1[4:]+bar_width,FR_pvoff_pyr2[4:]/adapted_fr,width=bar_width,label='pv_off1',color='green')
            ax11.bar(tone_number1[4:]+2*bar_width,FR_somoff_pyr2[4:]/adapted_fr,width=bar_width,label='som_off1',color='red')

            #ax12.plot([0,4],[1,1],ls='--',color='gray')
            #ax22.plot([0,4],[1,1],ls='--',color='gray')

            ax12.set_title('Diff from Control')
            ax12.bar(tone_number1[4:],np.abs(FR_control_pyr2-FR_pvoff_pyr2)[4:]/adapted_fr,
                     width=bar_width,label='control-pv_off',color='green')
            ax12.bar(tone_number1[4:]+bar_width,np.abs(FR_control_pyr2-FR_somoff_pyr2)[4:]/adapted_fr,
                     width=bar_width,label='control-som_off',color='red')


    elif (paradigm == 'fs1') or (paradigm == 'fs2') or (paradigm == 'fs3'):

        suffix = paradigm[-1]
        
        fig = plt.figure()
        gs = gridspec.GridSpec(3,1)

        ax11 = plt.subplot(gs[0,0]) # psth control
        ax21 = plt.subplot(gs[1,0]) # psth pvoff
        ax31 = plt.subplot(gs[2,0]) # psth somoff

        bin_factor = 1./2

        ax11.hist(fulldict1['spikemon_PYR'+suffix].t/ms, color='k',bins=int((fulldict1['T']/ms)*bin_factor))
        ax21.hist(fulldict2['spikemon_PYR'+suffix].t/ms, color='k',bins=int((fulldict2['T']/ms)*bin_factor))
        ax31.hist(fulldict3['spikemon_PYR'+suffix].t/ms, color='k',bins=int((fulldict3['T']/ms)*bin_factor))

        ax11.set_title('Pyr Activity (Control)')
        ax21.set_title('PV Activity (PV Off)')
        ax31.set_title('SOM Activity (SOM Off)')

        start_time1 = 160
        start_time2 = 230
        interval_time = 50

        n = fulldict['n_pyr']

        # 50-60...90-100,  start_time2-130...160-170    
        control_pa = get_FR(start_time1,start_time1+interval_time,fulldict1['defaultclock'],fulldict1['spikemon_PYR'+suffix].t,n)
        control_fs = get_FR(start_time2,start_time2+interval_time,fulldict1['defaultclock'],fulldict1['spikemon_PYR'+suffix].t,n)

        pv_pa = get_FR(start_time1,start_time1+interval_time,fulldict2['defaultclock'],fulldict2['spikemon_PYR'+suffix].t,n)
        pv_fs = get_FR(start_time2,start_time2+interval_time,fulldict2['defaultclock'],fulldict2['spikemon_PYR'+suffix].t,n)

        som_pa = get_FR(start_time1,start_time1+interval_time,fulldict3['defaultclock'],fulldict3['spikemon_PYR'+suffix].t,n)
        som_fs = get_FR(start_time2,start_time2+interval_time,fulldict3['defaultclock'],fulldict3['spikemon_PYR'+suffix].t,n)

        print('control:',control_fs/control_pa)
        print('pv off:',pv_fs/pv_pa)
        print('som off:',som_fs/som_pa)

    
    plt.show()
    
if __name__ == "__main__":
    main()
