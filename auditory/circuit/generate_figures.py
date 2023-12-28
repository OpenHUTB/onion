import rate_models.cols3_ssa as c3_ssa
import rate_models.cols3_fs as c3_fs
import rate_models.natan2015_simple_linear as c1_ssa
import spiking_models.col3_spiking as s3
import rate_models.natan2015_simple_linear as natan

import spiking_models.col1_ssa_spiking as s1_ssa
import spiking_models.col1_fs_spiking as s1_fs

from rate_models.xppcall import xpprun,read_pars_values_from_file,read_init_values_from_file

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.interpolate import interp1d
from matplotlib import rc

import os
import scipy.signal as sps
import matplotlib
import brian2 as b2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


matplotlib.rcParams['hatch.linewidth'] = 0.5
hatch=r'//'

#matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath \usepackage{bm} \usepackage{xcolor} \setlength{\parindent}{0pt}']
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
matplotlib.rcParams.update({'figure.autolayout':True})

rc('text',usetex=True)
#rc('font',family='serif',serif=['Computer Modern Roman'])
rc('font',family='serif',serif=['Times New Roman'])

som_color = '#66c2a5'
pv_color = '#fc8d62'



def r1_responses():
    # tone-evoked responses of the rate model

    
    fname = 'rate_models/xpp/natan2015_simple_linear.ode'
    pars = read_pars_values_from_file(fname)
    inits = read_init_values_from_file(fname)
    
    control = c1_ssa.run_experiment(
        fname,
        pars,inits,
        return_all=True)

    fig = plt.figure(figsize=(5,3))
    gs = gridspec.GridSpec(3,2)
    ax1 = plt.subplot(gs[:2,:2])
    ax2 = plt.subplot(gs[2,:2])

    gs.update(hspace=.5)

    control['t'] = control['t']*10
    
    idx_range = (control['t']<400)*(control['t']>270)

    xmin = control['t'][idx_range][0]
    xmax = control['t'][idx_range][-1]
    
    ax1.plot(control['t'][idx_range],control['v2'][idx_range],color=som_color,label='SOM',lw=1.5)
    
    ax1.plot(control['t'][idx_range],control['g'][idx_range],color='gray',label='g',lw=1.5)
    ax1.plot(control['t'][idx_range],control['u'][idx_range],color='k',lw=2,label='Pyr')
    ax1.plot(control['t'][idx_range],control['v1'][idx_range],color=pv_color,label='PV',lw=1.5,dashes=(5,2))

    ax2.plot(control['t'][idx_range],control['ia'][idx_range],label='Thalamus',color='tab:red')

    ax1.set_ylabel('Neural Activity')

    ax2.set_ylabel('Input')
    ax2.set_xlabel('Time (ms)')

    ax1.legend()
    ax2.legend()
    
    ax1.set_xticks([])
    ax1.set_xlim(xmin,xmax)
    ax2.set_xlim(xmin,xmax)

    
    #ax1.set_title(r'\textbf{A}',loc='left')
    #ax2.set_title(r'\textbf{B}',loc='left')

    ax1.set_title(r'\textbf{A}',x=0,y=.96)
    ax2.set_title(r'\textbf{B}',x=0,y=.925)
    
    return fig



def run_r3_responses(fname,pars,inits,return_all=False):
    
    npa, vn = xpprun(fname,
                     xppname='xppaut',
                     inits=inits,
                     parameters=pars,
                     clean_after=True)

    t = npa[:,0]
    sv = npa[:,1:]

    total_time = t[-1]

    u1 = sv[:,vn.index('u1')]
    u2 = sv[:,vn.index('u2')]
    u3 = sv[:,vn.index('u3')]
    
    p1 = sv[:,vn.index('p1')]
    p2 = sv[:,vn.index('p2')]
    p3 = sv[:,vn.index('p3')]
    
    s1 = sv[:,vn.index('s1')]
    s2 = sv[:,vn.index('s2')]
    s3 = sv[:,vn.index('s3')]
    
    i1a = sv[:,vn.index('i1a')]
    i2a = sv[:,vn.index('i2a')]
    i3a = sv[:,vn.index('i3a')]
    
    g1 = sv[:,vn.index('g1')]
    g2 = sv[:,vn.index('g2')]
    g3 = sv[:,vn.index('g3')]

    if return_all:

        # implement parameter return dict.
        return {'t':t,
                'u1':u1,'p1':p1,'s1':s1,
                'u2':u2,'p2':p2,'s2':s2,
                'u3':u3,'p3':p3,'s3':s3,
                'inits':inits,'parameters':pars,'sv':sv,'vn':vn,
                'i1a':i1a,'i2a':i2a,'i3a':i3a,
                'g1':g1,'g2':g2,'g3':g3}

    else:
        return {'t':t,'u':u,'v1':v1,'v2':v2}


def r3_responses():
    # tone-evoked responses of the 3-unit rate model
    
    fname = 'rate_models/xpp/cols3_responses.ode'
    pars = read_pars_values_from_file(fname)
    inits = read_init_values_from_file(fname)
    
    control = run_r3_responses(
        fname,
        pars,inits,
        return_all=True)

    fig = plt.figure(figsize=(5,5))
    gs = gridspec.GridSpec(7,2)
    ax1 = plt.subplot(gs[:2,:2])
    ax2 = plt.subplot(gs[2:4,:2])
    ax3 = plt.subplot(gs[4:6,:2])
    ax4 = plt.subplot(gs[6:7,:2])

    gs.update(hspace=.8)

    control['t'] = control['t']*10
    
    idx_range = (control['t']<1250)*(control['t']>0)

    xmin = control['t'][idx_range][0]
    xmax = control['t'][idx_range][-1]
    
    #ax1.plot(control['t'][idx_range],control['v2'][idx_range],color=som_color,label='SOM',lw=1.5)

    ax1.plot(control['t'][idx_range],control['u1'][idx_range],color='k',lw=2,label='Pyr')
    ax1b = ax1.twinx()
    ax1b.plot(control['t'][idx_range],control['g1'][idx_range],color='gray',label='g_1',lw=1.5)
    ax1b.set_ylim(0.5,1.1)
    
    ax2.plot(control['t'][idx_range],control['u2'][idx_range],color='k',lw=2,label='Pyr')    
    ax2b = ax2.twinx()
    ax2b.plot(control['t'][idx_range],control['g2'][idx_range],color='gray',label='g_2',lw=1.5)
    ax2b.set_ylim(0.5,1.1)    

    ax3.plot(control['t'][idx_range],control['u3'][idx_range],color='k',lw=2,label='Pyr')
    ax3b = ax3.twinx()
    ax3b.plot(control['t'][idx_range],control['g3'][idx_range],color='gray',label='g_3',lw=1.5)
    ax3b.set_ylim(0.5,1.1)
    
    #ax1.plot(control['t'][idx_range],control['v1'][idx_range],color=pv_color,label='PV',lw=1.5,dashes=(5,2))

    ax4.plot(control['t'][idx_range],control['i1a'][idx_range],label='Thalamus',color='gray')
    ax4.plot(control['t'][idx_range],control['i2a'][idx_range],label='Thalamus',color='black')
    ax4.plot(control['t'][idx_range],control['i3a'][idx_range],label='Thalamus',color='red')

    ax4.text(150,0.5,r'$f_1$',color='gray')
    ax4.text(550,0.5,r'$f*$',color='black')
    ax4.text(950,0.5,r'$f_2$',color='red')

    ax1.set_ylabel(r'$u_1$ Activity')
    ax2.set_ylabel(r'$u_2$ Activity')
    ax3.set_ylabel(r'$u_3$ Activity')

    ax1b.set_ylabel(r'$g_1$',color='gray')
    ax2b.set_ylabel(r'$g_2$',color='gray')
    ax3b.set_ylabel(r'$g_3$',color='gray')
    
    ax4.set_ylabel('Input')
    ax4.set_xlabel('Time (ms)')

    #ax1.legend()
    #ax1b.legend(loc='lower right')
    #ax2b.legend(loc='lower right')
    #ax3b.legend(loc='lower right')
    #ax2.legend()

    ax1b.tick_params(axis='y',labelcolor='gray')
    ax2b.tick_params(axis='y',labelcolor='gray')
    ax3b.tick_params(axis='y',labelcolor='gray')
    
    ax1.set_xticks([])
    ax2.set_xticks([])
    ax3.set_xticks([])

    ax1.set_xlim(xmin,xmax)
    ax2.set_xlim(xmin,xmax)
    ax3.set_xlim(xmin,xmax)
    ax4.set_xlim(xmin,xmax)


    ax1.set_title(r'\textbf{A}',x=0,y=.925)
    ax2.set_title(r'\textbf{B}',x=0,y=.925)
    ax3.set_title(r'\textbf{C}',x=0,y=.925)
    ax4.set_title(r'\textbf{D}',x=0,y=.925)
    
    #plt.tight_layout()
    
    return fig


def s1_responses():

    fulldict1 = s1_fs.setup_and_run(single_tone=True,q=.2)
    
    stim_arr = fulldict1['stim_arr']

    M_PYR = fulldict1['M_PYR']
    M_PV = fulldict1['M_PV']
    M_SOM = fulldict1['M_SOM']

    spikemon_PYR = fulldict1['spikemon_PYR']
    spikemon_PV = fulldict1['spikemon_PV']
    spikemon_SOM = fulldict1['spikemon_SOM']

    som_opto = fulldict1['som_opto']
    pv_opto = fulldict1['pv_opto']

    stim_arr = fulldict1['stim_arr']
    stim_dt = fulldict1['stim_dt']
    T = fulldict1['T']
    
    fig = plt.figure(figsize=(5,6))
    gs = gridspec.GridSpec(4,2)
    ax11 = plt.subplot(gs[0,0])
    ax21 = plt.subplot(gs[1,0])
    ax31 = plt.subplot(gs[2,0])

    ax12 = plt.subplot(gs[0,1])
    ax22 = plt.subplot(gs[1,1])
    ax32 = plt.subplot(gs[2,1])

    ax41 = plt.subplot(gs[3,:])

    start_time=100;end_time=250
    
    time_idx_PYR = (spikemon_PYR.t/b2.ms > start_time)*(spikemon_PYR.t/b2.ms < end_time)
    time_idx_PV = (spikemon_PV.t/b2.ms > start_time)*(spikemon_PV.t/b2.ms < end_time)
    time_idx_SOM = (spikemon_SOM.t/b2.ms > start_time)*(spikemon_SOM.t/b2.ms < end_time)
    
    ax11.scatter((spikemon_PYR.t/b2.ms)[time_idx_PYR], spikemon_PYR.i[time_idx_PYR],color='k',s=.1)
    ax21.scatter((spikemon_PV.t/b2.ms)[time_idx_PV], spikemon_PV.i[time_idx_PV],color=pv_color,s=.1)
    ax31.scatter((spikemon_SOM.t/b2.ms)[time_idx_SOM], spikemon_SOM.i[time_idx_SOM],color=som_color,s=.1)

    
    bin_factor = 1/4.
    
    nPyr,xPyr,_ = ax12.hist((spikemon_PYR.t/b2.ms)[time_idx_PYR],color='k',bins=int((T/b2.ms)*bin_factor))
    nPV,xPV,_ = ax22.hist((spikemon_PV.t/b2.ms)[time_idx_PV], color=pv_color,bins=int((T/b2.ms)*bin_factor))
    nSOM,xSOM,_ = ax32.hist((spikemon_SOM.t/b2.ms)[time_idx_SOM],color=som_color,bins=int((T/b2.ms)*bin_factor))

    ax41.plot(M_PYR.t/b2.ms,fulldict1['q']*M_PYR.IFast[0]*M_PYR.DSlow[0]*M_PYR.DFast[0],color='tab:red')
        
    #sp.signal.fftconvolve(npilSubTraces[i,:],sps.hanning(windowlen)/sps.hanning(windowlen).sum(),mode='same')
    
    ax11.set_xlim(start_time,end_time)
    ax21.set_xlim(start_time,end_time)
    ax31.set_xlim(start_time,end_time)

    ax12.set_xlim(start_time,end_time)
    ax22.set_xlim(start_time,end_time)
    ax32.set_xlim(start_time,end_time)

    ax41.set_xlim(start_time,end_time)

    ax11.set_ylim(0,np.amax(spikemon_PYR.i))
    ax21.set_ylim(0,np.amax(spikemon_PV.i))
    ax31.set_ylim(0,np.amax(spikemon_SOM.i))

    
    ax11.set_title(r'\textbf{A} \quad\quad Raster Plot',y=1,x=.35)
    ax12.set_title(r'\textbf{B} \quad\quad Histogram',y=1,x=.33)

    ax21.set_title(r'\textbf{C}',y=1,x=0)
    ax22.set_title(r'\textbf{D}',y=1,x=0)

    ax31.set_title(r'\textbf{E}',y=1,x=0)
    ax32.set_title(r'\textbf{F}',y=1,x=0)
    
    ax41.set_title(r'\textbf{G}',y=1,x=0)


    #ax31.set_xlabel('Time (ms)')
    #ax32.set_xlabel('Time (ms)')

    ax41.set_xlabel('Time (ms)')
    
    ax11.set_ylabel('Pyr Neuron Index')
    ax21.set_ylabel('PV Neuron Index')
    ax31.set_ylabel('SOM Neuron Index')

    ax12.set_ylabel('Pyr Spike Count')
    ax22.set_ylabel('PV Spike Count')
    ax32.set_ylabel('SOM Spike Count')

    ax41.set_ylabel('Thalamus (nA)')

    ax11.set_yticks([0,800,1600])
    ax21.set_yticks([0,100,200])
    ax31.set_yticks([0,100,200])
    

    #fig = s1_fs.plot(fulldict1,choice='rasters')

    return fig


def r3_s3_pv_full(recompute=True):

    ######################################### rate model

    fname = 'rate_models/xpp/cols3_pv.ode'
    pars = read_pars_values_from_file(fname)
    inits = read_init_values_from_file(fname)
    
    # run PV activation for correlation calculation.
    pars['q']=5#1.2
    pars['pv_opto']=-2.
    pars['som_opto']=0.
    pars['mode']=2
    pars['dt']=0.01
    
    pv_on = c3_fs.run_experiment(
        fname,
        pars,inits,
        return_all=True)

    pars['pv_opto']=2
    
    pv_off = c3_fs.run_experiment(
        fname,
        pars,inits,
        return_all=True)

    # run PV activation for correlation calculation.
    pars['pv_opto']=0
    pars['som_opto']=0.
    pars['mode']=2
    
    pv_control = c3_fs.run_experiment(
        fname,
        pars,inits,
        return_all=True)


    
    time = pv_on['t']
    input_trace = pv_on['sv'][:,pv_on['vn'].index('ia2')]

    time_short = time[time<20]
    input_trace_short = input_trace[time<20]

    time_ctrl = pv_control['t']
    input_trace_ctrl = pv_control['sv'][:,pv_control['vn'].index('ia2')]

    time_short_ctrl = time_ctrl[time_ctrl<20]
    input_trace_short_ctrl = input_trace_ctrl[time_ctrl<20]
    
    print "PV control corr = "+str(pearsonr(input_trace_short_ctrl,pv_control['u2'][time_ctrl<20]))
    print "PV act. corr = "+str(pearsonr(input_trace_short,pv_on['u2'][time<20]))
    print "PV inact. corr = "+str(pearsonr(input_trace_short,pv_off['u2'][time<20]))

    ######################################### spiking model
    
    paradigm = 'pv'
    seedlist = np.arange(0,20,1)
    #seedlist = np.arange(0,10,1)
    #seedlist = np.arange(0,2,1)

    # check if saved results exist
    fname_ctrl = "dat/pv_ctrl_seedlen="+str(len(seedlist))+".dat"
    fname_pv = "dat/pv_on_seedlen="+str(len(seedlist))+".dat"
    fname_pv_on = "dat/pv_off_seedlen="+str(len(seedlist))+".dat"

    fname_corr_ctrl = "dat/pv_ctrl_corr_seedlen="+str(len(seedlist))+".dat"
    fname_corr_pv = "dat/pv_corr_seedlen="+str(len(seedlist))+".dat"
    fname_corr_pv_off = "dat/pv_on_corr_seedlen="+str(len(seedlist))+".dat"

    fname_thal = "dat/pv_thal_seedlen="+str(len(seedlist))+".dat"

    if os.path.isfile(fname_ctrl) and os.path.isfile(fname_pv) and os.path.isfile(fname_pv_on) and \
       os.path.isfile(fname_corr_ctrl) and os.path.isfile(fname_corr_pv) and os.path.isfile(fname_corr_pv_off) and \
       os.path.isfile(fname_thal) and \
       not(recompute):
        print recompute,not(recompute)
        sol_ctrl_arr = np.loadtxt(fname_ctrl)
        sol_pv_arr = np.loadtxt(fname_pv)
        sol_pv_off = np.loadtxt(fname_pv_on)

        corr_ctrl = np.loadtxt(fname_corr_ctrl)
        corr_pv = np.loadtxt(fname_corr_pv)
        corr_pv_off = np.loadtxt(fname_corr_pv_off)

        thal_arr = np.loadtxt(fname_thal)

        xx = np.linspace(thal_arr[:,0][0],thal_arr[:,0][-1],250)

    else:
        # (seeds, time array)
        sol_ctrl_arr = np.zeros((len(seedlist),250))
        sol_pv_arr = np.zeros((len(seedlist),250))
        sol_pv_off = np.zeros((len(seedlist),250))

        corr_ctrl = np.zeros(len(seedlist))
        corr_pv = np.zeros(len(seedlist))
        corr_pv_off = np.zeros(len(seedlist))

        i = 0
        for seed in seedlist:
            fulldict1 = s3.setup_and_run(paradigm=paradigm,seed=seed,dt=0.1) # control
            #fulldict1 = s3.setup_and_run(pv_opto=-.2,paradigm=paradigm) # PV act
            fulldict2 = s3.setup_and_run(pv_opto=-.1,paradigm=paradigm,seed=seed,dt=0.1) # PV act
            fulldict2Off = s3.setup_and_run(pv_opto=.5,paradigm=paradigm,seed=seed,dt=0.1) # PV act

            # generate PSTH from raster
            delta = .5*b2.ms
            fulldict1['spikemon_PYR2'].t

            time = 0*b2.ms

            psth1 = np.zeros(int(fulldict1['T']/fulldict1['defaultclock'])+1)
            psth2 = np.zeros(int(fulldict2['T']/fulldict2['defaultclock'])+1)
            psth2Off = np.zeros(int(fulldict2Off['T']/fulldict2Off['defaultclock'])+1)

            counter = 0
            while time < fulldict1['T']:

                spike_idx1 = (fulldict1['spikemon_PYR2'].t > time)*(fulldict1['spikemon_PYR2'].t <= time+delta)
                spike_idx2 = (fulldict2['spikemon_PYR2'].t > time)*(fulldict2['spikemon_PYR2'].t <= time+delta)
                spike_idx3 = (fulldict2Off['spikemon_PYR2'].t > time)*(fulldict2Off['spikemon_PYR2'].t <= time+delta)

                psth1[counter] = np.sum(fulldict1['spikemon_PYR2'].t[spike_idx1])/fulldict1['n_pyr']
                psth2[counter] = np.sum(fulldict2['spikemon_PYR2'].t[spike_idx2])/fulldict2['n_pyr']
                psth2Off[counter] = np.sum(fulldict2Off['spikemon_PYR2'].t[spike_idx3])/fulldict2Off['n_pyr']

                counter += 1
                time += fulldict1['defaultclock']

            psth1 /= delta
            psth2 /= delta
            psth2Off /= delta

            x1 = np.linspace(0,fulldict1['T']/b2.ms,len(psth1))
            x2 = np.linspace(0,fulldict2['T']/b2.ms,len(psth2))
            x2Off = np.linspace(0,fulldict2Off['T']/b2.ms,len(psth2Off))
            x3 = np.linspace(0,fulldict2['T']/b2.ms,len(fulldict1['M_PYR2'].thal2[0]))

            def thal(x):
                return interp1d(x3,fulldict1['M_PYR2'].thal2[0]/b2.nA)(x)

            def psth_control(x):
                x = np.mod(x,x1[-1])
                return interp1d(x1,psth1)(x)

            def psth_pv(x):
                x = np.mod(x,x2[-1])
                return interp1d(x2,psth2)(x)

            def psth_pv_off(x):
                x = np.mod(x,x2Off[-1])
                return interp1d(x2Off,psth2Off)(x)

            #xx = np.linspace(0,fulldict1['T']/b2.ms,len(psth1))
            xx = np.linspace(0,fulldict1['T']/b2.ms,250)

            #print len(psth1)

            # determine time delay
            # get all upwards crossings of inhibited rate
            th_ctrl = .01
            th_pv = .1

            ctrl_arr = psth_control(xx)
            pv_arr = psth_pv(xx)
            pv_arr_off = psth_pv_off(xx)

            # mark indices where there is an upward crossing in pv activation case
            cross_idx_ctrl = (ctrl_arr[1:]>=th_ctrl)*(ctrl_arr[:-1]<th_ctrl)
            cross_idx_pv = (pv_arr[1:]>=th_pv)*(pv_arr[:-1]<th_pv)
            cross_idx_pv_off = (pv_arr_off[1:]>=th_pv)*(pv_arr_off[:-1]<th_pv)

            # mark first time where the upward crossing happens
            crossing_t_ctrl = xx[1:][cross_idx_ctrl]
            first_crossing_t_pv = xx[1:][cross_idx_pv]
            first_crossing_t_pv_off = xx[1:][cross_idx_pv_off]
            #print first_crossing_t_pv,crossing_t_ctrl

            # find difference between tone onset and first crossing time
            stim_start_times = np.where(fulldict1['stim_arr2']!=0)[0]*fulldict1['stim_dt'] # stim start time

            diff_t_pv = (first_crossing_t_pv - stim_start_times[0])
            diff_t_pv = diff_t_pv[np.argmin(np.abs(diff_t_pv))]
            #diff_t_pv = 0.

            print 'PV response delay',diff_t_pv
            sol_ctrl_arr[i,:] = psth_control(xx)
            sol_pv_arr[i,:] = psth_pv(xx+diff_t_pv)
            sol_pv_off[i,:] = psth_pv_off(xx)

            corr_ctrl[i] = pearsonr(thal(xx),psth_control(xx))[0]
            corr_pv[i] = pearsonr(thal(xx),psth_pv(xx+diff_t_pv))[0]
            corr_pv_off[i] = pearsonr(thal(xx),psth_pv_off(xx))[0]
            #print 'correlation thal-control, thal-pv act', pearsonr(thal(xx),psth_control(xx))[0],pearsonr(thal(xx),psth_pv(xx+diff_t_pv))[0]

            i += 1

            
        thal_arr = np.zeros((len(x3),2))
        thal_arr[:,0] = x3
        thal_arr[:,1] = thal(x3)
        
        np.savetxt(fname_thal,thal_arr)
        
        np.savetxt(fname_ctrl,sol_ctrl_arr)
        np.savetxt(fname_pv,sol_pv_arr)
        np.savetxt(fname_pv_on,sol_pv_off)

        np.savetxt(fname_corr_ctrl,corr_ctrl)
        np.savetxt(fname_corr_pv,corr_pv)
        np.savetxt(fname_corr_pv_off,corr_pv_off)
        
    ctrl_mean = np.mean(sol_ctrl_arr,axis=0)
    ctrl_dev = np.std(sol_ctrl_arr,axis=0)

    pv_mean = np.mean(sol_pv_arr,axis=0)
    pv_dev = np.std(sol_pv_arr,axis=0)

    pv_mean_off = np.mean(sol_pv_off,axis=0)
    pv_dev_off = np.std(sol_pv_off,axis=0)

    corr_ctrl_mean = np.mean(corr_ctrl)
    corr_ctrl_std = np.std(corr_ctrl)

    corr_pv_mean = np.mean(corr_pv)
    corr_pv_std = np.std(corr_pv)
    
    corr_pv_mean_off = np.mean(corr_pv_off)
    corr_pv_std_off = np.std(corr_pv_off)

    print 'corr ctrl mean', corr_ctrl_mean,corr_ctrl_std
    print 'corr pv mean', corr_pv_mean,corr_pv_std
    print 'corr pv off mean', corr_pv_mean_off,corr_pv_std_off


    #fig = plt.figure(figsize=(8,3))

    fig = plt.figure(figsize=(8,4))
    gs = gridspec.GridSpec(3,3)
    ax11 = plt.subplot(gs[:2,0])
    ax12 = plt.subplot(gs[:2,1])
    ax13 = plt.subplot(gs[:2,2])
    
    ax31 = plt.subplot(gs[-1,0])
    ax32 = plt.subplot(gs[-1,1])
    ax33 = plt.subplot(gs[-1,2])    
    
    ######################################### rate plots

    ax31.plot(time_short_ctrl*10,input_trace_short_ctrl,color='tab:red',zorder=0)
    ax11.plot(time_short_ctrl*10,pv_control['u2'][time_ctrl<20],color='k',lw=3,label='Control',zorder=3)    
    ax11.plot(time_short*10,pv_on['u2'][pv_on['t']<20],color=pv_color,lw=3,label='PV Act.',ls='--',zorder=4)
    
    ax33.plot(time_short_ctrl*10,input_trace_short_ctrl,color='tab:red')
    ax13.plot(time_short_ctrl*10,pv_control['u2'][time_ctrl<20],color='k',lw=2,label='Control')
    ax13.plot(time_short*10,pv_off['u2'][pv_off['t']<20],color=pv_color,lw=3,label='PV Inact.',ls='--',zorder=3)    
    
    ######################################### spike plots    
    
    ax12.plot(xx-40,ctrl_mean,label='Control',color='k',lw=2)
    ax12.fill_between(xx-40,ctrl_mean-ctrl_dev,ctrl_mean+ctrl_dev,facecolor='k',alpha=.25)
    
    ax12.plot(xx-40,pv_mean,label='PV Act.',color=pv_color,lw=3,ls='--',zorder=3)
    ax12.fill_between(xx-40,pv_mean-pv_dev,pv_mean+pv_dev,facecolor=pv_color,alpha=.25,zorder=3)

    ax32.plot(thal_arr[:,0]-40,thal_arr[:,1],color='tab:red',label='Thalamus')

    
    ax11.set_ylabel('Firing Rate')
    ax31.set_ylabel('Thalamus')
        
    ax11.set_title(r'\textbf{A} Rate Model',x=0.3)
    ax12.set_title(r'\textbf{B} Spiking Model',x=0.3)
    ax13.set_title(r'\textbf{C} Rate Model (Prediction)',x=0.5)

    ax12.set_xlim(0,xx[-1])

    ax11.set_xlim(80,150)
    #ax12.set_xlim(120,190)
    ax12.set_xlim(80,150)
    ax13.set_xlim(80,150)

    ax31.set_xlim(80,150)
    ax32.set_xlim(80,150)
    ax33.set_xlim(80,150)

    
    #ax22.set_xlim(120,190)
    
    #ax3.set_xlabel('Time (ms)')

    ax11.set_ylabel('Pyr Firing Rate')
    #ax13.set_ylabel('Pyr Firing Rate')
    #ax11b.set_ylabel('Thalamus',color='tab:red')
    #ax12.set_ylabel('Pyr Firing Rate')
    #ax12b.set_ylabel('Thalamus',color='tab:red')
    #ax22b.set_ylabel('Thalamus',color='tab:red')
    

    ax31.set_xlabel('Time (ms)')
    ax32.set_xlabel('Time (ms)')
    ax33.set_xlabel('Time (ms)')    
    #ax2.set_ylabel('Pyr Neuron Index')
    #ax3.set_ylabel('Pyr Neuron Index')    

    ax11.legend()
    ax13.legend()
    ax12.legend()
    #ax22.legend()
    
    return fig



def r1_adaptation():
    """
    adaptation in a single rate model
    """

    fname = 'rate_models/xpp/natan2015_simple_linear.ode'
    pars = read_pars_values_from_file(fname)
    inits = read_init_values_from_file(fname)

    # returns tuple (t,u,v1,v2,inits,parameters,tonelist)    
    control = c1_ssa.run_experiment(
        fname,
        pars,inits,
        return_all=True)

    pars['pv_offall']=2
    
    pv_off = c1_ssa.run_experiment(
        fname,
        pars,inits,
        return_all=True)
    
    pars['pv_offall']=0
    pars['som_offall']=1
    
    som_off = c1_ssa.run_experiment(
        fname,
        pars,inits,
        return_all=True)

    
    maxes_u_control,maxes_v1_control,maxes_v2_control = natan.get_tone_evoked_FR(
        control['t'],
        control['u'],
        control['v1'],
        control['v2'],
        control['tonelist'])

    maxes_u_pv_off,maxes_v1_pv_off,maxes_v2_pv_off = natan.get_tone_evoked_FR(
        pv_off['t'],
        pv_off['u'],
        pv_off['v1'],
        pv_off['v2'],
        pv_off['tonelist'])

    maxes_u_som_off,maxes_v1_som_off,maxes_v2_som_off = natan.get_tone_evoked_FR(
        som_off['t'],
        som_off['u'],
        som_off['v1'],
        som_off['v2'],
        som_off['tonelist'])

    
    fig = plt.figure(figsize=(12,5))
    gs = gridspec.GridSpec(3,4)
    ax11 = plt.subplot(gs[0,0])
    ax12 = plt.subplot(gs[0,1])
    
    ax21 = plt.subplot(gs[1,0])
    ax22 = plt.subplot(gs[1,1])

    ax31 = plt.subplot(gs[2,0])
    ax32 = plt.subplot(gs[2,1])

    ax13 = plt.subplot(gs[:,2])
    ax14 = plt.subplot(gs[:,3])
    
    #gs.update(hspace=.5)

    

    # tone1on, tone1off (before adaptation)    
    # get start/end time index
    idx_tone1_start = np.argmin(np.abs(control['t']-float(pars['tone1on'])))+1
    idx_tone1_end = np.argmin(np.abs(control['t']-float(pars['tone1off'])))-1

    # tone5on, tone5off (after adaptation)
    # get start/end time index
    idx_tone5_start = np.argmin(np.abs(control['t']-float(pars['tone5on'])))+1
    idx_tone5_end = np.argmin(np.abs(control['t']-float(pars['tone5off'])))-1
    
    
    
    control['t'] = control['t']*10
    pv_off['t'] = pv_off['t']*10
    som_off['t'] = som_off['t']*10
    
    idx_range1 = np.arange(idx_tone1_start-10,idx_tone1_end)
    idx_range5 = np.arange(idx_tone5_start-10,idx_tone5_end)

    xmin1 = control['t'][idx_range1][0]
    xmax1 = control['t'][idx_range1][-1]

    xmin5 = control['t'][idx_range5][0]
    xmax5 = control['t'][idx_range5][-1]

    ymin = -.01
    ymax = .9

    ###################################### control
    ax11.plot(control['t'][idx_range1],control['v2'][idx_range1],color=som_color,label='SOM',lw=1.5)    
    ax11.plot(control['t'][idx_range1],control['u'][idx_range1],color='k',lw=2,label='Pyr')
    ax11.plot(control['t'][idx_range1],control['v1'][idx_range1],color=pv_color,label='PV',lw=1.5,dashes=(5,2))

    ax12.plot(control['t'][idx_range5],control['v2'][idx_range5],color=som_color,label='SOM',lw=1.5)    
    ax12.plot(control['t'][idx_range5],control['u'][idx_range5],color='k',lw=2,label='Pyr')
    ax12.plot(control['t'][idx_range5],control['v1'][idx_range5],color=pv_color,label='PV',lw=1.5,dashes=(5,2))


    ###################################### pv off
    ax21.plot(control['t'][idx_range1],control['u'][idx_range1],color='gray',label='Control',lw=2)
    ax22.plot(control['t'][idx_range5],control['u'][idx_range5],color='gray',lw=2,label='')
    
    ax21.plot(pv_off['t'][idx_range1],pv_off['v2'][idx_range1],color=som_color,label='',lw=1.5)    
    ax21.plot(pv_off['t'][idx_range1],pv_off['u'][idx_range1],color='k',lw=2,label='')
    ax21.plot(pv_off['t'][idx_range1],pv_off['v1'][idx_range1],color=pv_color,label='',lw=1.5,dashes=(5,2))

    ax22.plot(pv_off['t'][idx_range5],pv_off['v2'][idx_range5],color=som_color,label='',lw=1.5)    
    ax22.plot(pv_off['t'][idx_range5],pv_off['u'][idx_range5],color='k',lw=2,label='')
    ax22.plot(pv_off['t'][idx_range5],pv_off['v1'][idx_range5],color=pv_color,label='',lw=1.5,dashes=(5,2))

    ###################################### som off
    ax31.plot(control['t'][idx_range1],control['u'][idx_range1],color='gray',label='Pyr',lw=2)
    ax32.plot(control['t'][idx_range5],control['u'][idx_range5],color='gray',lw=2,label='Pyr')
    
    ax31.plot(som_off['t'][idx_range1],som_off['v2'][idx_range1],color=som_color,label='SOM',lw=1.5)    
    ax31.plot(som_off['t'][idx_range1],som_off['u'][idx_range1],color='k',lw=2,label='Pyr')
    ax31.plot(som_off['t'][idx_range1],som_off['v1'][idx_range1],color=pv_color,label='PV',lw=1.5,dashes=(5,2))

    ax32.plot(som_off['t'][idx_range5],som_off['v2'][idx_range5],color=som_color,label='SOM',lw=1.5)    
    ax32.plot(som_off['t'][idx_range5],som_off['u'][idx_range5],color='k',lw=2,label='Pyr')
    ax32.plot(som_off['t'][idx_range5],som_off['v1'][idx_range5],color=pv_color,label='PV',lw=1.5,dashes=(5,2))

    #ax2.plot(control['t'][idx_range],control['ia'][idx_range],label='Thalamus')


    ###################################### FR

    tone_number = np.array([0,1,2,3,4])
    

    adapted_fr = maxes_u_control[-1,1]

    bar_wide = 0.4
    bar_width = 0.25
    ax13.set_title(r'\textbf{G} $\quad$ Mean Firing Rate',loc='left')
    ax13.bar(tone_number,maxes_u_control[:,1]/adapted_fr,width=bar_width,label='Control',color='k')
    ax13.bar(tone_number+bar_width,maxes_u_pv_off[:,1]/adapted_fr,width=bar_width,label='PV Off',edgecolor='k',facecolor=pv_color,hatch=hatch)
    ax13.bar(tone_number+2*bar_width,maxes_u_som_off[:,1]/adapted_fr,width=bar_width,label='SOM Off',edgecolor='k',color=som_color)
    ax13.plot([0,4],[1,1],ls='--',color='gray')

    ax13.legend()
    

    ax14.set_title(r'\textbf{H} $\quad$ Difference from Control',loc='left')
    ax14.bar(tone_number,np.abs(maxes_u_control[:,1]-maxes_u_pv_off[:,1])/adapted_fr,width=bar_wide,label='PV Off',edgecolor='k',facecolor=pv_color,hatch=hatch)
    ax14.bar(tone_number+bar_wide,np.abs(maxes_u_control[:,1]-maxes_u_som_off[:,1])/adapted_fr,edgecolor='k',width=bar_wide,label='SOM Off',color=som_color)
    #ax12.plot([0,4],[1,1],ls='--',color='gray')

    ax14.legend(loc='lower right')


    ax11.set_ylabel('Neural Activity\n (Control)')
    ax21.set_ylabel('Neural Activity\n (PV Off)')
    ax31.set_ylabel('Neural Activity\n (SOM Off)')

    #ax2.set_ylabel('Input')
    #ax2.set_xlabel('Time (ms)')

    ax11.legend()
    ax21.legend()
    #ax2.legend()
    
    #ax1.set_xticks([])

    ax11.set_ylim(ymin,ymax)
    ax12.set_ylim(ymin,ymax)

    ax21.set_ylim(ymin,ymax)
    ax22.set_ylim(ymin,ymax)

    ax31.set_ylim(ymin,ymax)
    ax32.set_ylim(ymin,ymax)
    
    ax11.set_xlim(xmin1,xmax1)
    ax12.set_xlim(xmin5,xmax5)
    
    ax21.set_xlim(xmin1,xmax1)
    ax22.set_xlim(xmin5,xmax5)
    
    ax31.set_xlim(xmin1,xmax1)
    ax32.set_xlim(xmin5,xmax5)

    ax31.set_xlabel('Time (ms)')
    ax32.set_xlabel('Time (ms)')

    ax13.set_xlabel('Tone Number')
    ax14.set_xlabel('Tone Number')

    ax11.set_title(r'\textbf{A} $\quad$ Before Adaptation',loc='left')
    ax12.set_title(r'\textbf{B} $\quad$ After Adaptation',loc='left')

    ax21.set_title(r'\textbf{C} $\quad$ Before Adaptation',loc='left')
    ax22.set_title(r'\textbf{D} $\quad$ After Adaptation',loc='left')

    ax31.set_title(r'\textbf{E} $\quad$ Before Adaptation',loc='left')
    ax32.set_title(r'\textbf{F} $\quad$ After Adaptation',loc='left')
    
    #ax2.set_title(r'\textbf{B}',loc='left')

    #ax1.set_title(r'\textbf{A}',x=0,y=.96)
    #ax2.set_title(r'\textbf{B}',x=0,y=.925)

    plt.tight_layout()
    return fig





def r3_s3_ssa_full(recompute=True):

    # dummy stimulation list for figure
    np.random.seed(0)
    pos_list = np.random.rand(30)
    
    standard_tone = -.25
    deviant_tone = 0.25
    deviant_idxs = pos_list>.95
    
    pos_list[deviant_idxs] = deviant_tone # 10% of tones are deviant (.25)
    pos_list[np.logical_not(deviant_idxs)] = standard_tone # 90% of tones are standard (-.25)

    vals = np.unique(pos_list) # get unique list of positions for later
    assert(len(vals) == 2) # this experiment is only for 2 tones
    #pos_list = [-1,1]

    tone_on_list = np.arange(len(pos_list),dtype=int)*40+30 # tone on times
    tone_dur_list = np.ones(len(tone_on_list),dtype=int)*10
    
    fname = 'rate_models/xpp/cols3_ssa.ode'
    pars = read_pars_values_from_file(fname)
    inits = read_init_values_from_file(fname)

    # returns tuple (t,u,v1,v2,inits,parameters,tonelist)    
    control = c3_ssa.run_experiment(
        fname,
        pars,inits,
        return_all=True)

    pars['pv_opto']=4

    pv_off = c3_ssa.run_experiment(
        fname,
        pars,inits,
        return_all=True)

    pars['pv_opto']=-0.5
    
    pv_on = c3_ssa.run_experiment(
        fname,
        pars,inits,
        return_all=True)
    
    pars['pv_opto']=0
    pars['som_opto']=2
    
    som_off = c3_ssa.run_experiment(
        fname,
        pars,inits,
        return_all=True)

    pars['som_opto']=-1.2
        
    som_on = c3_ssa.run_experiment(
        fname,
        pars,inits,
        return_all=True)
    
    maxes_u_control,maxes_v1_control,maxes_v2_control = c3_ssa.get_tone_evoked_FR(
        control['t'],
        control['u2'],
        control['p2'],
        control['s2'],
        control['tonelist'])

    maxes_u_pv_off,maxes_v1_pv_off,maxes_v2_pv_off = c3_ssa.get_tone_evoked_FR(
        pv_off['t'],
        pv_off['u2'],
        pv_off['p2'],
        pv_off['s2'],
        pv_off['tonelist'])

    maxes_u_som_off,maxes_v1_som_off,maxes_v2_som_off = c3_ssa.get_tone_evoked_FR(
        som_off['t'],
        som_off['u2'],
        som_off['p2'],
        som_off['s2'],
        som_off['tonelist'])



    maxes_u_pv_on,maxes_v1_pv_on,maxes_v2_pv_on = c3_ssa.get_tone_evoked_FR(
        pv_on['t'],
        pv_on['u2'],
        pv_on['p2'],
        pv_on['s2'],
        pv_on['tonelist'])

    maxes_u_som_on,maxes_v1_som_on,maxes_v2_som_on = c3_ssa.get_tone_evoked_FR(
        som_on['t'],
        som_on['u2'],
        som_on['p2'],
        som_on['s2'],
        som_on['tonelist'])


    fig = plt.figure(figsize=(8,6),constrained_layout=True)
    gs = gridspec.GridSpec(5, 3,figure=fig)
    #gs = fig.add_gridspec(3,4)
    #gs = gridspec.GridSpec(3, 3)
    ax11 = plt.subplot(gs[0, :2])
    #ax11.set_title('control')

    ax21 = plt.subplot(gs[1:3,0])
    ax22 = plt.subplot(gs[1:3,1])

    ax23 = plt.subplot(gs[1:3,2])
    
    
    ax41 = plt.subplot(gs[3:,0])
    ax42 = plt.subplot(gs[3:,1])

    ax43 = plt.subplot(gs[3:,2])
    

    gs.update(hspace=1.5,wspace=.5)
    

    #ax23 = plt.subplot(gs[1:,2])
    #ax24 = plt.subplot(gs[1:,3])


    ax11.scatter(tone_on_list[pos_list>0]*10,pos_list[pos_list>0],color='red',label='Deviant Tone')
    ax11.scatter(tone_on_list[pos_list<0]*10,pos_list[pos_list<0],color='gray',label='Standard Tone')
    

    bar_wide = 0.4
    
    tone_number = np.arange(-1,5)
    #adapted_fr = standard_FR_control4#maxes_u_control[-1,1]
    
    tone_number = np.array([0,1,2,3,4])
    
    adapted_fr = maxes_u_control[-1,1]

    alpha = .4
    bar_width = 0.25

    ax21.bar(tone_number-1.5*bar_width+bar_width/2.,maxes_u_control[:,1]/adapted_fr,width=bar_width,label='Control',color='black')
    ax21.bar(tone_number-bar_width/2.+0+bar_width/2.,maxes_u_pv_off[:,1]/adapted_fr,width=bar_width,label='PV Inact.',facecolor=pv_color,hatch=hatch,edgecolor='k')
    ax21.bar(tone_number+bar_width/2.+0+bar_width/2.,maxes_u_som_off[:,1]/adapted_fr,width=bar_width,label='SOM Inact.',facecolor=som_color,edgecolor='k')

    ax23.bar(tone_number-1.5*bar_width+bar_width/2.,maxes_u_control[:,1]/adapted_fr,width=bar_width,label='Control',color='black')
    ax23.bar(tone_number-bar_width/2.+0+bar_width/2.,maxes_u_pv_on[:,1]/adapted_fr,width=bar_width,label='PV Act.',facecolor=pv_color,hatch=hatch,edgecolor='k')
    ax23.bar(tone_number+bar_width/2.+0+bar_width/2.,maxes_u_som_on[:,1]/adapted_fr,width=bar_width,label='SOM Act.',facecolor=som_color,edgecolor='k')

    
    ax41.bar(tone_number-bar_wide/2,np.abs(maxes_u_control[:,1]-maxes_u_pv_off[:,1])/adapted_fr,label='PV Inact.',width=bar_wide,facecolor=pv_color,edgecolor='k',hatch=hatch)

    ax43.bar(tone_number-bar_wide/2,(maxes_u_pv_on[:,1]-maxes_u_control[:,1])/adapted_fr,label='PV Act.',width=bar_wide,facecolor=pv_color,edgecolor='k',hatch=hatch)

    #ax22.bar(0-bar_wide/2,np.abs(maxes_u_control[:,1][0]-maxes_u_pv_off[:,1][0])/adapted_fr,label='PV Off',width=bar_wide,facecolor=pv_color,edgecolor='k',hatch=hatch)
    #ax22.bar(1-bar_wide/2,np.abs(maxes_u_control[:,1][1]-maxes_u_pv_off[:,1][1])/adapted_fr,width=bar_wide,facecolor=pv_color,edgecolor='k',hatch=hatch)
    #ax22.bar(2-bar_wide/2,np.abs(maxes_u_control[:,1][2]-maxes_u_pv_off[:,1][2])/adapted_fr,width=bar_wide,facecolor=pv_color,edgecolor='k',hatch=hatch)
    #ax22.bar(3-bar_wide/2,np.abs(maxes_u_control[:,1][3]-maxes_u_pv_off[:,1][3])/adapted_fr,width=bar_wide,facecolor=pv_color,edgecolor='k',hatch=hatch)
    #ax22.bar(4-bar_wide/2,np.abs(maxes_u_control[:,1][4]-maxes_u_pv_off[:,1][4])/adapted_fr,width=bar_wide,facecolor=pv_color,edgecolor='k',hatch=hatch)
    
    y = np.abs(maxes_u_control[:,1]-maxes_u_som_off[:,1])/adapted_fr

    ax41.bar(tone_number+bar_wide/2.,y,width=bar_wide,facecolor=som_color,label='SOM Inact.',edgecolor='k')

    ax43.bar(tone_number+bar_wide/2.,maxes_u_som_on[:,1]-maxes_u_control[:,1],width=bar_wide,facecolor=som_color,label='SOM Act.',edgecolor='k')
    #ax22.bar(0+bar_wide/2.,y[0],width=bar_wide,facecolor=som_color,label='SOM Off',edgecolor='k')
    #ax22.bar(1+bar_wide/2.,y[1],width=bar_wide,facecolor=som_color,edgecolor='k')
    #ax22.bar(2+bar_wide/2.,y[2],width=bar_wide,facecolor=som_color,edgecolor='k')
    #ax22.bar(3+bar_wide/2.,y[3],width=bar_wide,facecolor=som_color,edgecolor='k')
    #ax22.bar(4+bar_wide/2.,y[4],width=bar_wide,facecolor=som_color,edgecolor='k')


    
    paradigm = 'ssa'
    #seed = 0

    #seedlist = np.arange(0,3,1)
    seedlist = np.arange(0,2,1)
    
    # check if saved results exist
    fname_ctrl = "dat/ssa_ctrl_seedlen="+str(len(seedlist))+".dat"
    fname_pv_off = "dat/ssa_pv_off_seedlen="+str(len(seedlist))+".dat"
    fname_som_off = "dat/ssa_som_off_seedlen="+str(len(seedlist))+".dat"

    fname_pv_on = "dat/ssa_pv_on_seedlen="+str(len(seedlist))+".dat"
    fname_som_on = "dat/ssa_som_on_seedlen="+str(len(seedlist))+".dat"

    # get std deviation for spike differences
    #pvdiff = np.abs(spike_pvoff_pyr2[:,4:] - spike_control_pyr2[:,4:])
    #somdiff = np.abs(spike_somoff_pyr2[:,4:] - spike_control_pyr2[:,4:])

    fname_pv_off_diff = "dat/ssa_pv_off_diff_seedlen="+str(len(seedlist))+".dat"
    fname_som_off_diff = "dat/ssa_som_off_diff_seedlen="+str(len(seedlist))+".dat"
    
    fname_pv_on_diff = "dat/ssa_pv_on_diff_seedlen="+str(len(seedlist))+".dat"
    fname_som_on_diff = "dat/ssa_som_on_diff_seedlen="+str(len(seedlist))+".dat"

    
    if os.path.isfile(fname_ctrl) and \
       os.path.isfile(fname_pv_off) and os.path.isfile(fname_som_off) and \
       os.path.isfile(fname_pv_off_diff) and os.path.isfile(fname_som_off_diff) and \
       os.path.isfile(fname_pv_on) and os.path.isfile(fname_som_on) and \
       os.path.isfile(fname_pv_on_diff) and os.path.isfile(fname_som_on_diff) and \
       not(recompute):
        
        results_ctrl = np.loadtxt(fname_ctrl)
        results_pv_off = np.loadtxt(fname_pv_off)
        results_som_off = np.loadtxt(fname_som_off)

        results_pv_on = np.loadtxt(fname_pv_on)
        results_som_on = np.loadtxt(fname_som_on)

        results_pv_off_diff = np.loadtxt(fname_pv_off_diff)
        results_som_off_diff = np.loadtxt(fname_som_off_diff)

        results_pv_on_diff = np.loadtxt(fname_pv_on_diff)
        results_som_on_diff = np.loadtxt(fname_som_on_diff)

    else:
        
        results_ctrl = np.zeros((len(seedlist),5)) # ( seed #, expt/column #) )
        results_pv_off = np.zeros((len(seedlist),5))
        results_som_off = np.zeros((len(seedlist),5))

        results_pv_on = np.zeros((len(seedlist),5))
        results_som_on = np.zeros((len(seedlist),5))

        results_pv_off_diff = np.zeros((len(seedlist),5))
        results_som_off_diff = np.zeros((len(seedlist),5))

        results_pv_on_diff = np.zeros((len(seedlist),5))
        results_som_on_diff = np.zeros((len(seedlist),5))

        
        i = 0

        for seed in seedlist:
            print 'seed',seed
            fulldict1 = s3.setup_and_run(paradigm=paradigm,seed=seed)
            fulldict2 = s3.setup_and_run(pv_opto=.2,paradigm=paradigm,seed=seed)#fulldict2 = setup_and_run(pv_opto=.3)
            fulldict3 = s3.setup_and_run(som_opto=1.,paradigm=paradigm,seed=seed)

            #fulldict1On = s3.setup_and_run(paradigm=paradigm,seed=seed)
            fulldict2On = s3.setup_and_run(pv_opto=-.2,paradigm=paradigm,seed=seed)#fulldict2 = setup_and_run(pv_opto=.3)
            fulldict3On = s3.setup_and_run(som_opto=-.5,paradigm=paradigm,seed=seed)

            n = fulldict1['n_pyr']

            FR_control_pyr2,dev_control_pyr2,spike_control_pyr2 = s3.collect_FR_dev(
                fulldict1['stim_arr1']+fulldict1['stim_arr3'],
                fulldict1['stim_dt'],
                fulldict1['defaultclock'],
                fulldict1['spikemon_PYR2'],n)

            FR_pvoff_pyr2,dev_pvoff_pyr2,spike_pvoff_pyr2 = s3.collect_FR_dev(
                fulldict2['stim_arr1']+fulldict2['stim_arr3'],
                fulldict2['stim_dt'],
                fulldict2['defaultclock'],
                fulldict2['spikemon_PYR2'],n)

            FR_somoff_pyr2,dev_somoff_pyr2,spike_somoff_pyr2 = s3.collect_FR_dev(
                fulldict3['stim_arr1']+fulldict3['stim_arr3'],
                fulldict3['stim_dt'],
                fulldict3['defaultclock'],
                fulldict3['spikemon_PYR2'],n)
            
            FR_pvon_pyr2,dev_pvon_pyr2,spike_pvon_pyr2 = s3.collect_FR_dev(
                fulldict2On['stim_arr1']+fulldict2On['stim_arr3'],
                fulldict2On['stim_dt'],
                fulldict2On['defaultclock'],
                fulldict2On['spikemon_PYR2'],n)

            FR_somon_pyr2,dev_somon_pyr2,spike_somon_pyr2 = s3.collect_FR_dev(
                fulldict3On['stim_arr1']+fulldict3On['stim_arr3'],
                fulldict3On['stim_dt'],
                fulldict3On['defaultclock'],
                fulldict3On['spikemon_PYR2'],n)


            # get std deviation for spike differences

            results_ctrl[i,:] = FR_control_pyr2[4:]
            results_pv_off[i,:] = FR_pvoff_pyr2[4:]
            results_som_off[i,:] = FR_somoff_pyr2[4:]
            
            results_pv_on[i,:] = FR_pvon_pyr2[4:]
            results_som_on[i,:] = FR_somon_pyr2[4:]

            results_pv_off_diff[i,:] = np.mean(np.abs(spike_pvoff_pyr2[:,4:] - spike_control_pyr2[:,4:]),axis=0)
            results_som_off_diff[i,:] = np.mean(np.abs(spike_somoff_pyr2[:,4:] - spike_control_pyr2[:,4:]),axis=0)
            
            results_pv_on_diff[i,:] = np.mean(np.abs(spike_pvon_pyr2[:,4:] - spike_control_pyr2[:,4:]),axis=0)
            results_som_on_diff[i,:] = np.mean(np.abs(spike_somon_pyr2[:,4:] - spike_control_pyr2[:,4:]),axis=0)


            i += 1

        np.savetxt(fname_ctrl,results_ctrl)
        np.savetxt(fname_pv_off,results_pv_off)
        np.savetxt(fname_som_off,results_som_off)
        
        np.savetxt(fname_pv_on,results_pv_on)
        np.savetxt(fname_som_on,results_som_on)
        
        
        np.savetxt(fname_pv_off_diff,results_pv_off_diff)
        np.savetxt(fname_som_off_diff,results_som_off_diff)
        
        np.savetxt(fname_pv_on_diff,results_pv_on_diff)
        np.savetxt(fname_som_on_diff,results_som_on_diff)
    

    bin_factor = 1./2

    tone_number1 = np.arange(0,9,1)#np.arange(len(np.where((fulldict1['stim_arr1']+fulldict1['stim_arr3'])!=0)[0]))
    adapted_fr = results_ctrl[0,-1]#FR_control_pyr2[-1]

    mean_ctrl = np.mean(results_ctrl,axis=0)
    std_ctrl = np.std(results_ctrl,axis=0)
    
    mean_pv_off = np.mean(results_pv_off,axis=0)
    std_pv_off = np.std(results_pv_off,axis=0)
    
    mean_som_off = np.mean(results_som_off,axis=0)
    std_som_off = np.std(results_som_off,axis=0)

    mean_pv_on = np.mean(results_pv_on,axis=0)
    std_pv_on = np.std(results_pv_on,axis=0)
    
    mean_som_on = np.mean(results_som_on,axis=0)
    std_som_on = np.std(results_som_on,axis=0)

    mean_pv_off_diff = np.mean(results_pv_off_diff,axis=0)
    std_pv_off_diff = np.std(results_pv_off_diff,axis=0)

    mean_pv_on_diff = np.mean(results_pv_on_diff,axis=0)
    std_pv_on_diff = np.std(results_pv_on_diff,axis=0)

    mean_som_on_diff = np.mean(results_som_on_diff,axis=0)
    std_som_on_diff = np.std(results_som_on_diff,axis=0)

    mean_som_off_diff = np.mean(results_som_off_diff,axis=0)
    std_som_off_diff = np.std(results_som_off_diff,axis=0)


    ############################## spiking model plots INACTIVATION

    ax22.bar(tone_number1[4:]-4-1.5*bar_width+bar_width/2.,mean_ctrl,
             width=bar_width,color='black',yerr=[(0,0,0,0,0),std_ctrl],
             error_kw=dict(lw=1, capsize=1, capthick=1),label='Control')
    ax22.bar(tone_number1[4:]-4,mean_pv_off,
             width=bar_width,facecolor=pv_color,yerr=[(0,0,0,0,0),std_pv_off],
             error_kw=dict(lw=1, capsize=1, capthick=1),edgecolor='k',hatch=hatch,label='PV Inact.')
    ax22.bar(tone_number1[4:]-4+bar_width,mean_som_off,
             width=bar_width,facecolor=som_color,yerr=[(0,0,0,0,0),std_som_off],
             error_kw=dict(lw=1, capsize=1, capthick=1),edgecolor='k',label='SOM Inact.')
    
    ax42.bar(tone_number1[4:]-4-bar_wide/2,mean_pv_off-mean_ctrl,
             width=bar_wide,label='PV Inact.',facecolor=pv_color,yerr=[(0,0,0,0,0),(std_ctrl+std_pv_off)/2.],
             error_kw=dict(lw=1, capsize=1, capthick=1),edgecolor='k',hatch=hatch)
    ax42.bar(tone_number1[4:]-4+bar_wide/2,mean_som_off-mean_ctrl,
             width=bar_wide,label='SOM Inact.',facecolor=som_color,yerr=[(0,0,0,0,0),(std_ctrl+std_som_off)/2.],
             error_kw=dict(lw=1, capsize=1, capthick=1),edgecolor='k')
    

    ############################## spiking model plots ACTIVATION
    """
    ax43.bar(tone_number1[4:]-4-1.5*bar_width+bar_width/2.,mean_ctrl,
             width=bar_width,color='black',yerr=[(0,0,0,0,0),std_ctrl],
             error_kw=dict(lw=1, capsize=1, capthick=1))
    ax43.bar(tone_number1[4:]-4,mean_pv_on,
             width=bar_width,facecolor=pv_color,yerr=[(0,0,0,0,0),std_pv_on],
             error_kw=dict(lw=1, capsize=1, capthick=1),edgecolor='k',hatch=hatch)
    ax43.bar(tone_number1[4:]-4+bar_width,mean_som_on,
             width=bar_width,facecolor=som_color,yerr=[(0,0,0,0,0),std_som_on],
             error_kw=dict(lw=1, capsize=1, capthick=1),edgecolor='k')

    
    ax44.bar(tone_number1[4:]-4-bar_wide/2,mean_pv_on-mean_ctrl,
             width=bar_wide,label='control-pv_off',facecolor=pv_color,yerr=[(std_ctrl+std_pv_on)/2.,(0,0,0,0,0)],
             error_kw=dict(lw=1, capsize=1, capthick=1),edgecolor='k',hatch=hatch)
    ax44.bar(tone_number1[4:]-4+bar_wide/2,mean_som_on-mean_ctrl,
             width=bar_wide,label='control-som_off',facecolor=som_color,yerr=[(std_ctrl+std_som_on)/2.,(0,0,0,0,0)],
             error_kw=dict(lw=1, capsize=1, capthick=1),edgecolor='k')
    """

    ax41.set_xlabel('Post-Deviant Tone \#')
    ax42.set_xlabel('Post-Deviant Tone \#')
    ax43.set_xlabel('Post-Deviant Tone \#')
    
    ax21.set_ylabel('Normalized Mean FR')
    ax22.set_ylabel('Normalized Mean FR')
    ax23.set_ylabel('Normalized Mean FR')

    ax41.set_ylabel('Mean FR Change')
    ax42.set_ylabel('Mean FR Change')    
    ax43.set_ylabel('Mean FR Change')

    ax11.set_title(r'\textbf{A}',x=0)
    ax21.set_title(r'\textbf{B}',x=0)
    ax22.set_title(r'\textbf{C}',x=0)
    ax23.set_title(r'\textbf{D}',x=0)

    #ax11.spines['bottom'].set_position('center')
    ax11.spines['left'].set_visible(False)
    ax11.spines['right'].set_visible(False)
    ax11.spines['top'].set_visible(False)
    ax11.set_yticks([])
    
    ax11.set_xlabel('Time (ms)')
    ax11.set_ylim(-.3,.3)

    #plt.tight_layout()
    
    #ax11.legend()    
    
    ax21.spines['right'].set_visible(False)
    ax21.spines['top'].set_visible(False)    
    ax21.xaxis.set_ticks_position('bottom')
    ax21.yaxis.set_ticks_position('left')

    ax22.spines['right'].set_visible(False)
    ax22.spines['top'].set_visible(False)    
    ax22.yaxis.set_ticks_position('left')
    ax22.xaxis.set_ticks_position('bottom')

    ax23.spines['right'].set_visible(False)
    ax23.spines['top'].set_visible(False)    
    ax23.xaxis.set_ticks_position('bottom')
    ax23.yaxis.set_ticks_position('left')


    ax41.spines['right'].set_visible(False)
    ax41.spines['top'].set_visible(False)    
    ax41.xaxis.set_ticks_position('bottom')
    ax41.yaxis.set_ticks_position('left')

    ax42.spines['right'].set_visible(False)
    ax42.spines['top'].set_visible(False)    
    ax42.xaxis.set_ticks_position('bottom')
    ax42.yaxis.set_ticks_position('left')

    ax43.spines['right'].set_visible(False)
    ax43.spines['bottom'].set_visible(False)    
    ax43.xaxis.set_ticks_position('bottom')
    ax43.yaxis.set_ticks_position('left')


    ax21.set_ylim(0,3)
    #ax22.set_ylim(0,1)
    #ax23.set_ylim(0,4)
    #ax24.set_ylim(0,1.2)
    ax43.set_ylim(-.5,0)

    ax21.set_xlim(-.5,4.5)
    ax22.set_xlim(-.5,4.5)
    ax23.set_xlim(-.5,4.5)
    

    ax21.set_xticks(np.arange(0,5))
    ax22.set_xticks(np.arange(0,5))
    ax23.set_xticks(np.arange(0,5))


    
    ax41.set_xticks(np.arange(0,5))
    ax42.set_xticks(np.arange(0,5))
    ax43.set_xticks(np.arange(0,5))


    ax11.legend(bbox_to_anchor=(.5, 1.45), loc='upper center', borderaxespad=0.,ncol=2,prop={'size': 7})
    ax21.legend(prop={'size': 7})
    ax22.legend(prop={'size': 7})

    ax23.legend(prop={'size': 7})

    ax41.legend(prop={'size': 7},loc='lower right',framealpha=0.85)
    ax42.legend(prop={'size': 7},loc='lower right',framealpha=0.85)

    ax43.legend(prop={'size': 7})

    #plt.tight_layout()

    return fig






def r3_s3_fs_full(recompute=False):

    ############################################# rate model
    
    fname = 'rate_models/xpp/cols3_fs.ode'
    pars = read_pars_values_from_file(fname)
    inits = read_init_values_from_file(fname)

    # repeat simulation for different frequencies.
    # pars['mode'] = '1' makes the first tone appear in the first column
    # pars['mode'] = '2' makes the first tone appear in the second column.. etc.

    results = np.zeros((3,10))
    
    for i in range(3):
        pars['mode']=str(i+1)

        pars['pv_opto']=0
        pars['som_opto']=0

        # returns tuple (t,u,v1,v2,inits,parameters,tonelist)    
        control = c3_fs.run_experiment(
            fname,
            pars,inits,
            return_all=True)

        pars['pv_opto']=.1

        pv_off = c3_fs.run_experiment(
            fname,
            pars,inits,
            return_all=True)

        pars['pv_opto']=-.025
        pv_on = c3_fs.run_experiment(
            fname,
            pars,inits,
            return_all=True)

        
        pars['pv_opto']=0
        #pars['som_opto']=.2
        pars['som_opto']=.5

        som_off = c3_fs.run_experiment(
            fname,
            pars,inits,
            return_all=True)

        pars['som_opto']=-.1
        
        som_on = c3_fs.run_experiment(
            fname,
            pars,inits,
            return_all=True)

        
        # get first max u (i), second max u (u2).
        # get first max u with pv opto (i), second max u with pv opto (u2).
        # get first max u with som opto (i), second max u with som opto (u2).

        # get tone list stard and end index for first and second tones
        # get tone list times
        
        tone1On,tone1Off = control['tonelist'][0]
        tone2On,tone2Off = control['tonelist'][1]
        
        idx1_start = np.argmin(np.abs(control['t']-tone1On))+1 # first time interval index
        idx1_end = np.argmin(np.abs(control['t']-tone1Off))-1

        idx2_start = np.argmin(np.abs(control['t']-tone2On))+1 # second time interval index
        idx2_end = np.argmin(np.abs(control['t']-tone2Off))-1
        
        # get first tone (varies as a function of i)
        control1 = c3_fs.get_max_FR(control['u'+str(i+1)],idx1_start,idx1_end)
        pv1 = c3_fs.get_max_FR(pv_off['u'+str(i+1)],idx1_start,idx1_end)
        som1 = c3_fs.get_max_FR(som_off['u'+str(i+1)],idx1_start,idx1_end)

        pv1on = c3_fs.get_max_FR(pv_on['u'+str(i+1)],idx1_start,idx1_end)
        som1on = c3_fs.get_max_FR(som_on['u'+str(i+1)],idx1_start,idx1_end)

        
        # get second tone (always u2)
        control2 = c3_fs.get_max_FR(control['u2'],idx2_start,idx2_end)
        pv2 = c3_fs.get_max_FR(pv_off['u2'],idx2_start,idx2_end)
        som2 = c3_fs.get_max_FR(som_off['u2'],idx2_start,idx2_end)

        pv2on = c3_fs.get_max_FR(pv_on['u2'],idx2_start,idx2_end)
        som2on = c3_fs.get_max_FR(som_on['u2'],idx2_start,idx2_end)
        
        results[i,:] = [control1,control2,pv1,pv2,som1,som2,pv1on,pv2on,som1on,som2on]
        #results[i,:] = [control1,control2,pv1,pv2,som1,som2]

        # end 3 tone loop


    ############################################# spiking model
    paradigms = ['fs1','fs2','fs3']
    #seedlist = np.arange(0,30,1)
    seedlist = np.arange(0,2,1)
    
    # check if saved results exist
    fname_ctrl = "dat/fs_ctrl_seedlen="+str(len(seedlist))+".dat"
    fname_pv = "dat/fs_pv_seedlen="+str(len(seedlist))+".dat"
    fname_som = "dat/fs_som_seedlen="+str(len(seedlist))+".dat"

    fname_pv_on = "dat/fs_pv_on_seedlen="+str(len(seedlist))+".dat"
    fname_som_on = "dat/fs_som_on_seedlen="+str(len(seedlist))+".dat"

    
    if os.path.isfile(fname_ctrl) and os.path.isfile(fname_pv) and os.path.isfile(fname_som) and \
       os.path.isfile(fname_pv_on) and os.path.isfile(fname_som_on) and \
       not(recompute):
        results_ctrl = np.loadtxt(fname_ctrl)
        results_pv = np.loadtxt(fname_pv)
        results_som = np.loadtxt(fname_som)

        results_pv_on = np.loadtxt(fname_pv_on)
        results_som_on = np.loadtxt(fname_som_on)

    else:
        results_ctrl = np.zeros((len(seedlist),3)) # ( seed #, expt/column #) )
        results_pv = np.zeros((len(seedlist),3))
        results_som = np.zeros((len(seedlist),3))

        results_pv_on = np.zeros((len(seedlist),3))
        results_som_on = np.zeros((len(seedlist),3))

        i = 0
        for seed in seedlist:
            for paradigm in paradigms:

                fulldict1 = s3.setup_and_run(paradigm=paradigm,seed=seed)
                fulldict2 = s3.setup_and_run(pv_opto=.1,paradigm=paradigm,seed=seed)#fulldict2 = setup_and_run(pv_opto=.3)
                fulldict3 = s3.setup_and_run(som_opto=1.,paradigm=paradigm,seed=seed)

                fulldict2On = s3.setup_and_run(pv_opto=-.1,paradigm=paradigm,seed=seed)#fulldict2 = setup_and_run(pv_opto=.3)
                fulldict3On = s3.setup_and_run(som_opto=-.2,paradigm=paradigm,seed=seed)
                
                suffix = paradigm[-1]

                bin_factor = 1./2

                start_time1 = 160
                start_time2 = 230
                interval_time = 50

                # 50-60...90-100,  start_time2-130...160-170    
                control_pa = s3.get_FR(start_time1,start_time1+interval_time,fulldict1['defaultclock'],fulldict1['spikemon_PYR'+suffix].t,fulldict1['n_pyr'])
                control_fs = s3.get_FR(start_time2,start_time2+interval_time,fulldict1['defaultclock'],fulldict1['spikemon_PYR2'].t,fulldict1['n_pyr'])

                pv_pa = s3.get_FR(start_time1,start_time1+interval_time,fulldict2['defaultclock'],fulldict2['spikemon_PYR'+suffix].t,fulldict1['n_pyr'])
                pv_fs = s3.get_FR(start_time2,start_time2+interval_time,fulldict2['defaultclock'],fulldict2['spikemon_PYR2'].t,fulldict1['n_pyr'])
                
                som_pa = s3.get_FR(start_time1,start_time1+interval_time,fulldict3['defaultclock'],fulldict3['spikemon_PYR'+suffix].t,fulldict1['n_pyr'])
                som_fs = s3.get_FR(start_time2,start_time2+interval_time,fulldict3['defaultclock'],fulldict3['spikemon_PYR2'].t,fulldict1['n_pyr'])


                pv_pa_on = s3.get_FR(start_time1,start_time1+interval_time,fulldict2On['defaultclock'],fulldict2On['spikemon_PYR'+suffix].t,fulldict1['n_pyr'])
                pv_fs_on = s3.get_FR(start_time2,start_time2+interval_time,fulldict2On['defaultclock'],fulldict2On['spikemon_PYR2'].t,fulldict1['n_pyr'])
                
                som_pa_on = s3.get_FR(start_time1,start_time1+interval_time,fulldict3On['defaultclock'],fulldict3On['spikemon_PYR'+suffix].t,fulldict1['n_pyr'])
                som_fs_on = s3.get_FR(start_time2,start_time2+interval_time,fulldict3On['defaultclock'],fulldict3On['spikemon_PYR2'].t,fulldict1['n_pyr'])

                
                control = control_fs/control_pa
                pv_off = pv_fs/pv_pa
                som_off = som_fs/som_pa

                pv_on = pv_fs_on/pv_pa_on
                som_on = som_fs_on/som_pa_on

                #results[int(suffix)-1,:] = [control,pv_off,som_off]
                results_ctrl[i,int(suffix)-1] = control
                results_pv[i,int(suffix)-1] = pv_off
                results_som[i,int(suffix)-1] = som_off

                results_pv_on[i,int(suffix)-1] = pv_on
                results_som_on[i,int(suffix)-1] = som_on
                

                if paradigm == 'fs2':
                    print('control,pvoff,somoff:',control,pv_off,som_off,'seed,paradigm',seed,paradigm)
            i += 1

        np.savetxt(fname_ctrl,results_ctrl)
        np.savetxt(fname_pv,results_pv)
        np.savetxt(fname_som,results_som)
        
        np.savetxt(fname_pv_on,results_pv_on)
        np.savetxt(fname_som_on,results_som_on)


    print results_ctrl
    print results_pv
    print results_som


    fig = plt.figure(figsize=(8,4))
    gs = gridspec.GridSpec(2,3)
    
    ax11 = plt.subplot(gs[0,0]) # pv off rate
    ax12 = plt.subplot(gs[0,1]) # som off rate
    ax13 = plt.subplot(gs[0,2]) # pv off spike

    ax21 = plt.subplot(gs[1,0])
    ax22 = plt.subplot(gs[1,1])
    ax23 = plt.subplot(gs[1,2])


    markersize = 4
    marker = 'o'
    ################################################## rate model plots
        
    bar_width = 0.2
    #ax11.set_title('Peak Response')
    #ax11.scatter(0,maxes_u_control[0,1],label='Control 1st Tone',color='black')
    #ax11.scatter(0,maxes_u_pv_off[0,1],label='',color=pv_color)

    #ax11.set_title('Normalized Peak Response (2nd Tone)')

    x = [-1,0,1]

    #results[i,:] = [control1,control2,pv1,pv2,som1,som2,pv1on,pv2on,som1on,som2on]
    
    control_probe = results[1,0]
    ax11.plot(x,results[:,1]/control_probe,label='Control',color='black',marker=marker,markersize=markersize)

    pv_probe = results[1,2]
    ax11.plot(x,results[:,3]/pv_probe,label='PV Inact.',color=pv_color,marker=marker,markersize=markersize,ls='--')

    #ax12.set_title('Normalized Peak Response (2nd Tone)')

    control_probe = results[1,0]    
    ax21.plot(x,results[:,1]/control_probe,label='Control',color='black',marker=marker,markersize=markersize)

    som_probe = results[1,4]
    ax21.plot(x,results[:,5]/som_probe,label='SOM Inact.',color=som_color,marker=marker,markersize=markersize)



    control_probe = results[1,0]
    ax13.plot(x,results[:,1]/control_probe,label='Control',color='black',marker=marker,markersize=markersize)

    pv_on_probe = results[1,6]
    ax13.plot(x,results[:,7]/pv_probe,label='PV Act.',color=pv_color,marker=marker,markersize=markersize,ls='--',alpha=1)

    control_probe = results[1,0]
    ax23.plot(x,results[:,1]/control_probe,label='Control',color='black',marker=marker,markersize=markersize)

    som_on_probe = results[1,8]
    ax23.plot(x,results[:,9]/som_probe,label='SOM Act.',color=som_color,marker=marker,markersize=markersize,alpha=1)

    
    #ax11.plot()
    
    #ax11.set_xlabel('Distance from Preferred Frequency')
    #ax12.set_xlabel('Distance from Preferred Frequency')
    
    
    #plt.tight_layout()


    ################################################### spiking model plots


    

    freqs = np.array([-1,0,1])

    mean_ctrl = np.mean(results_ctrl,axis=0)
    std_ctrl = np.std(results_ctrl,axis=0)
    
    mean_pv = np.mean(results_pv,axis=0)
    std_pv = np.std(results_pv,axis=0)
    
    mean_som = np.mean(results_som,axis=0)
    std_som = np.std(results_som,axis=0)


    mean_pv_on = np.mean(results_pv_on,axis=0)
    std_pv_on = np.std(results_pv_on,axis=0)
    
    mean_som_on = np.mean(results_som_on,axis=0)
    std_som_on = np.std(results_som_on,axis=0)

    print mean_ctrl,mean_pv,mean_som
    print std_ctrl,std_pv,std_som

    shift = 0.05
    
    #ax12.set_title('Normalized Peak Response (2nd Tone)')
    #ax12.plot(freqs,mean_ctrl,color='k',lw=2,marker=marker,markersize=markersize) # plot control over all tones
    ax12.errorbar(freqs,mean_ctrl,yerr=std_ctrl,label='Control',color='black',
             lw=2, capsize=2, capthick=1) # plot control over all tones

    #ax12.plot(freqs+shift,mean_pv,color=pv_color,lw=2,marker=marker,markersize=markersize,ls='--')
    ax12.errorbar(freqs+shift,mean_pv,yerr=std_pv,label='PV Inact.',color=pv_color,
                  lw=2, capsize=2, capthick=1,zorder=3,ls='--') # plot PV off over all tones
    
    #ax22.plot(freqs,mean_ctrl,color='k',lw=2,marker=marker,markersize=markersize)
    ax22.errorbar(freqs,mean_ctrl,yerr=std_ctrl,label='Control',color='black',
             lw=2, capsize=2, capthick=1) # plot control over all tones

    #ax22.plot(freqs+shift,mean_som,color=som_color,lw=2,marker=marker,markersize=markersize)
    ax22.errorbar(freqs+shift,mean_som,yerr=std_som,label='SOM Inact.',color=som_color,
                  lw=2, capsize=2, capthick=1,zorder=3) # plot SOM off over all tones


        
    #ax12.set_title('Normalized Peak Response (2nd Tone)')
    #ax23.plot(freqs,mean_ctrl,color='k',lw=2,marker=marker,markersize=markersize) # plot control over all tones
    #ax23.errorbar(freqs,mean_ctrl,yerr=std_ctrl,label='Control',color='black',
    #lw=1, capsize=1, capthick=1) # plot control over all tones

    #ax23.plot(freqs-shift,mean_pv_on,color=pv_color,lw=2,marker=marker,markersize=markersize,ls='--',alpha=1)
    #ax23.errorbar(freqs-shift,mean_pv_on,yerr=std_pv_on,label='PV Act.',color=pv_color,
    #    lw=1, capsize=1, capthick=1,zorder=3,ls='--',alpha=1) # plot PV off over all tones

        
    #ax12.set_title('Normalized Peak Response (2nd Tone)')
    #ax24.plot(freqs,mean_ctrl,color='k',lw=2,marker=marker,markersize=markersize) # plot control over all tones
    #ax24.errorbar(freqs,mean_ctrl,yerr=std_ctrl,label='Control',color='black',
    #lw=1, capsize=1, capthick=1) # plot control over all tones


    #ax24.plot(freqs-shift,mean_som_on,color=som_color,lw=2,marker=marker,markersize=markersize,alpha=1)
    #ax24.errorbar(freqs-shift,mean_som_on,yerr=std_som_on,label='SOM Act.',color=som_color,
    #lw=1, capsize=1, capthick=1,zorder=3,alpha=1) # plot SOM off over all tones


    ax11.set_ylabel('Pyr FR')
    ax21.set_ylabel('Pyr FR')

    ax21.set_xlabel('Distance from preferred frequency')
    ax22.set_xlabel('Distance from preferred frequency')
    ax23.set_xlabel('Distance from preferred frequency')
    #ax23.set_xlabel('Distance from preferred frequency')
    #ax24.set_xlabel('Distance from preferred frequency')

    ax11.set_xticks(x)
    ax12.set_xticks(x)
    ax13.set_xticks(x)

    ax21.set_xticks(x)
    ax22.set_xticks(x)
    ax23.set_xticks(x)

    ax11.legend(loc='upper center',prop={'size': 6})
    ax12.legend(loc='upper center',prop={'size': 6})
    ax13.legend(loc='upper center',prop={'size': 6})
    
    ax21.legend(loc='upper center',prop={'size': 6})
    ax22.legend(loc='upper center',prop={'size': 6})
    ax23.legend(loc='upper center',prop={'size': 6})

    ax11.set_title(r'\textbf{A} Rate Model',x=.3)
    ax12.set_title(r'\textbf{B} Spiking Model',x=.3)
    #ax12.set_title(r'\textbf{C} Spiking Model',x=.3)
    ax13.set_title(r'\textbf{C} Rate Model (Prediction)',x=.5)
    
    #ax12.set_title(r'\textbf{E}',x=0)
    #ax23.set_title(r'\textbf{F}',x=0)
    #ax23.set_title(r'\textbf{G}',x=0)
    #ax24.set_title(r'\textbf{H}',x=0)


    
    return fig




def CSI(df1,df2,sf1,sf2):
    return (df1+df2-sf1-sf2)/(df1+df2+sf1+sf2)


def run_sweep(px,pxname,py,pyname,recompute=False):
    """
    first pair of inputs is for the x axis, second pair of inputs is for the  y axis

    inputs in terms of cartesan coordinates so matrix is filled in a weird order.
    
    returns CSI matrix and corresponding x and y domains
    """

    x = px
    y = py

    # check if data directory exists and create if it does not.
    if not(os.path.isdir("dat")):
        os.mkdir("dat")

    
    fname_sol = "dat/"+pxname+'_'+pyname+'_'+str(len(x))+'_'+str(len(y))+\
                'xlo='+str(x[0])+\
                'xhi='+str(x[-1])+\
                'ylo='+str(y[0])+\
                'yhi='+str(y[-1])+\
                'sol.dat'
    
    fname_x = "dat/"+pxname+'_'+pyname+'_'+str(len(x))+'_'+str(len(y))+\
              'xlo='+str(x[0])+\
              'xhi='+str(x[-1])+\
              'ylo='+str(y[0])+\
              'yhi='+str(y[-1])+\
              'x.dat'
    
    fname_y = "dat/"+pxname+'_'+pyname+'_'+str(len(x))+'_'+str(len(y))+\
              'xlo='+str(x[0])+\
              'xhi='+str(x[-1])+\
              'ylo='+str(y[0])+\
              'yhi='+str(y[-1])+\
              'y.dat'
    
    # check if data exists for chosen parameters.
    if os.path.isfile(fname_sol) and os.path.isfile(fname_x) and os.path.isfile(fname_y) and not(recompute):
        return np.loadtxt(fname_sol),np.loadtxt(fname_x),np.loadtxt(fname_y)
    
    else:
        fname = 'rate_models/xpp/cols3_ssa.ode'
        inits = read_init_values_from_file(fname)

        sol = np.zeros((len(py),len(px)))

        for i in range(len(py)):
            for j in range(len(px)):
                pars = read_pars_values_from_file(fname)
                pars[pxname] = px[j]
                pars[pyname] = py[i]

                expt = c3_ssa.run_experiment(
                    fname,
                    pars,inits,
                    return_all=True)
                
                print expt['t'][-1]
                if pxname == 'pv_opto':
                    max_FRs,_,_ = c3_ssa.get_tone_evoked_FR(
                        expt['t'],
                        expt['u2'],
                        expt['p2'],
                        expt['s2'],
                        expt['tonelist'])

                elif pxname == 'som_opto':
                    max_FRs,_,_ = c3_ssa.get_tone_evoked_FR(
                        expt['t'],
                        expt['u2'],
                        expt['p2'],
                        expt['s2'],
                        expt['tonelist'])
                else:
                    raise ValueError('invalid domain name choice',str(pyname)+' '+str(pxname))

                # get SSA index

                if max_FRs[-1,1] > 1e-1:
                    sol[i,j] = CSI(max_FRs[0,1],max_FRs[0,1],max_FRs[-1,1],max_FRs[-1,1])
                else:
                    sol[i,j] = np.nan

                print sol[i,j],i,j

        # transpose and flip matrix to match cartesian coordinates
        sol = sol[::-1,:]
        
        np.savetxt(fname_sol,sol)
        np.savetxt(fname_x,x)
        np.savetxt(fname_y,y)
        
        return sol,x,y
    

def r3_CSI_params():
    """
    generate plot of SSA index as a function of two parameters
    """

    # start with opto terms for now

    # first pair of inputs is for the y axis, second pair of inputs is for the  x axis
    # returns x and y domains
    #sol11,x11,y11 = run_sweep(np.linspace(0,2.5,12),'wee',np.linspace(-4,5,18),'pv_opto')
    #sol12,x12,y12 = run_sweep(np.linspace(0,2.5,12),'wee',np.linspace(-4,5,18),'som_opto')

    wee_dx=.1;wee_range = np.arange(0,2+wee_dx,wee_dx)
    taud1_dx=10;taud1_range = np.arange(100,300+taud1_dx,taud1_dx)
    pv_dx=.2;pv_range=np.arange(-2,5+pv_dx,pv_dx)
    som_dx=.25;som_range=np.arange(-1.5,3+som_dx,som_dx)
    
    sol11,x11,y11 = run_sweep(pv_range,'pv_opto',wee_range,'wee')
    sol12,x12,y12 = run_sweep(som_range,'som_opto',wee_range,'wee')
    sol21,x21,y21 = run_sweep(pv_range,'pv_opto',taud1_range,'taud1')
    sol22,x22,y22 = run_sweep(som_range,'som_opto',taud1_range,'taud1')

    fig = plt.figure()
    ax11 = fig.add_subplot(221)
    ax12 = fig.add_subplot(222)
    ax21 = fig.add_subplot(223)
    ax22 = fig.add_subplot(224)

    #fig, axs = plt.subplots(2, 2,figsize=(5,5))
    #fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
    
    #im11 = axs[0,0].imshow(sol11,extent = [x11[0],x11[-1],y11[0],y11[-1]],aspect='auto')
    #im12 = axs[0,1].imshow(sol12,extent = [x12[0],x12[-1],y12[0],y12[-1]],aspect='auto')
    #im21 = axs[1,0].imshow(sol21,extent = [x21[0],x21[-1],y21[0],y21[-1]],aspect='auto')
    #im22 = axs[1,1].imshow(sol22,extent = [x22[0],x22[-1],y22[0],y22[-1]],aspect='auto')

    
    # plot control locations
    # control param values
    pars = read_pars_values_from_file('rate_models/xpp/cols3_ssa.ode')

    # add negative to be consistent with paper.
    ax11.scatter(-float(pars['pv_opto']),float(pars['wee']),color='white',zorder=2,edgecolor='black')
    ax12.scatter(-float(pars['som_opto']),float(pars['wee']),color='white',zorder=2,edgecolor='black')
    ax21.scatter(-float(pars['pv_opto']),float(pars['taud1']),color='white',zorder=2,edgecolor='black')
    ax22.scatter(-float(pars['som_opto']),float(pars['taud1']),color='white',zorder=2,edgecolor='black')

    # get approx CSI at control locations. no negative needed here
    x11_idx = np.argmin(abs(x11-float(pars['pv_opto'])));y11_idx = np.argmin(abs(y11[::-1]-float(pars['wee'])));
    x12_idx = np.argmin(abs(x12-float(pars['som_opto'])));y12_idx = np.argmin(abs(y12[::-1]-float(pars['wee'])));
    x21_idx = np.argmin(abs(x21-float(pars['pv_opto'])));y21_idx = np.argmin(abs(y21[::-1]-float(pars['taud1'])));
    x22_idx = np.argmin(abs(x22-float(pars['som_opto'])));y22_idx = np.argmin(abs(y22[::-1]-float(pars['taud1'])));
    
    print 'CSI sol11 control',sol11[y11_idx,x11_idx]
    print 'CSI sol12 control',sol12[y12_idx,x12_idx]
    print 'CSI sol21 control',sol21[y21_idx,x21_idx]
    print 'CSI sol22 control',sol22[y22_idx,x22_idx]

    # plot opto locations
    ax11.scatter(-4,float(pars['wee']),color='white',zorder=2,marker='^',edgecolor='black')
    ax12.scatter(-2,float(pars['wee']),color='white',zorder=2,marker='^',edgecolor='black')
    ax21.scatter(-4,float(pars['taud1']),color='white',zorder=2,marker='^',edgecolor='black')
    ax22.scatter(-2,float(pars['taud1']),color='white',zorder=2,marker='^',edgecolor='black')
    
    # get approx CSI at opto locations
    x11_idx_opt = np.argmin(abs(x11-4));y11_idx_opt = np.argmin(abs(y11[::-1]-float(pars['wee'])));
    x12_idx_opt = np.argmin(abs(x12-2));y12_idx_opt = np.argmin(abs(y12[::-1]-float(pars['wee'])));
    x21_idx_opt = np.argmin(abs(x21-4));y21_idx_opt = np.argmin(abs(y21[::-1]-float(pars['taud1'])));
    x22_idx_opt = np.argmin(abs(x22-2));y22_idx_opt = np.argmin(abs(y22[::-1]-float(pars['taud1'])));
    
    print 'CSI sol11 opto',sol11[y11_idx_opt,x11_idx_opt]
    print 'CSI sol12 opto',sol12[y12_idx_opt,x12_idx_opt]
    print 'CSI sol21 opto',sol21[y21_idx_opt,x21_idx_opt]
    print 'CSI sol22 opto',sol22[y22_idx_opt,x22_idx_opt]

    
    # plot opto locations
    ax11.scatter(.5,float(pars['wee']),color='white',zorder=2,marker='s',edgecolor='black')
    ax12.scatter(.5,float(pars['wee']),color='white',zorder=2,marker='s',edgecolor='black')
    ax21.scatter(.5,float(pars['taud1']),color='white',zorder=2,marker='s',edgecolor='black')
    ax22.scatter(.5,float(pars['taud1']),color='white',zorder=2,marker='s',edgecolor='black')

    # get approx CSI at opto activation locations
    x11_idx_opt_act = np.argmin(abs(x11+.5));y11_idx_opt_act = np.argmin(abs(y11[::-1]-float(pars['wee'])));
    x12_idx_opt_act = np.argmin(abs(x12+.5));y12_idx_opt_act = np.argmin(abs(y12[::-1]-float(pars['wee'])));
    x21_idx_opt_act = np.argmin(abs(x21+.5));y21_idx_opt_act = np.argmin(abs(y21[::-1]-float(pars['taud1'])));
    x22_idx_opt_act = np.argmin(abs(x22+.5));y22_idx_opt_act = np.argmin(abs(y22[::-1]-float(pars['taud1'])));
    
    print 'CSI sol11 opto act.',sol11[y11_idx_opt_act,x11_idx_opt_act]
    print 'CSI sol12 opto act.',sol12[y12_idx_opt_act,x12_idx_opt_act]
    print 'CSI sol21 opto act.',sol21[y21_idx_opt_act,x21_idx_opt_act]
    print 'CSI sol22 opto act.',sol22[y22_idx_opt_act,x22_idx_opt_act]

    
    #im11 = ax11.imshow(sol11,extent = [x11[0],x11[-1],y11[0],y11[-1]],aspect='auto')
    #im12 = ax12.imshow(sol12,extent = [x12[0],x12[-1],y12[0],y12[-1]],aspect='auto')
    #im21 = ax21.imshow(sol21,extent = [x21[0],x21[-1],y21[0],y21[-1]],aspect='auto')
    #im22 = ax22.imshow(sol22,extent = [x22[0],x22[-1],y22[0],y22[-1]],aspect='auto')

    im11 = ax11.imshow(sol11[:,::-1],extent = [-x11[-1],-x11[0],y11[0],y11[-1]],aspect='auto')
    im12 = ax12.imshow(sol12[:,::-1],extent = [-x12[-1],-x12[0],y12[0],y12[-1]],aspect='auto')
    im21 = ax21.imshow(sol21[:,::-1],extent = [-x21[-1],-x21[0],y21[0],y21[-1]],aspect='auto')
    im22 = ax22.imshow(sol22[:,::-1],extent = [-x22[-1],-x22[0],y22[0],y22[-1]],aspect='auto')

    

    
    
    
    fig.colorbar(im11,ax=ax11)
    c12 = fig.colorbar(im12,ax=ax12)
    fig.colorbar(im21,ax=ax21)
    c22 = fig.colorbar(im22,ax=ax22)

    c12.set_label('CSI Index')
    c22.set_label('CSI Index')
    
    ax11.set_title(r'\textbf{A}',x=0)
    ax12.set_title(r'\textbf{B}',x=0)
    ax21.set_title(r'\textbf{C}',x=0)
    ax22.set_title(r'\textbf{D}',x=0)

    #ax11.set_xlim(x11[0],x11[-1])
    #ax12.set_xlim(x12[0],x12[-1])
    #ax21.set_xlim(x21[0],x21[-1])
    #ax22.set_xlim(x22[0],x22[-1])

     
    #ax11.set_ylim(y11[0],y11[-1])
    #ax12.set_ylim(y12[0],y12[-1])
    #ax21.set_ylim(y21[0],y21[-1])
    #ax22.set_ylim(y22[0],y22[-1])

    #print y11[0]
    #'$I_\text{Opt,SOM}=1.2$'

    ax11.set_ylabel(r'$w_{ee}$')
    ax21.set_ylabel(r'$\tau_{d_1}$')
    ax21.set_xlabel(r'$I_\text{Opt,PV}$')
    ax22.set_xlabel(r'$I_\text{Opt,SOM}$')
        
    plt.tight_layout()

    return fig




def run_s3_sweep(px,pxname,py,pyname,recompute=False):
    """
    first pair of inputs is for the y axis, second pair of inputs is for the  x axis

    inputs in terms of cartesan coordinates.
    
    returns CSI matrix and corresponding x and y domains
    """

    x = px
    y = py

    # check if data directory exists and create if it does not.
    if not(os.path.isdir("dat")):
        os.mkdir("dat")

    paradigm = 'ssa'
        
    fname_sol = "dat/"+pxname+'_'+pyname+'_'+str(len(x))+'_'+str(len(y))+\
                'xlo='+str(x[0])+\
                'xhi='+str(x[-1])+\
                'ylo='+str(y[0])+\
                'yhi='+str(y[-1])+\
                'sol_s3.dat'
    
    fname_x = "dat/"+pxname+'_'+pyname+'_'+str(len(x))+'_'+str(len(y))+\
              'xlo='+str(x[0])+\
              'xhi='+str(x[-1])+\
              'ylo='+str(y[0])+\
              'yhi='+str(y[-1])+\
              'x_s3.dat'
    
    fname_y = "dat/"+pxname+'_'+pyname+'_'+str(len(x))+'_'+str(len(y))+\
              'xlo='+str(x[0])+\
              'xhi='+str(x[-1])+\
              'ylo='+str(y[0])+\
              'yhi='+str(y[-1])+\
              'y_s3.dat'
    
    # check if data exists for chosen parameters.
    if os.path.isfile(fname_sol) and os.path.isfile(fname_x) and os.path.isfile(fname_y) and not(recompute):
        return np.loadtxt(fname_sol),np.loadtxt(fname_x),np.loadtxt(fname_y)
    
    else:
        fname = 'rate_models/xpp/cols3_ssa.ode'
        inits = read_init_values_from_file(fname)

        sol = np.zeros((len(py),len(px)))

        #pars = {}
        pars = {'wee':1,'taud1':1000}

        for i in range(len(py)):
            for j in range(len(px)):
                pars[pyname] = py[i]

                if pxname == 'pv_opto':
                    expt = s3.setup_and_run(pv_opto=x[j],paradigm=paradigm,pars=pars)
                if pxname == 'som_opto':
                    expt = s3.setup_and_run(som_opto=x[j],paradigm=paradigm,pars=pars)
                
                #print expt['T']
                max_FRs,_,_ = s3.collect_FR_dev(
                    expt['stim_arr1']+expt['stim_arr3'],
                    expt['stim_dt'],
                    expt['defaultclock'],
                    expt['spikemon_PYR2'],expt['n_pyr'])

                # get relevant portion
                max_FRs = max_FRs[4:]

                #print max_FRs

                # get SSA index

                if max_FRs[-1] > 1e-1:
                    sol[i,j] = CSI(max_FRs[0],max_FRs[0],max_FRs[-1],max_FRs[-1])
                else:
                    sol[i,j] = np.nan

                print sol[i,j],i,j,100*(1.*i*len(px)+j)/(len(px)*len(py)),"%"

        # transpose and flip matrix to match cartesian coordinates
        sol = sol[::-1,:]
        
        np.savetxt(fname_sol,sol)
        np.savetxt(fname_x,x)
        np.savetxt(fname_y,y)
        
        return sol,x,y


def s3_CSI_params():
    """
    CSI parameter sweep for the spiking model
    """

    wee_dx=.2#.1
    taud1_dx=200#10
    pv_dx=.1#.2
    som_dx=.4#.25
    
    wee_range = np.arange(0,2+wee_dx,wee_dx)
    taud1_range = np.arange(600,3000+taud1_dx,taud1_dx)
    pv_range=np.arange(-.5,.5+pv_dx,pv_dx)
    som_range=np.arange(-2,2+som_dx,som_dx)
    
    sol11,x11,y11 = run_s3_sweep(pv_range,'pv_opto',wee_range,'wee')
    sol12,x12,y12 = run_s3_sweep(som_range,'som_opto',wee_range,'wee')
    sol21,x21,y21 = run_s3_sweep(pv_range,'pv_opto',taud1_range,'taud1')
    sol22,x22,y22 = run_s3_sweep(som_range,'som_opto',taud1_range,'taud1')

    fig = plt.figure()
    ax11 = fig.add_subplot(221)
    ax12 = fig.add_subplot(222)
    ax21 = fig.add_subplot(223)
    ax22 = fig.add_subplot(224)

    #fig, axs = plt.subplots(2, 2,figsize=(5,5))
    #fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
    
    #im11 = axs[0,0].imshow(sol11,extent = [x11[0],x11[-1],y11[0],y11[-1]],aspect='auto')
    #im12 = axs[0,1].imshow(sol12,extent = [x12[0],x12[-1],y12[0],y12[-1]],aspect='auto')
    #im21 = axs[1,0].imshow(sol21,extent = [x21[0],x21[-1],y21[0],y21[-1]],aspect='auto')
    #im22 = axs[1,1].imshow(sol22,extent = [x22[0],x22[-1],y22[0],y22[-1]],aspect='auto')

    
    # plot control locations
    # control param values
    #pars = read_pars_values_from_file('rate_models/xpp/cols3_ssa.ode')
    pars = {'pv_opto':0,'som_opto':0,'wee':1.,'taud1':1000}

    ax11.scatter(-float(pars['pv_opto']),float(pars['wee']),color='white',zorder=2,edgecolor='black')
    ax12.scatter(-float(pars['som_opto']),float(pars['wee']),color='white',zorder=2,edgecolor='black')
    ax21.scatter(-float(pars['pv_opto']),float(pars['taud1']),color='white',zorder=2,edgecolor='black')
    ax22.scatter(-float(pars['som_opto']),float(pars['taud1']),color='white',zorder=2,edgecolor='black')

    # get approx CSI at control locations
    x11_idx = np.argmin(abs(x11-float(pars['pv_opto'])));y11_idx = np.argmin(abs(y11[::-1]-float(pars['wee'])));
    x12_idx = np.argmin(abs(x12-float(pars['som_opto'])));y12_idx = np.argmin(abs(y12[::-1]-float(pars['wee'])));
    x21_idx = np.argmin(abs(x21-float(pars['pv_opto'])));y21_idx = np.argmin(abs(y21[::-1]-float(pars['taud1'])));
    x22_idx = np.argmin(abs(x22-float(pars['som_opto'])));y22_idx = np.argmin(abs(y22[::-1]-float(pars['taud1'])));
    
    print 'CSI sol11 control',sol11[y11_idx,x11_idx]
    print 'CSI sol12 control',sol12[y12_idx,x12_idx]
    print 'CSI sol21 control',sol21[y21_idx,x21_idx]
    print 'CSI sol22 control',sol22[y22_idx,x22_idx]

    # plot opto locations
    ax11.scatter(-.2,float(pars['wee']),color='white',zorder=2,marker='^',edgecolor='black')
    ax12.scatter(-1,float(pars['wee']),color='white',zorder=2,marker='^',edgecolor='black')
    ax21.scatter(-.2,float(pars['taud1']),color='white',zorder=2,marker='^',edgecolor='black')
    ax22.scatter(-1,float(pars['taud1']),color='white',zorder=2,marker='^',edgecolor='black')
    
    # get approx CSI at opto locations
    x11_idx_opt = np.argmin(abs(x11-1));y11_idx_opt = np.argmin(abs(y11[::-1]-float(pars['wee'])));
    x12_idx_opt = np.argmin(abs(x12-.2));y12_idx_opt = np.argmin(abs(y12[::-1]-float(pars['wee'])));
    x21_idx_opt = np.argmin(abs(x21-1));y21_idx_opt = np.argmin(abs(y21[::-1]-float(pars['taud1'])));
    x22_idx_opt = np.argmin(abs(x22-.2));y22_idx_opt = np.argmin(abs(y22[::-1]-float(pars['taud1'])));
    
    print 'CSI sol11 opto',sol11[y11_idx_opt,x11_idx_opt]
    print 'CSI sol12 opto',sol12[y12_idx_opt,x12_idx_opt]
    print 'CSI sol21 opto',sol21[y21_idx_opt,x21_idx_opt]
    print 'CSI sol22 opto',sol22[y22_idx_opt,x22_idx_opt]

    
    # plot opto locations
    ax11.scatter(.2,float(pars['wee']),color='white',zorder=2,marker='s',edgecolor='black')
    ax12.scatter(.5,float(pars['wee']),color='white',zorder=2,marker='s',edgecolor='black')
    ax21.scatter(.2,float(pars['taud1']),color='white',zorder=2,marker='s',edgecolor='black')
    ax22.scatter(.5,float(pars['taud1']),color='white',zorder=2,marker='s',edgecolor='black')

    # get approx CSI at opto activation locations
    x11_idx_opt_act = np.argmin(abs(x11+.5));y11_idx_opt_act = np.argmin(abs(y11[::-1]-float(pars['wee'])));
    x12_idx_opt_act = np.argmin(abs(x12+.25));y12_idx_opt_act = np.argmin(abs(y12[::-1]-float(pars['wee'])));
    x21_idx_opt_act = np.argmin(abs(x21+.5));y21_idx_opt_act = np.argmin(abs(y21[::-1]-float(pars['taud1'])));
    x22_idx_opt_act = np.argmin(abs(x22+.25));y22_idx_opt_act = np.argmin(abs(y22[::-1]-float(pars['taud1'])));

    print 'y21_idx_opt_act',y21_idx_opt_act
    print 'y22_idx_opt_act',y22_idx_opt_act
    
    print 'CSI sol11 opto act.',sol11[y11_idx_opt_act,x11_idx_opt_act]
    print 'CSI sol12 opto act.',sol12[y12_idx_opt_act,x12_idx_opt_act]
    print 'CSI sol21 opto act.',sol21[y21_idx_opt_act,x21_idx_opt_act]
    print 'CSI sol22 opto act.',sol22[y22_idx_opt_act,x22_idx_opt_act]

    
    im11 = ax11.imshow(sol11[:,::-1],extent = [x11[-1],x11[0],y11[0],y11[-1]],aspect='auto')
    im12 = ax12.imshow(sol12[:,::-1],extent = [x12[-1],x12[0],y12[0],y12[-1]],aspect='auto')
    im21 = ax21.imshow(sol21[:,::-1],extent = [x21[-1],x21[0],y21[0],y21[-1]],aspect='auto')
    im22 = ax22.imshow(sol22[:,::-1],extent = [x22[-1],x22[0],y22[0],y22[-1]],aspect='auto')

    
    
    fig.colorbar(im11,ax=ax11)
    c12 = fig.colorbar(im12,ax=ax12)
    fig.colorbar(im21,ax=ax21)
    c22 = fig.colorbar(im22,ax=ax22)

    c12.set_label('CSI Index')
    c22.set_label('CSI Index')

    ax11.set_title(r'\textbf{A}',x=0)
    ax12.set_title(r'\textbf{B}',x=0)
    ax21.set_title(r'\textbf{C}',x=0)
    ax22.set_title(r'\textbf{D}',x=0)

    #ax11.set_xlim(x11[0],x11[-1])
    #ax12.set_xlim(x12[0],x12[-1])
    #ax21.set_xlim(x21[0],x21[-1])
    #ax22.set_xlim(x22[0],x22[-1])

     
    #ax11.set_ylim(y11[0],y11[-1])
    #ax12.set_ylim(y12[0],y12[-1])
    #ax21.set_ylim(y21[0],y21[-1])
    #ax22.set_ylim(y22[0],y22[-1])

    #print y11[0]


    ax11.set_ylabel(r'$w_{ee}$')
    ax21.set_ylabel(r'$\tau_{d_1}$')

    ax21.set_xlabel(r'$I_\text{Opt,PV}$')
    ax22.set_xlabel(r'$I_\text{Opt,SOM}$')

    #ax21.set_xlabel('PV Opto')
    #ax22.set_xlabel('SOM Opto')
    
    
    
    plt.tight_layout()

    plt.show()
    

    return fig




def r3_s3_tuning_full(recompute=False):

    fname = 'rate_models/xpp/cols3_tuning.ode'
    pars = read_pars_values_from_file(fname)
    inits = read_init_values_from_file(fname)

    ############################################## pv opto varying sim of rate model


    ################################################## direct sim for rate model

    # repeat simulation for different frequencies.
    # pars['mode'] = '1' makes the first tone appear in the first column
    # pars['mode'] = '2' makes the first tone appear in the second column.. etc.

    results = np.zeros((3,10))

    #pv_o_par = 2
    pv_o_par = .5
    for i in range(3):
        pars['mode']=str(i+1)

        pars['pv_opto']=0
        pars['som_opto']=0

        # returns tuple (t,u,v1,v2,inits,parameters,tonelist)    
        control = c3_fs.run_experiment(
            fname,
            pars,inits,
            return_all=True)

        pars['pv_opto']=pv_o_par

        pv_off = c3_fs.run_experiment(
            fname,
            pars,inits,
            return_all=True)

        pars['pv_opto']=-1.2
        
        pv_on = c3_fs.run_experiment(
            fname,
            pars,inits,
            return_all=True)

        pars['pv_opto']=0
        #pars['som_opto']=2
        pars['som_opto']=1

        som_off = c3_fs.run_experiment(
            fname,
            pars,inits,
            return_all=True)

        pars['som_opto']=-.1

        som_on = c3_fs.run_experiment(
            fname,
            pars,inits,
            return_all=True)

        
        # get first max u (i), second max u (u2).
        # get first max u with pv opto (i), second max u with pv opto (u2).
        # get first max u with som opto (i), second max u with som opto (u2).

        # get tone list stard and end index for first and second tones
        # get tone list times
        
        tone1On,tone1Off = control['tonelist'][0]
        tone5On,tone5Off = control['tonelist'][4]
        
        idx1_start = np.argmin(np.abs(control['t']-tone1On))+1 # first time interval index
        idx1_end = np.argmin(np.abs(control['t']-tone1Off))-1

        idx2_start = np.argmin(np.abs(control['t']-tone5On))+1 # last time interval index
        idx2_end = np.argmin(np.abs(control['t']-tone5Off))-1
        
        # get first tone response at u2
        control1 = c3_fs.get_max_FR(control['u2'],idx1_start,idx1_end)
        pv1 = c3_fs.get_max_FR(pv_off['u2'],idx1_start,idx1_end)
        som1 = c3_fs.get_max_FR(som_off['u2'],idx1_start,idx1_end)

        pv1_on = c3_fs.get_max_FR(pv_on['u2'],idx1_start,idx1_end)
        som1_on = c3_fs.get_max_FR(som_on['u2'],idx1_start,idx1_end)
        
        # get last tone tone response at u2
        control2 = c3_fs.get_max_FR(control['u2'],idx2_start,idx2_end)
        pv2 = c3_fs.get_max_FR(pv_off['u2'],idx2_start,idx2_end)
        som2 = c3_fs.get_max_FR(som_off['u2'],idx2_start,idx2_end)

        pv2_on = c3_fs.get_max_FR(pv_on['u2'],idx2_start,idx2_end)
        som2_on = c3_fs.get_max_FR(som_on['u2'],idx2_start,idx2_end)
        
        results[i,:] = [control1,control2,pv1,pv2,som1,som2,pv1_on,pv2_on,som1_on,som2_on]


    ################################################ spiking
    
    seedlist = np.arange(0,2,1)
    
    # check if saved results exist
    fname_ctrl1 = "dat/tuning_ctrl1_seedlen="+str(len(seedlist))+".dat"
    fname_pv1 = "dat/tuning_pv1_seedlen="+str(len(seedlist))+".dat"
    fname_som1 = "dat/tuning_som1_seedlen="+str(len(seedlist))+".dat"

    fname_pv1_on = "dat/tuning_pv1_on_seedlen="+str(len(seedlist))+".dat"
    fname_som1_on = "dat/tuning_som1_on_seedlen="+str(len(seedlist))+".dat"

    
    fname_ctrl2 = "dat/tuning_ctrl2_seedlen="+str(len(seedlist))+".dat"
    fname_pv2 = "dat/tuning_pv2_seedlen="+str(len(seedlist))+".dat"
    fname_som2 = "dat/tuning_som2_seedlen="+str(len(seedlist))+".dat"

    fname_pv2_on = "dat/tuning_pv2_on_seedlen="+str(len(seedlist))+".dat"
    fname_som2_on = "dat/tuning_som2_on_seedlen="+str(len(seedlist))+".dat"    

    if os.path.isfile(fname_ctrl1) and os.path.isfile(fname_pv1) and os.path.isfile(fname_som1) and \
       os.path.isfile(fname_ctrl2) and os.path.isfile(fname_pv2) and os.path.isfile(fname_som2) and \
       os.path.isfile(fname_pv2_on) and os.path.isfile(fname_som2_on) and \
       not(recompute):
        
        results_ctrl1 = np.loadtxt(fname_ctrl1)
        results_pv1 = np.loadtxt(fname_pv1)
        results_som1 = np.loadtxt(fname_som1)

        results_pv1_on = np.loadtxt(fname_pv1_on)
        results_som1_on = np.loadtxt(fname_som1_on)

        results_ctrl2 = np.loadtxt(fname_ctrl2)
        results_pv2 = np.loadtxt(fname_pv2)
        results_som2 = np.loadtxt(fname_som2)

        results_pv2_on = np.loadtxt(fname_pv2_on)
        results_som2_on = np.loadtxt(fname_som2_on)
        
    else:
        results_ctrl1 = np.zeros((len(seedlist),3))
        results_pv1 = np.zeros((len(seedlist),3))
        results_som1 = np.zeros((len(seedlist),3))

        results_pv1_on = np.zeros((len(seedlist),3))
        results_som1_on = np.zeros((len(seedlist),3))

        results_ctrl2 = np.zeros((len(seedlist),3))
        results_pv2 = np.zeros((len(seedlist),3))
        results_som2 = np.zeros((len(seedlist),3))

        results_pv2_on = np.zeros((len(seedlist),3))
        results_som2_on = np.zeros((len(seedlist),3))

        j = 0
        for seed in seedlist:
            for i in range(3):
                
                paradigm = 'tuning'+str(i+1)
                
                fulldict1 = s3.setup_and_run(paradigm=paradigm,seed=seed)
                
                #fulldict2 = s3.setup_and_run(pv_opto=.2,paradigm=paradigm,seed=seed)
                #fulldict3 = s3.setup_and_run(som_opto=1.,paradigm=paradigm,seed=seed)
                
                #fulldict2 = s3.setup_and_run(pv_opto=.05,paradigm=paradigm,seed=seed)
                fulldict2 = s3.setup_and_run(pv_opto=1,paradigm=paradigm,seed=seed)
                fulldict3 = s3.setup_and_run(som_opto=1.,paradigm=paradigm,seed=seed)

                fulldict2On = s3.setup_and_run(pv_opto=-.3,paradigm=paradigm,seed=seed)
                fulldict3On = s3.setup_and_run(som_opto=-.1,paradigm=paradigm,seed=seed)

                if (seed == 0) and (i==1) and (False):
                    M2 = fulldict2['M_PYR2']
                    
                    fig = plt.figure()
                    ax1 = fig.add_subplot(211)
                    ax2 = fig.add_subplot(212)
                    ax1.plot((1-1.7*(1-M2.DSlow2[0])))
                    plt.show()
                    plt.close()

                if (seed == 0) and (i==1) and (False):
                    fig = plt.figure()
                    
                    ax = fig.add_subplot(111)
                    ax.scatter(fulldict1['spikemon_PYR2'].t/b2.ms, fulldict1['spikemon_PYR2'].i+1,color='k')
                    ax.plot(fulldict1['M_PYR2'].t/b2.ms, fulldict1['M_PYR2'].thal2[0]/b2.nA,color='k')
                    
                    plt.show()
                    plt.close()
                
                # get rates
                FR_control_pyr = s3.collect_FR(fulldict1['stim_arr'+str(i+1)],
                                               fulldict1['stim_dt'],
                                               fulldict1['defaultclock'],
                                               fulldict1['spikemon_PYR2'].t,
                                               fulldict1['n_pyr'])

                print fulldict1['stim_arr'+str(i+1)]
                
                FR_pv_pyr = s3.collect_FR(fulldict2['stim_arr'+str(i+1)],
                                          fulldict2['stim_dt'],
                                          fulldict2['defaultclock'],
                                          fulldict2['spikemon_PYR2'].t,
                                          fulldict2['n_pyr'])

                FR_som_pyr = s3.collect_FR(fulldict3['stim_arr'+str(i+1)],
                                           fulldict3['stim_dt'],
                                           fulldict3['defaultclock'],
                                           fulldict3['spikemon_PYR2'].t,
                                           fulldict3['n_pyr'])
                
                FR_pv_pyr_on = s3.collect_FR(fulldict2On['stim_arr'+str(i+1)],
                                           fulldict2On['stim_dt'],
                                           fulldict2On['defaultclock'],
                                           fulldict2On['spikemon_PYR2'].t,
                                           fulldict2On['n_pyr'])
                
                FR_som_pyr_on = s3.collect_FR(fulldict3On['stim_arr'+str(i+1)],
                                           fulldict3On['stim_dt'],
                                           fulldict3On['defaultclock'],
                                           fulldict3On['spikemon_PYR2'].t,
                                           fulldict3On['n_pyr'])

                print 'paradigm',paradigm,'FR_control_pyr',FR_control_pyr
                control1 = FR_control_pyr[0]
                control2 = FR_control_pyr[-1]

                pv1 = FR_pv_pyr[0]
                pv2 = FR_pv_pyr[-1]

                pv1_on = FR_pv_pyr_on[0]
                pv2_on = FR_pv_pyr_on[-1]

                som1 = FR_som_pyr[0]
                som2 = FR_som_pyr[-1]

                som1_on = FR_som_pyr_on[0]
                som2_on = FR_som_pyr_on[-1]

                #results_spike[i,:] = [control1,control2,pv1,pv2,som1,som2]
                
                #results[int(suffix)-1,:] = [control,pv_off,som_off]
                results_ctrl1[j,i] = control1
                results_pv1[j,i] = pv1
                results_som1[j,i] = som1

                results_pv1_on[j,i] = pv1_on
                results_som1_on[j,i] = som1_on

                results_ctrl2[j,i] = control2
                results_pv2[j,i] = pv2
                results_som2[j,i] = som2
                
                results_pv2_on[j,i] = pv2_on
                results_som2_on[j,i] = som2_on
                
            j += 1

        np.savetxt(fname_ctrl1,results_ctrl1)
        np.savetxt(fname_pv1,results_pv1)
        np.savetxt(fname_som1,results_som1)

        np.savetxt(fname_pv1_on,results_pv1_on)
        np.savetxt(fname_som1_on,results_som1_on)

        np.savetxt(fname_ctrl2,results_ctrl2)
        np.savetxt(fname_pv2,results_pv2)
        np.savetxt(fname_som2,results_som2)
        
        np.savetxt(fname_pv2_on,results_pv2_on)
        np.savetxt(fname_som2_on,results_som2_on)

        
    fig = plt.figure(figsize=(8,6))

    ax11 = fig.add_subplot(331)
    ax12 = fig.add_subplot(332)
    ax13 = fig.add_subplot(333)
    #ax14 = fig.add_subplot(344)

    ax21 = fig.add_subplot(334)
    ax22 = fig.add_subplot(335)
    ax23 = fig.add_subplot(336)
    #ax24 = fig.add_subplot(348)

    #fig2 = 
    ax31 = fig.add_subplot(337)
    ax32 = fig.add_subplot(338)
    ax33 = fig.add_subplot(339)
    #ax34 = fig.add_subplot(3,4,12)

    #ax31 = fig.add_subplot(325)
    #ax32 = fig.add_subplot(326)
    
    bar_width = 0.2

    #ax11.set_title('Normalized Peak Response (2nd Tone)')

    x = np.array([-1,0,1])
    
    #control_probe = results[1,0]
    control_probe = 1
    ax11.plot(x,results[:,0]/control_probe,label='Control',color='black',marker='o')


    def f(x,th=0,mx=.7):
        
        #x-=th
        
        #x[x>=mx]=mx
        #x[x<=0]=0
        
        return x


    #pars['pv_opto']

    mx = np.amax(results[:,0])
    
    #pv_probe = results[1,2]
    pv_probe = 1
    ax11.plot(x,f(results[:,2]/pv_probe,mx=mx),label='PV Inact.',color=pv_color,marker='o',ls='--')
    ax11.plot(x,f(results[:,4]/pv_probe),label='SOM Inact.',color=som_color,marker='o')
    
    if False:
        fig2 = plt.figure()
        ax11_2 = fig2.add_subplot(111)
        ax11_2.plot([0,np.amax(results[:,0])],[0,np.amax(results[:,0])],c='k')
        ax11_2.plot(results[:,0],results[:,1])
        plt.show()

    ax13.plot(x,results[:,0]/control_probe,label='Control',color='black',marker='o')
    ax13.plot(x,results[:,6]/pv_probe,label='PV Act.',color=pv_color,marker='o',ls='--')
    ax13.plot(x,results[:,8]/pv_probe,label='SOM Act.',color=som_color,marker='o')

    #ax12.set_title('Normalized Peak Response (2nd Tone)')

    control_probe = 1#results[1,0]    
    ax21.plot(x,results[:,1]/control_probe,color='black',marker='o')

    som_probe = 1#results[1,4]
    ax21.plot(x,f(results[:,3],mx=mx),color=pv_color,marker='o',ls='--') # PV inact after adaptation
    ax21.plot(x,results[:,5]/som_probe,color=som_color,marker='o') # SOM inact after adaptation

    
    ax23.plot(x,results[:,1]/control_probe,color='black',marker='o')

    som_probe = 1#results[1,4]
    ax23.plot(x,results[:,7]/som_probe,color=pv_color,marker='o',ls='--')
    ax23.plot(x,results[:,9]/som_probe,color=som_color,marker='o')

    ctrl1_mean = np.mean(results_ctrl1,axis=0)
    ctrl1_std = np.std(results_ctrl1,axis=0)

    ctrl2_mean = np.mean(results_ctrl2,axis=0)
    ctrl2_std = np.std(results_ctrl2,axis=0)


    pv1_mean = np.mean(results_pv1,axis=0)
    pv1_std = np.std(results_pv1,axis=0)

    pv2_mean = np.mean(results_pv2,axis=0)
    pv2_std = np.std(results_pv2,axis=0)


    
    pv1_mean_on = np.mean(results_pv1_on,axis=0)
    pv1_std_on = np.std(results_pv1_on,axis=0)

    pv2_mean_on = np.mean(results_pv2_on,axis=0)
    pv2_std_on = np.std(results_pv2_on,axis=0)

    
    som1_mean = np.mean(results_som1,axis=0)
    som1_std = np.std(results_som1,axis=0)
    
    som2_mean = np.mean(results_som2,axis=0)
    som2_std = np.std(results_som2,axis=0)

    som1_mean_on = np.mean(results_som1_on,axis=0)
    som1_std_on = np.std(results_som1_on,axis=0)
    
    som2_mean_on = np.mean(results_som2_on,axis=0)
    som2_std_on = np.std(results_som2_on,axis=0)

    
    ########################################## spike plots
    ax12.plot(x,ctrl1_mean,c='k',marker='o',label='Control')
    ax12.errorbar(x,ctrl1_mean,yerr=ctrl1_std,c='k')
    
    ax12.plot(x,pv1_mean,c=pv_color,marker='o',ls='--',label='PV Inact.')
    ax12.errorbar(x,pv1_mean,yerr=pv1_std,c=pv_color,ls='--')
    
    ax12.plot(x,som1_mean,c=som_color,marker='o',label='SOM Inact.')
    ax12.errorbar(x,som1_mean,yerr=som1_std,c=som_color)


        
    if False:
        fig2 = plt.figure()
        ax11_2 = fig2.add_subplot(111)
        ax11_2.plot([np.amin(ctrl1_mean),np.amax(ctrl1_mean)],[np.amin(ctrl1_mean),np.amax(ctrl1_mean)])
        ax11_2.plot(ctrl1_mean,ctrl2_mean)
        plt.show()


    #ax14.errorbar(freqs,mean_som,yerr=std_som,label='SOM Off 2nd Tone',color=som_color,
    #              lw=1, capsize=1, capthick=1,zorder=3) # plot SOM off over all tones


    ax22.plot(x,ctrl2_mean,c='k',marker='o')
    ax22.errorbar(x,ctrl2_mean,yerr=ctrl2_std,c='k')

    ax22.plot(x,pv2_mean,c=pv_color,marker='o',ls='--')
    ax22.errorbar(x,pv2_mean,yerr=pv2_std,c=pv_color,ls='--')    

    ax22.plot(x,som2_mean,c=som_color,marker='o')
    ax22.errorbar(x,som2_mean,yerr=som2_std,c=som_color)


    
    #ax23.plot(x,ctrl1_mean,c='k',marker='o')
    #ax23.errorbar(x,ctrl1_mean,yerr=ctrl2_std,c='k')
    
    #ax23.plot(x,pv1_mean_on,c=pv_color,marker='o',ls='--')
    #ax23.errorbar(x,pv1_mean_on,yerr=pv1_std_on,c=pv_color,ls='--')
    
    #ax23.plot(x,som1_mean_on,c=som_color,marker='o')
    #ax23.errorbar(x,som1_mean_on,yerr=som1_std_on,c=som_color)



    #ax24.plot(x,ctrl2_mean,c='k',marker='o')
    #ax24.errorbar(x,ctrl2_mean,yerr=ctrl2_std,c='k')
    
    #ax24.plot(x,pv2_mean_on,c=pv_color,marker='o',ls='--')
    #ax24.errorbar(x,pv2_mean_on,yerr=pv2_std_on,c=pv_color,ls='--')    

    #ax24.plot(x,som2_mean_on,c=som_color,marker='o')
    #ax24.errorbar(x,som2_mean_on,yerr=som2_std_on,c=som_color)


    #ax31.set_title(r'\textbf{E} Rate Model',x=0)
    #ax31.plot(results[:,2]-results[:,0],results[:,0]-results[:,1])
    #ax31.plot(results[:,3]-results[:,1],results[:,0]-results[:,1])

    #ax32.set_title(r'\textbf{F} Spiking Model',x=0)
    #ax32.plot(pv1_mean-ctrl1_mean,ctrl1_mean-ctrl2_mean)
    #ax32.plot(pv2_mean-ctrl2_mean,ctrl1_mean-ctrl2_mean)

    
    #print ctrl2_mean,ctrl2_std


    ############################################################# ratio plots (rate)
    #ax31.plot([np.amin(results[:,0]),np.amax(results[:,0])],[np.amin(results[:,0]),np.amax(results[:,0])],color='k')
    ax31.plot([0,np.amax(results[:,0])],[0,np.amax(results[:,0])],color='k')
    ax31.plot(results[:,0][:2],results[:,2][:2],label='PV Inact.',color=pv_color,lw=3,ls='--')
    ax31.plot(results[:,0][:2],results[:,4][:2],label='SOM Inact.',color=som_color,lw=3)

    xr31_pv = np.linspace(0,np.amin(results[:,0][:2]))
    ### linear fits
    p31_pv = np.polyfit(results[:,0][:2],results[:,2][:2],1)
    ax31.plot(xr31_pv,p31_pv[0]*xr31_pv+p31_pv[1],color=pv_color,ls='--',alpha=0.6)

    xr31_som = np.linspace(0,np.amin(results[:,0][:2]))
    ### linear fits
    p31_som = np.polyfit(results[:,0][:2],results[:,4][:2],1)
    ax31.plot(xr31_som,p31_som[0]*xr31_som+p31_som[1],color=som_color,alpha=0.6)
    
    #print p31_pv,p31_som


    ax33.plot([0,np.amax(results[:,0])],[0,np.amax(results[:,0])],color='k')
    ax33.plot(results[:,0][:2],results[:,6][:2],label='PV Act.',color=pv_color,lw=3,ls='--')
    ax33.plot(results[:,0][:2],results[:,8][:2],label='SOM Act.',color=som_color,lw=3)
    
    xr33_pv = np.linspace(0,np.amin(results[:,0][:2]))
    ### linear fits
    p33_pv = np.polyfit(results[:,0][:2],results[:,6][:2],1)
    ax33.plot(xr33_pv,p33_pv[0]*xr33_pv+p33_pv[1],color=pv_color,ls='--',alpha=0.6)

    xr33_som = np.linspace(0,np.amin(results[:,0][:2]))
    ### linear fits
    p33_som = np.polyfit(results[:,0][:2],results[:,8][:2],1)

    #print results[:,8][:2],np.amin(results[:,8][:2])
    ax33.plot(xr33_som,p33_som[0]*xr33_som+p33_som[1],color=som_color,alpha=0.6)

    
    ############################################################# ratio plots (spike)

    ax32.plot([0,np.amax(ctrl1_mean)],[0,np.amax(ctrl1_mean)],color='k')
    ax32.plot(ctrl1_mean[:2],pv1_mean[:2],label='PV Inact.',color=pv_color,lw=3,ls='--')
    ax32.plot(ctrl1_mean[:2],som1_mean[:2],label='SOM Inact.',color=som_color,lw=3)


        
    xr32_pv = np.linspace(0,np.amin(ctrl1_mean))
    ### linear fits
    p32_pv = np.polyfit(ctrl1_mean[:2],pv1_mean[:2],1)
    ax32.plot(xr32_pv,p32_pv[0]*xr32_pv+p32_pv[1],color=pv_color,ls='--',alpha=0.6)

    xr32_som = np.linspace(0,np.amin(ctrl1_mean))
    ### linear fits
    p32_som = np.polyfit(ctrl1_mean[:2],som1_mean[:2],1)
    ax32.plot(xr32_som,p32_som[0]*xr32_som+p32_som[1],color=som_color,alpha=0.6)
    #ax34.plot([np.amin(ctrl1_mean),np.amax(ctrl1_mean)],[np.amin(ctrl1_mean),np.amax(ctrl1_mean)],ls='--',color='k')
    #ax34.plot(ctrl1_mean[:2],pv1_mean_on[:2],label='PV Act.',color=pv_color,lw=2,ls='--')
    #ax34.plot(ctrl1_mean[:2],som1_mean_on[:2],label='SOM Act.',color=som_color,lw=2)
    

    
    ax21.set_xlabel('Distance from Preferred Frequency')
    ax22.set_xlabel('Distance from Preferred Frequency')
    ax23.set_xlabel('Distance from Preferred Frequency')


    ax31.set_xlabel('Light Off')
    ax32.set_xlabel('Light Off')
    ax33.set_xlabel('Light Off')

    
    
    #ax24.set_xlabel('Distance from Preferred Frequency')

    #ax31.set_xlabel('Light Off')
    #ax32.set_xlabel('Light Off')
    
    
    ax11.set_ylabel('Pyr FR (Before Adapt.)')
    ax21.set_ylabel('Pyr FR (After Adapt.)')
    

    ax31.set_ylabel('Light On')
    ax32.set_ylabel('Light On')
    ax33.set_ylabel('Light On')
    
    #ax31.set_ylabel('Light On')
    #ax32.set_ylabel('Light On')
    
    ax11.set_title(r'\textbf{A} Rate Model',x=.3)
    ax12.set_title(r'\textbf{B} Spiking Model',x=.3)
    ax13.set_title(r'\textbf{C} Rate Model (Prediction)',x=.5)

    ax31.set_title(r'\textbf{D}',x=0)
    ax32.set_title(r'\textbf{E}',x=0)
    ax33.set_title(r'\textbf{F}',x=0)

    #ax31.set_title(r'\textbf{E} ',x=.0)
    #ax32.set_title(r'\textbf{F} ',x=.0)
    #ax33.set_title(r'\textbf{G} ',x=.0)
    #ax34.set_title(r'\textbf{H} ',x=.0)

    ax31.spines['top'].set_visible(False)
    ax31.spines['right'].set_visible(False)

    ax32.spines['top'].set_visible(False)
    ax32.spines['right'].set_visible(False)

    ax33.spines['top'].set_visible(False)
    ax33.spines['right'].set_visible(False)
    
    
    ax11.set_ylim(0,.45)
    ax21.set_ylim(0,.45)

    ax13.set_ylim(-.1,.5)
    ax23.set_ylim(-.1,.5)

    ax12.set_ylim(1,10.5)
    ax22.set_ylim(1,10.5)

    #ax23.set_ylim(0,11)
    #ax24.set_ylim(0,11)
        
    ax11.set_xticks(x)
    ax12.set_xticks(x)
    ax13.set_xticks(x)
    #ax14.set_xticks(x)

    ax11.legend(loc='lower center',prop={'size': 8})
    ax12.legend(loc='lower center',prop={'size': 8})
    ax13.legend(loc='lower center',prop={'size': 8})
    #ax21.legend()
    
    #ax31.legend()
    #ax32.legend()
    #ax33.legend()
    #ax34.legend()
    
    #ax12.legend()
    
    #plt.tight_layout()
    #plt.show()

    return fig


def attentional():

    heav = np.heaviside


    T = 30000
    dt = 1
    t = np.linspace(0,T-dt,int(T/dt))



    def dan(x,tl=1):
        return np.exp(-tl*x)

    def gt(x,ton,toff):
        return heav(x-ton,1)*heav(toff-x,1)*dan(x-ton)


    #################### SSA
    t0=100;dur=100;isi_ssa=300

    t_on_ssa = np.zeros(80)
    t_off_ssa = np.zeros(len(t_on_ssa))

    t_on_ssa[0] = t0
    t_off_ssa[0] = t_on_ssa[0]+dur

    stim_ssa = np.zeros(len(t))

    stim_ssa += gt(t,t_on_ssa[0],t_off_ssa[0])

    for i in range(1,len(t_on_ssa)):
        t_on_ssa[i] = t_off_ssa[i-1]+isi_ssa
        t_off_ssa[i] = t_on_ssa[i]+dur

        j = np.argmin(np.abs(t-t_on_ssa[i]))
        stim_ssa[j:]+= gt(t[j:],t_on_ssa[i],t_off_ssa[i])


    def i_ssa(x):
        x = np.mod(x,T)
        return interp1d(t,stim_ssa)(x)


    ########################## FS
    #p t0_fs=10,dur_fs=5,gap_fs=2,isi_fs=38

    t0=100;dur_fs=50;gap_fs=20;isi_fs=380

    ta_on_fs = np.zeros(80)
    ta_off_fs = np.zeros(len(ta_on_fs))

    tb_on_fs = np.zeros(len(ta_on_fs))
    tb_off_fs = np.zeros(len(ta_on_fs))

    ta_on_fs[0] = t0
    ta_off_fs[0] = ta_on_fs[0]+dur_fs
    tb_on_fs[0] = ta_off_fs[0]+gap_fs
    tb_off_fs[0] = tb_on_fs[0]+dur_fs

    stim_fs = np.zeros(len(t))
    stim_fs += gt(t,ta_on_fs[0],ta_off_fs[0])
    stim_fs += gt(t,tb_on_fs[0],tb_off_fs[0])

    for i in range(1,len(ta_on_fs)):

        ta_on_fs[i] = tb_off_fs[i-1]+isi_fs
        ta_off_fs[i] = ta_on_fs[i]+dur_fs
        tb_on_fs[i] = ta_off_fs[i]+gap_fs
        tb_off_fs[i] = tb_on_fs[i]+dur_fs

        j1 = np.argmin(np.abs(t-ta_on_fs[i]))
        j2 = np.argmin(np.abs(t-tb_on_fs[i]))

        stim_fs[j1:]+= gt(t[j1:],ta_on_fs[i],ta_off_fs[i])
        stim_fs[j2:]+= gt(t[j2:],tb_on_fs[i],tb_off_fs[i])

    def i_fs(x):
        x = np.mod(x,T)
        return interp1d(t,stim_fs)(x)


    ############################### TCA
    #p iti_tca=240,isi_tca=30,dur_tca=10

    t0=100;iti_tca=2400;isi_tca=300;dur_tca=100

    t_on_tca = np.zeros(50)
    t_off_tca = np.zeros(len(t_on_tca))

    t_on_tca[0] = t0
    t_off_tca[0] = t_on_tca[0]+dur_tca

    stim_tca = np.zeros(len(t))
    stim_tca += gt(t,t_on_tca[0],t_off_tca[0])

    for i in range(1,len(t_on_tca)):

        if i%8==0:
        #if i%5==0:
            add = iti_tca
        else:
            add = 0

        t_on_tca[i] = t_off_tca[i-1]+isi_tca + add
        t_off_tca[i] = t_on_tca[i]+dur_tca
        #tb_on_tca[i] = ta_off_tca[i]+isi_tca
        #tb_off_tca[i] = tb_on_tca[i]+dur_tca

        j1 = np.argmin(np.abs(t-t_on_tca[i]))

        stim_tca[j1:]+= gt(t[j1:],t_on_tca[i],t_off_tca[i])

    def i_tca(x):
        x = np.mod(x,T)
        return interp1d(t,stim_tca)(x)

    ################################## PV act
    t0=100;isi_pv=950;dur_pv=50

    t_on_pv = np.zeros(50)
    t_off_pv = np.zeros(len(t_on_pv))

    t_on_pv[0] = t0
    t_off_pv[0] = t_on_pv[0]+dur_pv

    stim_pv = np.zeros(len(t))
    stim_pv += gt(t,t_on_pv[0],t_off_pv[0])

    for i in range(1,len(t_on_pv)):

        t_on_pv[i] = t_off_pv[i-1]+isi_pv
        t_off_pv[i] = t_on_pv[i]+dur_pv
        #tb_on_pv[i] = ta_off_pv[i]+isi_pv
        #tb_off_pv[i] = tb_on_pv[i]+dur_pv

        j1 = np.argmin(np.abs(t-t_on_pv[i]))

        stim_pv[j1:]+= gt(t[j1:],t_on_pv[i],t_off_pv[i])

    def i_pv(x):
        x = np.mod(x,T)
        return interp1d(t,stim_pv)(x)



    def rhs(y,t,taud1=1500,taud2=100):
        # originally taud1=1500,taud2=100
        gSSA = y[0];gFS=y[1];gTCA=y[2];gPV=y[3]

        # thalamic input modifiers, slow timescale depression. Into all neurons
        return np.array([-gSSA**2/taud1+i_ssa(t)/taud2,
                         -gFS**2/taud1+i_fs(t)/taud2,
                         -gTCA**2/taud1+i_tca(t)/taud2,
                         -gPV**2/taud1+i_pv(t)/taud2,
        ])


    sol = np.zeros((len(t),4))


    for i in range(1,len(t)):
        sol[i,:] = sol[i-1,:]+dt*rhs(sol[i-1,:],t[i])

    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(111)
    

    # threshold label pad
    th_pad = 200

    # plot threshold
    ax.plot([0,T+3*th_pad],[0.22,0.22],color='gray')
    ax.text(2000,0.225,r'$F_{th}$')

    ax.plot(t,sol[:,0],label='SSA',color='k',lw=1.5)
    #ax.plot(t,sol[:,1],label='FWS',lw=1.5,ls=(0,(1,1)))
    ax.plot(t,sol[:,1],label='FWS',lw=1.5)
    ax.plot(t,sol[:,2],label='TCA',dashes=(5,1),lw=1.5)
    ax.plot(t,sol[:,3],label='PV Act.',lw=1.5,ls=(0,(3,1,1,1,1,1)))

    # label different attentional states
    t = ax.text(0+th_pad,.26,'Low Inhibition')
    t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='white'))
    
    #ax.text(1,.75,'Low Inhibition',transform=plt.gcf().transFigure)
    
    ax.annotate('', xy=(4*th_pad, .26),  xycoords='data',
                xytext=(4*th_pad, 0.22),
                arrowprops=dict(facecolor='gray', shrink=0.05),
                horizontalalignment='left', verticalalignment='top',
    )

    t = ax.text(0+th_pad,.16,'High Inhibition')
    t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='white'))
    #ax.text(1,.5,'High Inhibition',transform=plt.gcf().transFigure)
    
    ax.annotate('', xy=(4*th_pad, .18),  xycoords='data',
                xytext=(4*th_pad, 0.22),
                arrowprops=dict(facecolor='gray', shrink=0.05),
                horizontalalignment='left', verticalalignment='top',
    )



    ax.set_xlabel('Time (ms)')
    
    ax.set_ylabel(r'Total Input Integration ($\bar F$)')
    
    ax.legend(loc='lower right')

    ax.set_xlim(0,T)

    #fig.canvas.draw()

    
    return fig


def transition():
    """
    figure showing transition of SSA results over 15 sec.
    """
    
    

def generate_figure(function, args, filenames, title="", title_pos=(0.5,0.95)):

    fig = function(*args)

    if type(filenames) == list:
        for name in filenames:
            #fig.savefig(name,bbox_inches="tight")
            fig.savefig(name)

    else:
        fig.savefig(filenames)

def main():

    figures = [
        (r1_responses,[],['r1_responses.pdf']), # Fig 1
        (r3_responses,[],['r3_responses.pdf']), # Fig 2

        (attentional,[],['attentional.pdf']), # Fig 3
        
        (s1_responses,[],['s1_responses.pdf']), # Fig 4

        (r1_adaptation,[],['r1_adaptation.pdf']), # Fig 6
        
        (r3_CSI_params,[],['r3_CSI_params.pdf']), # Fig 8
        (s3_CSI_params,[],['s3_CSI_params.pdf']), # Fig 9

        (r3_s3_ssa_full,[False],['r3_s3_ssa_full.pdf']),  # Fig 7
        (r3_s3_fs_full,[False],['r3_s3_fs_full.pdf']), # Fig 10
        (r3_s3_tuning_full,[False],['r3_s3_tuning_full.pdf']), # Fig 11
        (r3_s3_pv_full,[False],['r3_s3_pv_full.pdf']), # Fig 12

        #(transition,[],['transition.pdf']),
        
        ]

    for fig in figures:
        print fig
        generate_figure(*fig)

if __name__ == "__main__":
    main()
