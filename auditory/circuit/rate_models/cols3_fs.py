#import xppcall as pxc
from xppcall import xpprun,read_pars_values_from_file,read_init_values_from_file

from scipy.stats.stats import pearsonr
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

som_color = '#66c2a5'
pv_color = '#fc8d62'

def get_max_FR(u,start_idx,end_idx):
    """
    get max FR in given index interval
    max should be unique
    """
    
    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(start_idx,end_idx),u[start_idx:end_idx])
        plt.show()

    return np.amax(u[start_idx:end_idx])

def get_tone_evoked_FR(t,u,v1,v2,tonelist):
    """
    brute force method for extracting peaks following tone-evoked activity.

    loop over each tone-evoked time frame.
    """
    
    maxes_u = np.zeros((len(tonelist),2))
    maxes_v1 = np.zeros((len(tonelist),2))
    maxes_v2 = np.zeros((len(tonelist),2))
    
    i = 0
    for toneOn,toneOff in tonelist:

        # get start/end time index
        idx_start = np.argmin(np.abs(t-toneOn))+1
        idx_end = np.argmin(np.abs(t-toneOff))-1

        utemp = u[idx_start:idx_end]
        v1temp = v1[idx_start:idx_end]
        v2temp = v2[idx_start:idx_end]
        ttemp = t[idx_start:idx_end]

        # https://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
        maxes_utemp = np.r_[True, utemp[1:] > utemp[:-1]] & np.r_[utemp[:-1] > utemp[1:], True]
        maxes_v1temp = np.r_[True, v1temp[1:] > v1temp[:-1]] & np.r_[v1temp[:-1] > v1temp[1:], True]
        maxes_v2temp = np.r_[True, v2temp[1:] > v2temp[:-1]] & np.r_[v2temp[:-1] > v2temp[1:], True]

        if False:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(ttemp,utemp)
            ax.plot(ttemp,v1temp)
            ax.plot(ttemp,v2temp)

            ax.scatter(ttemp[maxes_utemp],utemp[maxes_utemp])
            ax.scatter(ttemp[maxes_v1temp],v1temp[maxes_v1temp])
            ax.scatter(ttemp[maxes_v2temp],v2temp[maxes_v2temp])
            plt.show()

        if True:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(t,u)
            ax.plot(t,v1)
            ax.plot(t,v2)
            plt.show()

            
        # take the first max for now.
        if np.sum(ttemp[maxes_utemp]) > 1:    
            maxes_u[i,:] = [ttemp[maxes_utemp][-1],utemp[maxes_utemp][-1]]
            
        elif np.sum(ttemp[maxes_utemp]) < 1:
            pass
        
        else:
            maxes_u[i,:] = [ttemp[maxes_utemp],utemp[maxes_utemp]]

        if np.sum(ttemp[maxes_v1temp]) > 1:            
            maxes_v1[i,:] = [ttemp[maxes_v1temp][-1],v1temp[maxes_v1temp][-1]]
            
        elif np.sum(ttemp[maxes_v1temp]) < 1:
            pass
                
        else:
            maxes_v1[i,:] = [ttemp[maxes_v1temp],v1temp[maxes_v1temp]]
            
        if np.sum(ttemp[maxes_v2temp]) > 1: 
            maxes_v2[i,:] = [ttemp[maxes_v2temp][-1],v2temp[maxes_v2temp][-1]]

        elif np.sum(ttemp[maxes_v2temp]) < 1:
            pass
            
        else:
            print [ttemp[maxes_v2temp],v2temp[maxes_v2temp]]
            maxes_v2[i,:] = [ttemp[maxes_v2temp],v2temp[maxes_v2temp]]

        i += 1
        
    return maxes_u,maxes_v1,maxes_v2


def run_experiment(fname,pars,inits,return_all=False):
    
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
    

    print vn
    
    if return_all:

        tonelist = []
        t0 = float(pars['t0'])
        dur = float(pars['dur'])
        isi = float(pars['isi'])
        
        for i in range(5):
            tonelist.append((t0+i*(dur+isi),t0+(i+1)*dur+i*isi))
            
        #tonelist = [(float(pars['tone1on']),float(pars['tone1off'])),
        #            (float(pars['tone2on']),float(pars['tone2off'])),
        #            (float(pars['tone3on']),float(pars['tone3off'])),
        #            (float(pars['tone4on']),float(pars['tone4off'])),
        #            (float(pars['tone5on']),float(pars['tone5off']))
        #]

        # implement parameter return dict.
        return {'t':t,'u1':u1,'u2':u2,'u3':u3,'inits':inits,'parameters':pars,'tonelist':tonelist,'sv':sv,'vn':vn}

    else:
        return {'t':t,'u1':u1,'u2':u2,'u3':u3}

def main():

    fname = 'xpp/cols3_fs.ode'
    pars = read_pars_values_from_file(fname)
    inits = read_init_values_from_file(fname)

    # repeat simulation for different frequencies.
    # pars['mode'] = '1' makes the first tone appear in the first column
    # pars['mode'] = '2' makes the first tone appear in the second column.. etc.

    results = np.zeros((3,6))
    
    for i in range(3):
        pars['mode']=str(i+1)

        pars['pv_opto']=0
        pars['som_opto']=0

        # returns tuple (t,u,v1,v2,inits,parameters,tonelist)    
        control = run_experiment(
            fname,
            pars,inits,
            return_all=True)

        pars['pv_opto']=.01

        pv_off = run_experiment(
            fname,
            pars,inits,
            return_all=True)

        pars['pv_opto']=0
        pars['som_opto']=.2

        som_off = run_experiment(
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
        control1 = get_max_FR(control['u'+str(i+1)],idx1_start,idx1_end)
        pv1 = get_max_FR(pv_off['u'+str(i+1)],idx1_start,idx1_end)
        som1 = get_max_FR(som_off['u'+str(i+1)],idx1_start,idx1_end)
        
        # get second tone (always u2)
        control2 = get_max_FR(control['u2'],idx2_start,idx2_end)
        pv2 = get_max_FR(pv_off['u2'],idx2_start,idx2_end)
        som2 = get_max_FR(som_off['u2'],idx2_start,idx2_end)
        
        results[i,:] = [control1,control2,pv1,pv2,som1,som2]

        
        if (i == 1) and False:
            fig = plt.figure()
            ax11 = fig.add_subplot(111)
            ax11.plot(control['u2'])
            ax11.plot(som_off['u2'])
            plt.show()


        # end 3 tone loop

    # run PV activation for correlation calculation.
    pars['pv_opto']=-.2
    pars['som_opto']=0.
    pars['mode']=2
    
    pv_on = run_experiment(
        fname,
        pars,inits,
        return_all=True)


    # run PV activation for correlation calculation.
    pars['pv_opto']=0
    pars['som_opto']=0.
    pars['mode']=2
    
    pv_control = run_experiment(
        fname,
        pars,inits,
        return_all=True)


    print results

    # correlation
    fig3 = plt.figure(figsize=(8,3))
    ax1 = fig3.add_subplot(121)
    ax2 = fig3.add_subplot(122)

    
    time = pv_on['t']
    input_trace = pv_on['sv'][:,pv_on['vn'].index('ia2')]

    time_short = time[time<20]
    input_trace_short = input_trace[time<20]

    time_ctrl = pv_control['t']
    input_trace_ctrl = pv_control['sv'][:,pv_control['vn'].index('ia2')]

    time_short_ctrl = time_ctrl[time_ctrl<20]
    input_trace_short_ctrl = input_trace_ctrl[time_ctrl<20]

    ax1b = ax1.twinx()
    ax1b.plot(time_short_ctrl*10,input_trace_short_ctrl,color='tab:red')
    ax1.plot(time_short_ctrl*10,pv_control['u2'][time_ctrl<20])

    ax1.set_title('Pyr Control FR Rate')
    ax1.set_ylabel('Firing Rate')
    ax1b.set_ylabel('Thalamus',color='tab:red')
    
    print "PV act. corr = "+str(pearsonr(input_trace_short,pv_on['u2'][time<20]))
    



    ax2b = ax2.twinx()
    ax2b.plot(time_short*10,input_trace_short,color='tab:red')    
    ax2.plot(time_short*10,pv_on['u2'][time<20])

    ax1.set_title('Pyr FR Rate with PV Activation')
    ax2.set_ylabel('Firing Rate')
    ax2b.set_ylabel('Thalamus',color='tab:red')

    print "PV control corr = "+str(pearsonr(input_trace_short_ctrl,pv_control['u2'][time_ctrl<20]))

    ax1.set_xlabel('t')
    ax2.set_xlabel('t')
    plt.tight_layout()

    


    fig = plt.figure()
    #sv[:,vn.index('u1')]    

    # plot relative firing rates
    ax11 = fig.add_subplot(121)
    ax12 = fig.add_subplot(122)
        
    bar_width = 0.2
    #ax11.set_title('Peak Response')
    #ax11.scatter(0,maxes_u_control[0,1],label='Control 1st Tone',color='black')
    #ax11.scatter(0,maxes_u_pv_off[0,1],label='',color=pv_color)

    ax11.set_title('Normalized Peak Response (2nd Tone)')
    
    control_probe = results[1,0]
    ax11.scatter(-1,results[0,1]/control_probe,label='Control 2nd Tone',color='black')
    ax11.scatter(0,results[1,1]/control_probe,label='',color='black')
    ax11.scatter(1,results[2,1]/control_probe,label='',color='black')

    pv_probe = results[1,2]
    ax11.scatter(-1,results[0,3]/pv_probe,label='PV Off 2nd Tone',color=pv_color)
    ax11.scatter(0,results[1,3]/pv_probe,label='',color=pv_color)
    ax11.scatter(1,results[2,3]/pv_probe,label='',color=pv_color)

    ax12.set_title('Normalized Peak Response (2nd Tone)')

    control_probe = results[1,0]    
    ax12.scatter(-1,results[0,1]/control_probe,label='Control 2nd Tone',color='black')
    ax12.scatter(0,results[1,1]/control_probe,label='',color='black')
    ax12.scatter(1,results[2,1]/control_probe,label='',color='black')

    som_probe = results[1,4]
    ax12.scatter(-1,results[0,5]/som_probe,label='SOM Off 2nd Tone',color=som_color)
    ax12.scatter(0,results[1,5]/som_probe,label='',color=som_color)
    ax12.scatter(1,results[2,5]/som_probe,label='',color=som_color)

    
    #ax11.scatter()
    
    """
    ax12.set_title('Normalized Peak Response (2nd Tone)')
    ax12.scatter(0,maxes_u_control[1,1]/maxes_u_control[0,1],label='Control 2nd Tone',color='black')
    ax12.scatter(0,maxes_u_som_off[1,1]/maxes_u_som_off[0,1],label='SOM Off',color='red')
    
    #ax12.bar(tone_number+bar_width,maxes_u_pv_off[:,1]/adapted_fr,width=bar_width,label='pv_off',color='green')
    #ax12.bar(tone_number+2*bar_width,maxes_u_som_off[:,1]/adapted_fr,width=bar_width,label='som_off',color='red')
    #ax12.plot([0,4],[1,1],ls='--',color='gray')
    """
    
    ax11.set_xlabel('Distance from Preferred Frequency')
    ax12.set_xlabel('Distance from Preferred Frequency')
    
    ax11.legend()
    ax12.legend()
    
    plt.tight_layout()

    # plot synapses
    if False:
        sv = control['sv']
        vn = control['vn']

        aie2 = float(control['parameters']['aie2']) # som to pn
        asom2pv = float(control['parameters']['asom2pv']) # som to pv

        ws2p = sv[:,vn.index('ws2p')] # som to pn
        ws2v = sv[:,vn.index('ws2v')] # som to pv

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(control['t'],aie2*ws2p,label='som to pn')
        ax2.plot(control['t'],asom2pv*ws2v,label='som to pv')
        
    plt.show()


if __name__ == "__main__":
    main()
