#import xppcall as pxc
from xppcall import xpprun,read_pars_values_from_file,read_init_values_from_file

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


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

        #print idx_start,idx_end,toneOn,toneOff

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

        
        #assert(np.sum(maxes_utemp)==1)
        #assert(np.sum(maxes_v1temp)==1)
        #assert(np.sum(maxes_v2temp)==1)

        # take the first max for now.
        #print ttemp,maxes_utemp,np.shape(ttemp),np.shape(maxes_utemp)
        if np.sum(ttemp[maxes_utemp]) > 1:    
            maxes_u[i,:] = [ttemp[maxes_utemp][0],utemp[maxes_utemp][0]]
        elif np.sum(ttemp[maxes_utemp]) < 1:
            pass
        else:
            maxes_u[i,:] = [ttemp[maxes_utemp],utemp[maxes_utemp]]

        if np.sum(ttemp[maxes_v1temp]) > 1:            
            maxes_v1[i,:] = [ttemp[maxes_v1temp][0],v1temp[maxes_v1temp][0]]
            
        elif np.sum(ttemp[maxes_v1temp]) < 1:
            pass
                
        else:
            maxes_v1[i,:] = [ttemp[maxes_v1temp],v1temp[maxes_v1temp]]
            
        if np.sum(ttemp[maxes_v2temp]) > 1: 
            maxes_v2[i,:] = [ttemp[maxes_v2temp][0],v2temp[maxes_v2temp][0]]

        elif np.sum(ttemp[maxes_v2temp]) < 1:
            pass
            
                

        else:
            print [ttemp[maxes_v2temp],v2temp[maxes_v2temp]]
            maxes_v2[i,:] = [ttemp[maxes_v2temp],v2temp[maxes_v2temp]]

        i += 1

    #print maxes_u,np.shape(maxes_u)
        
    return maxes_u,maxes_v1,maxes_v2

    #print maxes_u
    #print maxes_v1
    #print maxes_v2

    
    

def run_experiment(fname,pars,inits,return_all=False):
    
    npa, vn = xpprun(fname,
                     xppname='xppaut',
                     inits=inits,
                     parameters=pars,
                     clean_after=True)

    t = npa[:,0]
    sv = npa[:,1:]

    total_time = t[-1]

    u = sv[:,vn.index('u')]
    v1 = sv[:,vn.index('v1')]
    v2 = sv[:,vn.index('v2')]
    ia = sv[:,vn.index('ia')]
    g = sv[:,vn.index('g')]

    if return_all:

        tonelist = [(float(pars['tone1on']),float(pars['tone1off'])),
                    (float(pars['tone2on']),float(pars['tone2off'])),
                    (float(pars['tone3on']),float(pars['tone3off'])),
                    (float(pars['tone4on']),float(pars['tone4off'])),
                    (float(pars['tone5on']),float(pars['tone5off']))
        ]

        # implement parameter return dict.
        return {'t':t,'u':u,'v1':v1,'v2':v2,'inits':inits,'parameters':pars,'tonelist':tonelist,'sv':sv,'vn':vn,'ia':ia,'g':g}

    else:
        return {'t':t,'u':u,'v1':v1,'v2':v2}

def main():

    fname = 'xpp/natan2015_simple_linear.ode'
    pars = read_pars_values_from_file(fname)
    inits = read_init_values_from_file(fname)
    
    #pars['taud2']=7

    # Pyr input weights
    #pars['aee']=1.5
    #pars['aie1']=1 # PV to pyr weight
    #pars['aie2']=1 # SOM to pyr weight
        
    #pars['aei1']=3;pars['aii1']=1;pars['asom2pv']=1 # pv input weights
    #pars['aei2']=5;pars['aii2']=2;pars['apv2som']=1 # som input weights

    #pars['taud1s2p']=200;pars['tads2p']=5 # som->pn depression
    #pars['taud1s2v']=150;pars['tads2v']=5 # som->pv depression
    
    # returns tuple (t,u,v1,v2,inits,parameters,tonelist)    
    control = run_experiment(
        fname,
        pars,inits,
        return_all=True)

    pars['pv_offall']=2
    
    pv_off = run_experiment(
        fname,
        pars,inits,
        return_all=True)
    
    pars['pv_offall']=0
    pars['som_offall']=1
    
    som_off = run_experiment(
        fname,
        pars,inits,
        return_all=True)
    
    maxes_u_control,maxes_v1_control,maxes_v2_control = get_tone_evoked_FR(
        control['t'],
        control['u'],
        control['v1'],
        control['v2'],
        control['tonelist'])

    maxes_u_pv_off,maxes_v1_pv_off,maxes_v2_pv_off = get_tone_evoked_FR(
        pv_off['t'],
        pv_off['u'],
        pv_off['v1'],
        pv_off['v2'],
        pv_off['tonelist'])

    maxes_u_som_off,maxes_v1_som_off,maxes_v2_som_off = get_tone_evoked_FR(
        som_off['t'],
        som_off['u'],
        som_off['v1'],
        som_off['v2'],
        som_off['tonelist'])



    
    gs = gridspec.GridSpec(3, 3)
    ax11 = plt.subplot(gs[0, 0])
    ax11.set_title('control')

    ax11.plot(control['t'],control['u'],label='pyr',color='blue')
    ax11.plot(control['t'],control['v1'],label='PV',color='green')
    ax11.plot(control['t'],control['v2'],label='SOM',color='red')

    # plot detected peaks
    ax11.scatter(maxes_u_control[:,0],maxes_u_control[:,1],color='blue')

    ax11.legend()

    ax21 = plt.subplot(gs[1,0])
    ax21.set_title('PV off')
    
    ax21.plot(pv_off['t'],pv_off['u'],label='pyr',color='blue')
    ax21.plot(pv_off['t'],pv_off['v1'],label='PV',color='green')
    ax21.plot(pv_off['t'],pv_off['v2'],label='SOM',color='red')

    ax21.scatter(maxes_u_pv_off[:,0],maxes_u_pv_off[:,1],color='blue')

    ax31 = plt.subplot(gs[2,0])
    ax31.set_title('SOM off')
    
    ax31.plot(som_off['t'],som_off['u'],label='pyr',color='blue')
    ax31.plot(som_off['t'],som_off['v1'],label='PV',color='green')
    ax31.plot(som_off['t'],som_off['v2'],label='SOM',color='red')

    ax31.scatter(maxes_u_som_off[:,0],maxes_u_som_off[:,1],color='blue')

    # plot relative firing rates
    ax12 = plt.subplot(gs[:,1])
    
    tone_number = np.array([0,1,2,3,4])
    
    adapted_fr = maxes_u_control[-1,1]
    
    bar_width = 0.2
    ax12.set_title('Mean FR')
    ax12.bar(tone_number,maxes_u_control[:,1]/adapted_fr,width=bar_width,label='control',color='blue')
    ax12.bar(tone_number+bar_width,maxes_u_pv_off[:,1]/adapted_fr,width=bar_width,label='pv_off',color='green')
    ax12.bar(tone_number+2*bar_width,maxes_u_som_off[:,1]/adapted_fr,width=bar_width,label='som_off',color='red')
    ax12.plot([0,4],[1,1],ls='--',color='gray')

    ax12.legend()
    
    plt.tight_layout()

    # plot diff in firing rates
    ax13 = plt.subplot(gs[:,2])

    ax13.set_title('Diff from Control')
    ax13.bar(tone_number,np.abs(maxes_u_control[:,1]-maxes_u_pv_off[:,1])/adapted_fr,width=bar_width,label='control-pv_off',color='green')
    ax13.bar(tone_number+bar_width,np.abs(maxes_u_control[:,1]-maxes_u_som_off[:,1])/adapted_fr,width=bar_width,label='control-som_off',color='red')
    #ax13.plot([0,4],[1,1],ls='--',color='gray')

    ax13.legend()

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
