#Synapse and network setup
#IGABA, IAMPA, INMDA synapse setup
from brian import *
from model import *
import stimuli

#Used for synapse variable calculations
def tpeak(t1, t2):
	return (t1 * t2) / (t1 - t2)

#Used for synapse variable calculations
#Convolution of spike train, sj(t), and synaptic activation profile, a(t)
def hjt(a, t, t1, t2, tpeak, tdelay, V_pre):
	at = a * (e**(-(t-tdelay)/t1) - e**(-(t-tdelay)/t2)) / e**(-tpeak/t1) - e**(-tpeak/t2)
	return V_pre * at #5 (hj(t) = sj(t) * a(t))

#Connection probability
#Used for IGABA, IAMPA and INMDA connections
def p():
	p = 0.6
	return (rand() < p)

#Creates a neuron group using the model parametres imported from eqs
def createNeuronGroup(no_of_neurons):
	return NeuronGroup(N=no_of_neurons, model=V_eqs, threshold=Vt, reset=Vr, clock=1*ms)

#Sets up horizontal synapses using a single model for each neurongroup in a given list
#Returns that list of synapses
def setupHorizontalSynapses(neurongroup_list, synapse_model):
	synapse_list = [Synapses(neurongroup_list[x], model=synapse_model, pre=IGABA_pre)
					for x in range(len(neurongroup_list))]
	return synapse_list


#Columnar organisation and connectivity
#Thalamocortical columns (80 excitatory, 40 inhibitory total per modular subsystem) 2003, 1c
#Split into four layers
#Each layer can be referenced by a list of inhibitory and excitatory neurons
def setupColumn():
	column = createNeuronGroup(120)
	column.thalamus = column[0:30]
	column.thalamus_ie = [column.thalamus[0:10], column.thalamus[10:30]]
	column.infragranular = column[30:60]
	column.infra_ie = [column.infragranular[0:10], column.infragranular[10:30]]
	column.iv = column[60:90]
	column.iv_ie = [column.iv[0:10], column.iv[10:30]]
	column.supragranular = column[90:120]
	column.supra_ie = [column.supragranular[0:10], column.supragranular[10:30]]
	column.inhibitory = [column.thalamus_ie[0], column.infra_ie[0], column.iv_ie[0], column.supra_ie[0]]

	#4a
	#Inhibitory neurons sent intralaminar horizontal connections both within a column 
	synapses_intraGABA = setupHorizontalSynapses(column.inhibitory, 'IGABA : uA')
	for x in range(len(synapses_intraGABA)):
		column.inhibitory[x].IGABA = synapses_intraGABA[x].IGABA
		synapses_intraGABA[x][:,:] = True

	#and toward other competing columns within the same area (2005)
	#synapses_interGABA = setupSynapses(model_interGABA, 'IGABA : uA')

	#4b
	#Intracolumnar, bottom-up excitation
	IAMPA_thalamic2iv = Synapses(column.thalamus_ie[1], column.iv,
		model='''IAMPA : uA
				 gAMPA = numpy.random.normal(0.2, 0.02) : 1
				 tdelay = numpy.random.normal(3, 0.3) : 1''',
		pre=IAMPA_pre)
	IAMPA_thalamic2iv[:,:] = True

	IAMPA_thalamic2infragranular = Synapses(column.thalamus_ie[1], column.infragranular,
		model='''IAMPA : uA
				  gAMPA = numpy.random.normal(0.1, 0.01) : 1
				  tdelay = numpy.random.normal(3, 0.3) : 1''',	
		pre = IAMPA_pre)
	IAMPA_thalamic2infragranular[:,:] = True
		
	IAMPA_iv2supragranular = Synapses(column.iv_ie[1], column.supragranular,
		model='''IAMPA : uA
				 gAMPA = numpy.random.normal(0.15, 0.015) : 1
				 tdelay = numpy.random.normal(2, 0.2) : 1''',	
		pre = IAMPA_pre)
	IAMPA_iv2supragranular[:,:] = True
		
	IAMPA_supragranular2infragranular = Synapses(column.supra_ie[1], column.infragranular,
		model='''IAMPA : uA
				 gAMPA = numpy.random.normal(0.1, 0.01) : 1
				 tdelay = numpy.random.normal(2, 0.2) : 1''',	
		pre = IAMPA_pre)
	IAMPA_supragranular2infragranular[:,:] = True

	IAMPA_infragranular2iv = Synapses(column.infra_ie[1], column.iv,
		model='''IAMPA : uA
				 gAMPA = numpy.random.normal(0.05, 0.005) : 1
				 tdelay = numpy.random.normal(7, 0.7) : 1''',
		pre = IAMPA_pre)
	IAMPA_infragranular2iv[:,:] = True

	IAMPA_infragranular2supragranular = Synapses(column.infra_ie[1], column.supragranular,
		model='''IAMPA : uA
				 gAMPA = numpy.random.normal(0.05, 0.005) : 1
				 tdelay = numpy.random.normal(7, 0.7) : 1''',	
		pre = IAMPA_pre)
	IAMPA_infragranular2supragranular[:,:] = True

	IAMPA_infragranular2thalamus = Synapses(column.infra_ie[1], column.thalamus,
		model='''IAMPA : uA
				 gAMPA = numpy.random.normal(0.075, 0.0075) : 1
				 tdelay = numpy.random.normal(8, 0.8) : 1''',	
		pre = IAMPA_pre)
	IAMPA_infragranular2thalamus[:,:] = True

	#7
	#After each spike, the conductance gSRA is increased by a small quantity (here 0.01 * mS), 2005
	ISRA_S = Synapses(column, pre='''gSRA += gSRA_inc''')
	ISRA_S[:,:] = True

	return column

#4b
#Corticocortical projections
#Supragranular excitatory neurons of each layer projected to layer IV of the next layer
def setupCorticoIAMPASynapse(pre_supra, post_iv):
	IAMPA_supra2iv = Synapses(pre_supra, post_iv,
		model='''IAMPA : uA
				 gAMPA = numpy.random.normal(0.05, 0.005) : 1
				 tdelay = numpy.random.normal(3, 0.3) : 1''',	
		pre = IAMPA_pre)
	IAMPA_supra2iv[:,:] = True

#4c
#Supra and infra granular excitation of a given column to the supra and infra granular layers of all areas of a lower level
#Transmission delays increased with cortical distance, d = 0, d += 1 for each area apart in the heirarchy
#Strong top-down connections linked columns coding for the same stimulus
def setupStrongINMDASynapse(input, output, d):
	INMDA_strong = Synapses(input, output,
		model='''INMDA : uA
				 gNMDA = numpy.random.normal(0.05, 0.005) : 1
				 tdelay = 5 + 3*d : 1''',	
		pre = INMDA_pre)
	
#whereas weaker top-down connections projected to all columns of a lower area
def setupWeakINMDASynapse(input, output, d):
	INMDA_weak = Synapses(input, output,
		model='''INMDA : uA
				 gNMDA = numpy.random.normal(0.025, 0.0025) : 1
				 tdelay = 5 + 3*d : 1''',	
		pre = INMDA_pre)
	INMDA_weak[:,:] = True

#Synapses
#t in ms but removed for equations
#4a
aGABA = 0.175
t1GABA = 1
t2GABA = 7
tpeakGABA = tpeak(t1GABA, t2GABA)
VGABA = -70 * mV
IGABA_pre = '''IGABA = (gGABA_intra * hjt(aGABA, t, t1GABA, t2GABA, tpeakGABA, tdelay_intraGABA, V_pre)
			   * (V - VGABA)) * p()'''
gGABA_intra = numpy.random.normal(0.12, 0.012)	#0.12 mean with 10% RSD
tdelay_intraGABA = numpy.random.normal(2, 0.2)
gGABA_inter = numpy.random.normal(0.6, 0.0012) * msiemens
tdelay_interGABA = numpy.random.normal(2, 0.2) * ms

#4b
aIAMPA = 0.05
t1IAMPA = 0.5
t2IAMPA = 2.4
tpeakIAMPA = tpeak(t1IAMPA, t2IAMPA)
VAMPA = 0 * mV
IAMPA_pre = '''IAMPA = (gAMPA * hjt(aIAMPA, t, t1IAMPA, t2IAMPA, tpeakIAMPA, tdelay, V_pre)
			   * (V - VAMPA)) * p()'''

#4c
aINMDA = 0.0075
t1INMDA = 4
t2INMDA = 40
tpeakINMDA = tpeak(t1INMDA, t2INMDA)
VNMDA = 0 * mV
mNMDAV = '''(1 + 0.280 * e**(-V_post/16.1))**-1'''	#scaling
INMDA_pre = '''INMDA = dot((gNMDA * mNMDAV * V * hjt(aINMDA, t, t1INMDA, t2INMDA, tpeakINMDA, tdelay, V_pre)),
			   (V_post - VNMDA)) * p()'''
	
def rewardSignal(synapse, reward):
	reward_signal = Synapses(pre_synapse, post_synapse, 
		model='''w : mV
				 change = epsilon*(synapse_pre * (2*synapse_post - 1) * reward) : mV''',
		pre='v += w + change')
	reward_signal[:,:]