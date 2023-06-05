from brian import *
from model import *
from synapses import *

def createNeuronGroup(no_of_neurons):
	return NeuronGroup(N=no_of_neurons, model=V_eqs, threshold=Vt, reset=Vr)

#for testing different values of Ineuromodul
def createNeuroGroup(no_of_neurons, ineuromodul):
	V_ineuro = Equations('''
		#1
		dV/dt = I/C : mV
		I = -gleak * (V - Vrest) - INaP - IKS - IGABA - IAMPA - INMDA - ISRA - Iinput - Ineuromodul: uA
		
		#2 (Origins of spontaneous activity)
		INaP = gNaP * mNaP * (V - VNa) : uA
		mNaP = (1 + e**(-((V/mV) + 51)/5))**-1 : 1	#activation
		
		#3
		IKS = gKS * mKS * (V - VK) : uA
		depolar_mKS = 1 + e**(-((V/mV) + 34)/6.5)**-1 : 1
		dmKS/dt = (depolar_mKS - mKS) / tKS : 1
		
		#4
		IGABA : uA
		IAMPA : uA
		INMDA : uA
		
		#7
		ISRA = gSRA * (V - VSRA) : uA
		dgSRA/dt = -gSRA / tSRA : msiemens
		
		#8
		Iinput : uA
		''')
	return NeuronGroup(N=no_of_neurons, model=V_ineuro, threshold=Vt, reset=Vr)
	
#tests neuron against different levels of Ineuromodul according to spikes per second
def testIneuro(no_of_neurons=10, seconds=10, interval=0.25, steps=8):
	print 'Testing ' + str(no_of_neurons) + ' neurons for ' + str(seconds) + ' seconds per ' + str(steps+1) + ' Ineuromodul levels.'
	
	ineuro_spike_freq = []
	ineuro = [0 * uA]

	for x in range(steps):
		ineuro.append(ineuro[x] - interval * uA)

	for x in ineuro:
		print 'Test ' + str(ineuro.index(x) + 1) + ' of ' + str(steps + 1)
		
		ng = createNeuroGroup(no_of_neurons, x)		
		M=SpikeMonitor(ng)
		run(seconds * second)
		# print 'Number of spikes: ' + str(M.nspikes)
		# print 'Spikes per second: ' + str((M.nspikes)/seconds)
		ineuro_spike_freq.append((float(M.nspikes))/seconds)
	
	print 'Test complete, now plotting...'
	
	for x in range(len(ineuro)):
		ineuro[x] /= uA
		
	print 'Ineuromodul values: ' + str(ineuro)
	print 'Spikes per second: ' + str(ineuro_spike_freq)
	plot(ineuro, ineuro_spike_freq)
	xlabel('Ineuromodul (uA)')
	ylabel('Spike frequency (per second)')
	show()
	
def testVar(var_string, seconds=10):
	M = StateMonitor(monitor, var_string, record=True)
	run(seconds * second)
	M.plot()
	show()