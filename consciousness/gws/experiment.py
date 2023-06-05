from brian import *
import brian_no_units
from network import *
import model, stimuli, synapses
					
#set stimuli to appear at separate intervals, then run the simulation
#(rather than running the network and testing the experiment)		
#the thalamus neuron group is sent the stimuli and its connections to the workspace interpret it
#the stimuli are interpreted as a variable within the thalamus model equations		

def inputStimuli(stimuli):
	suboptimal_ms = 4
	optimal_ms = 2000
	
	for stimulus in len(stimuli):
		if stimulus[0] != 0: #if stimulus does not have a prime
			if stimulus[0] == -1: #if stimulus has suboptimal prime
				for x in xrange(suboptimal_ms):
					for column in xrange(len(modules[0][0])):
						rand = randrange(0,20+1)
						setupInput(stimulus[1][column], modules[0][0][column+rand])
					

	
n = Network(modules)

n.run(1*second)