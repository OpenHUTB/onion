from brian import *
from synapses import *
from random import randrange

#Creates a layer of a random amount of thalamocortical columns with a default max of 3
def createColumns(min=1, max=4):
	return [setupColumn() for x in xrange(randrange(min,max+1))]

#Creates a cognitive system
def createProcessor(layers=5):
	processor = []
	processor.append(createColumns(4)) #for the very first layer, create four columns
	for x in xrange(2,layers):
		processor.append(createColumns())
	return processor

def setupInput(input, column):
	connection = Connection(column.thalamus, column.thalamus, weight=input)
	#8
	#Current applied during stimulus presentation only
	Iinput_synapse = Synapses(column.thalamus, 
		model='''ginput = 0.06 * input : msiemens
				 Iinput = ginput * (V - VAMPA) : uA''')
	Iinput_synapse[:,:] = True

perceptual = createProcessor()
evaluative = createProcessor()
attentional = createProcessor()
emotional = createProcessor()
motor = createProcessor()

modules = [perceptual, evaluative, attentional, emotional, motor]

for system_i in xrange(len(modules)):
	#bottom up corticocortical projections
	#Supragranular excitatory neurons of each area projected to layer IV of the next area
	#(Area: a layer of thalamocortical columns)
	for layer_i in xrange(len(modules[system_i])):
		for column_i in xrange(len(modules[system_i][layer_i])):
			if layer_i < (len(modules[system_i])-1): #for each layer apart from the topmost
				#connect each column to the columns in the above layer
				for above_column in modules[system_i][layer_i+1]:
					setupCorticoIAMPASynapse(modules[system_i][layer_i][column_i].supra_ie[1], above_column.iv)
	
	#top down connected the supra and infra granular excitatory neurons of a given column
	#to the supra and infra granular layers of all areas of a lower level
	for layer_i in reversed(xrange(len(modules[system_i]))):
		if layer_i != 0:	#for each layer apart from the first
			#connect the top layer columns to all lower columns, starting from the top
			for column in modules[system_i][layer_i]:
				layer_count = len(modules[system_i]) - layer_i - 1
				for layer_j in xrange(layer_count):
					for lower_column in modules[system_i][layer_i-1]:
						d = layer_count - layer_j
						#Strong top-down connections linked columns coding for the same stimulus
						setupStrongINMDASynapse(column.supra_ie[1], lower_column.supragranular, d)
						setupStrongINMDASynapse(column.supra_ie[1], lower_column.infragranular, d)
						setupStrongINMDASynapse(column.infra_ie[1], lower_column.supragranular, d)
						setupStrongINMDASynapse(column.infra_ie[1], lower_column.infragranular, d)
