from brian import *
from matplotlib import pylab
from scipy.sparse import lil_matrix
import os
import numpy
import random

def setupStimuli(dir):
	stimuli = []
	stimuli = array([pylab.imread(dir+'\\'+file) for file in os.listdir(dir)])
	matrices = [ lil_matrix((len(stimuli[0]),len(stimuli[0][0]))) for x in xrange(len(stimuli)) ]
	
	for x in xrange(len(stimuli)): #for each stimulus
		for y in xrange(len(stimuli[x])): #for each row (100)
			for z in xrange(len(stimuli[x][y])): #for each column item in the row (100)
				matrices[x][y,z] = int(stimuli[x][y][z][0])
	
	random.shuffle(matrices)
	return matrices

def setupRatings(stimuli):
	stimuli_w_ratings = []
	return stimuli_w_ratings
	
angry_training = setupStimuli('stimuli\\training\\angry')
happy_training = setupStimuli('stimuli\\training\\happy')
test_set = setupStimuli('stimuli\\test')
irrelevant_set = setupStimuli('stimuli\\irrelevant')