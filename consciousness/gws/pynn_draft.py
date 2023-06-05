"""
This model is based on Dehaenes and Changeux (2003, 2005),
 implemented using PyNN (Davidson et al., 2008) with
 Brian 1 (Goodman and Brette, 2009) and python 2.7
"""

"""
For information on conductance based models, see
 Frances K. Skinner (2006) Conductance-based models. Scholarpedia, 1(11):1408.
 
PyNN Unit Conventions
qty			unit
----------------
time		ms
voltage		mV
current		nA
conductance	uS
capacitance	nF
firing rate	/s
phase/angle	deg
"""

from pyNN.brian import *
setup()

params = {
		'v_thresh':		-48,	# Spike threshold, in mV
		'v_reset':		-80,	# Reset potential after a spike, in mV
		'tau_refrac':	4,		# Duration of refractory period, in ms
		'tau_m':		10, 	# Membrane time constant, in ms
		'cm':			1000, 	# Capacity of the membrane, in nF (1 uF)
		'g_leak':		100		# Leak conductance, in uS (0.1 mS)
	}
p = Population(12, IF_cond_exp(**params)