from brian import *
import numpy

#Model neurons (single-compartment intergrate-and-fire units)
C = 1 * uF								#capacitance
gleak = 0.1 * msiemens					#leak conductance
#tau = C/gleak 							#membrane time constant (C/gleak) in ms
Vrest = -67 * mV						#resting potential (El)
Vt = -48 * mV							#spike threshold
Vr = -80 * mV							#reset value

VNa = 55 * mV
gNaP = numpy.random.normal(0.2, 0.01) * msiemens	#0.2*mS mean with 5% SD and Gaussian distribution

VK = -90 * mV
tKS = 6 * ms
gKS = numpy.random.normal(8, 0.4) * msiemens	#8*mS mean with 5% SD and Gaussian distribution

VSRA = -70 * mV
tSRA = 200 * ms

#Ionic current variables
Ineuromodul = numpy.random.normal(-1, 0.05) * uA	#-1*uA mean with 5% SD and Gaussian distribution
ginput = 0.06 * msiemens

#Temporal evolution of V
#C dV/dt = -gleak(V-Vrest) - INaP - IKS - IGABA - IAMPA - INMDA - ISRA - Iinput - Ineuromodul
#IGABA, IAMPA, INMDA to be calculated separately

V_eqs = Equations('''
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


#7
ISRA_eq = '''dgSRA/dt = -gSRA / tSRA
			 gSRA * (V-VSRA)'''
