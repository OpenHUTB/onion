from brian import *

def tpeak(t1, t2):
	return (t1 * t2) / (t1 - t2)
	
#convolution of spike train, sj(t), and synaptic activation profile, a(t)
def hjt(a, t1, t2, tpeak):
	at = a * (e**(-t/t1) - e**(-t/t2)) / e**(-tpeak/t1) - e**(-tpeak/t2)
	return V_pre * at #5 (hj(t) = sj(t) * a(t))

#Model neurons (single-compartment integrate-and-fire units)
C = 1 * uF								#capacitance
gleak = 0.1 * msiemens					#leak conductance
#tau = C/gleak 							#membrane time constant (C/gleak) in ms
Vrest = -67 * mV						#resting potential (El)
Vt = -48 * mV							#spike threshold
Vr = -80 * mV							#reset value

VNa = 55 * mV
gNaP = 0.2 * msiemens					#mean with a 5% SD and Gaussian distribution

VK = -90 * mV
tKS = 6 * ms
gKS = 8 * msiemens						#mean with a 5% SD and Gaussian distribution

VSRA = -70 * mV
tSRA = 200 * ms

#Ionic current variables
Ineuromodul = -1 * uA
ginput = 0.06 * msiemens

#Temporal evolution of V
#C dV/dt = -gleak(V-Vrest) - INaP - IKS - IGABA - IAMPA - INMDA - ISRA - Iinput - Ineuromodul
#IGABA, IAMPA, INMDA to be calculated separately

V_eqs = Equations('''
	#1
	dV/dt = I/C : mV
	I = -gleak * (V - Vrest) - INaP - IKS - IGABA - IAMPA - INMDA - ISRA - Ineuromodul: uA
	
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
	#Iinput = ginput * (V - VAMPA) : uA
	''')

#Synapses
#t in ms but removed for equations
#4a
aGABA = 0.175
t1GABA = 1
t2GABA = 7
tpeakGABA = tpeak(t1GABA, t2GABA)
VGABA = -70 * mV
IGABA_pre = '''IGABA = gGABA * (hjt(aGABA, t1GABA, t2GABA, tpeakGABA) - tdelay) * (V_post - VGABA)'''
gGABA_intra = 0.12 * msiemens
tdelay_intraGABA = 2 * ms
gGABA_inter = 0.6 * msiemens
tdelay_interGABA = 2 * ms

#4b
aIAMPA = 0.05
t1IAMPA = 0.5
t2IAMPA = 2.4
tpeakIAMPA = tpeak(t1IAMPA, t2IAMPA)
VAMPA = 0 * mV
IAMPA_eq = '''S.(gAMPA * (hjt(aIAMPA, t1IAMPA, t2IAMPA, tpeakIAMPA) - tdelay) * (V_post - VAMPA))'''

#4c
aINMDA = 0.0075
t1INMDA = 4
t2INMDA = 40
tpeakINMDA = tpeak(t1INMDA, t2INMDA)
VNMDA = 0 * mV
mNMDAV = '''(1 + 0.280 * e**(-V_post/16.1))**-1'''	#scaling
INMDA_eq = '''S.(gNMDA * mNMDAV * hjt(aINMDA, t1INMDA, t2INMDA, tpeakINMDA))*(V_post - VNMDA)'''

#7
ISRA_eq = '''dgSRA/dt = -gSRA / tSRA
			gSRA * (V-VSRA)'''

