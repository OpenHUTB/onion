#!/usr/bin/env python
# coding: utf-8

# In[12]:
# Pinsky-Rinzel(PR)模型的放电行为.PR模型能够模拟癫痫样式放电和正常放电,
# 癫痫样式的行为由29Hz的高频簇放电模拟,6Hz的低频放电模拟正常的神经元放电


import numpy as np
import scipy as sp
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from peaks import *


# In[13]:


#Constants
Gna = 30
Gkdr = 15
Gkca = 15
Gkahp = 0.8
Gca = 10
Gl = 0.1
Gc = 1.425
Gnmda = 1.75
Vna = 60 #Paper says 60, DB says 55 - DB has actually used 60.
Vk = -75
Vca = 80
Vl = -60
p = 0.5

#Ca taken from modelDB (=Cad)
Ca = 0.21664282 
Cm = 3

#Is and Id-diff values of them are given in the modelDB code.
Isapp = -0.5
Idapp = 0.0


# In[14]:


#Alpha and Beta functions and Chi
#Heaviside Step function - Not used in the code
def alphaM(Vs):
    return (0.32*(-46.9 - Vs)/(np.exp((-46.9 - Vs)/4.0) - 1.0))

def betaM(Vs):
    return (0.28*(Vs + 19.9)/(np.exp((Vs + 19.9)/5.0) - 1.0))

def Minf(Vs):
    return (alphaM(Vs)/(alphaM(Vs) + betaM(Vs)))

def alphaN(Vs):
    return ((0.016*(-24.9 - Vs))/(np.exp((-24.9 - Vs)/5.0) - 1.0))

def betaN(Vs):
    return (0.25*np.exp(-1.0 - 0.025*Vs))

def alphaH(Vs):
    return (0.125*np.exp((-43.0-Vs)/18.0))

def betaH(Vs):
    return (4.0/(1.0 + np.exp((-20.0-Vs)/5.0)))

def alphaS(Vd):
    return (1.6/(1.0 + np.exp(-0.072*(Vd - 5.0))))

def betaS(Vd):
    return (0.02*(Vd + 8.9)/(np.exp((Vd + 8.9)/5.0) - 1.0))
    
def qinf(Ca):
    return ((0.7894*np.exp(0.0002726*Ca)) - (0.7292*np.exp(-0.01672*Ca)))

def tauq(Ca):
    return (657.9*np.exp(-0.02023*Ca)) + (301.8*np.exp(-0.002381*Ca))

def cinf(Vd):
    return ((1.0/(1.0 + np.exp((-10.1 - Vd)/0.1016)))**0.00925)

def tauc(Vd):
    return (3.627 * np.exp(0.03704*Vd))

def chi(Ca):
    return (1.073*np.sin(0.003453*Ca+0.08095) + 0.08408*np.sin(0.01634*Ca-2.34) + 0.01811*np.sin(0.0348*Ca-0.9918))


def dVdt(V, t):
    Vs, Vd, m, n, h, s, c, q, Ca = V #V is a tuple
    
    #Equations for currents
    Ina = Gna * (Minf(Vs))**2 * h * (Vs - Vna)
    Ikdr = Gkdr * n * (Vs - Vk)
    Ica = Gca * s**2 * (Vd - Vca)
    Ikca = Gkca * c * chi(Ca) * (Vd - Vk)
    Ikahp = Gkahp * q * (Vd - Vk)
    Isd = Gc * (Vd - Vs)
    Ids = -Isd
    Ileakd = Gl * (Vd - Vl)
    Ileaks = Gl * (Vs - Vl)
    Inmda = Gnmda*(Vd)/(1 + 0.28*np.exp(-0.062*(Vd)))
    #Inmda taken from modelDB
    dCadt = (-0.13*Ica - 0.075*Ca)
    dvsdt = (-Ileaks - Ina - Ikdr - Ids/p + Isapp/p)/Cm
    #NOTE - Change incorporated: -Isd/(1-p) instead of +Isd/(1-p)
    dvddt = (-Ileakd - Ica - Ikca - Ikahp - Isd/(1-p) - Inmda/(1-p) + Idapp/(1-p))/Cm
    dmdt = (((alphaM(Vs)/(alphaM(Vs) + betaM(Vs))) - m)/(1/(alphaM(Vs) + betaM(Vs))))
    dndt = (((alphaN(Vs)/(alphaN(Vs) + betaN(Vs))) - n)/(1/(alphaN(Vs) + betaN(Vs))))
    dhdt = (((alphaH(Vs)/(alphaH(Vs) + betaH(Vs))) - h)/(1/(alphaH(Vs) + betaH(Vs))))
    dsdt = (((alphaS(Vd)/(alphaS(Vd) + betaS(Vd))) - s)/(1/(alphaS(Vd) + betaS(Vd))))
    #dcdt = (((alphaC(Vd)/(alphaC(Vd) + betaC(Vd))) - c)/(1/(alphaC(Vd) + betaC(Vd))))
    #dqdt = (((alphaQ(Ca)/(alphaQ(Ca) + betaQ(Ca))) - q)/(1/(alphaQ(Ca) + betaQ(Ca))))
    dqdt = (qinf(Ca) - q)/tauq(Ca)
    dcdt = (cinf(Vd) - c)/tauc(Vd)
    #dqdt = alphaQ(Ca) - (alphaQ(Ca) + betaQ(Ca))*q
    
    
    return dvsdt, dvddt, dmdt, dndt, dhdt, dsdt, dcdt, dqdt, dCadt


# In[15]:


tmin, tmax, dt = 0, 1000, 0.01 #ms unit
T = np.arange(tmin,tmax,dt)

V0 = [-62.89223689, -62.98248752, 0.5, 0.00068604, 0.99806345, 0.01086703, 0.00809387, 0.0811213, 0.21664282] #Initial values need to be feeded
sol = odeint(dVdt, V0, T)


# In[16]:


Vdmin, Vdmax, dVd = -100, 100, 0.1 #mV unit
Vd = np.arange(Vdmin,Vdmax,dVd)
#Here C is Ca - unit is probably Um
Cmin, Cmax, dC = 0, 500, 2
C = np.arange(Cmin, Cmax, dC)
plt.figure(figsize = (15,10))

plt.subplot(3,1,1)
plt.plot(Vd, cinf(Vd))
plt.ylabel('Cinf')
plt.xlabel('Dendritic Membrane Potential (mV)')
plt.legend

plt.subplot(3,1,2)
plt.plot(C, qinf(C))
plt.ylabel('Qinf')
plt.xlabel('Ca')
plt.legend

plt.subplot(3,1,3)
plt.plot(C, [chi(c) for c in C])
plt.ylabel('Chi')
plt.xlabel('Ca')
plt.legend

#plt.savefig('Functions of Cinf_Qinf_Chi.png')
#print qinf(C), cinf(Vd), chi(C)

plt.show()


# In[17]:


Vs, Vd, m, n, h, s, c, q, Ca = sol.T

plt.figure(figsize=(15, 10))

plt.subplot(3,1,1)
plt.title('Pinsky Rinzel Model')
plt.plot(T, Vs, label = 'Vs')
plt.plot(T, Vd, label = 'Vd')
plt.ylabel('Membrane Potential (mV)')
plt.legend()

plt.subplot(3,1,2)
plt.plot(T, m, label='m')
plt.plot(T, n, label='n')
plt.plot(T, h, label='h')
plt.ylabel('Gating Value')
plt.legend()

plt.subplot(3,1,3)
plt.plot(T, s, label='s')
plt.plot(T, c, label='c')
plt.plot(T, q, label='q')
plt.ylabel('Gating Value')
plt.legend()

plt.savefig('Dendritic_input_weak_coupling.png')

plt.show()


# In[7]:


#Frequency part - work in progress
print(Vs)
iv = detect_peaks(Vs,mph=10,show = True)
for i in iv:
    print('%f, %f, %f' %(i,Vs[i],T[i]))


# In[ ]:




