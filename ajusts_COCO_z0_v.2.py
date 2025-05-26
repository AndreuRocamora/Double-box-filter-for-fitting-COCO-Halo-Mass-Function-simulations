# -*- coding: utf-8 -*-
"""
Created on Wed May  20 09:34:11 2024

@author: Andreu Rocamora Martorell
"""
#The following code allows fitting the Halo Mass Function (HMF) with simulations, here, Bose et.al (2016),
#using several filters, besides the canonic k-sharp filter and the smooth-k filter from Leo et.al (2018),
#a new filter, a special case of a generic Double Box is proposed, defined using two k-sharp filters (one weighted at 3/4 and one
#weighted at 1/4 with an scale variation).

#In the following code this new filter is referred as smooth Double Box to distinguish it 
#from the simple Double Box filter which is formed using two equally weighted boxes (k-sharp filters) with different scale variations

#Although the final selected filter is the smooth Double Box, the Double Box and a similar filter composed of four boxes has also been programmed 
#just to compare the trends of the filtered HMF

#An important element is the fitting parameter al which normalizes the power spectrum acording to simulations. 

#Another relevant detail is that for Sheth and Tormen (ST) mass fraction, the value q=0.707 has been taken
#from Leo et. al (2018) and Bose et. al (2016) differing from the value q=1 proposed by Sheth and Tormen (2002).
#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy 
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline

#%%
#Univers parameters
omm=0.272 #omega matter
omb=0.046 #omega baryon
h0=70.4 # Hubble constant
zobs=0 #redshift d'observacio
gcos=4.499450941e-15   # G, in Mpc^3/Gyr^2/M_o
gyr=9.778131302e2  #Gyr, in (km/s/Mpc)^{-1}
ih0=gyr/h0   # H_0^{-1} in Gyr
h_z=np.sqrt(h0**2*(omm*(1+zobs)**3+(1-omm))) #constant de hubble per una certa z
ih_z=gyr/h_z 
rhoc=1/(2*np.pi*gcos*ih_z**2) #densitat crítica
rho0=9.9e-27*(3.0857e22)**3/(2e30) #densitat mitjana
#deltac=(rhoc-rho0)/rho0 #perturbació referent a la densitat crítica de colapse
deltac=1.6865 #de moment considero aquest valor de deltac
omWDM=omm-omb
mWDM=3.3
al=2e8
#%%
#CDM spectrum (including tranfer function)
hreal=h0/100
tn=-0.034
n_s=1+tn
gam=omm*hreal*np.exp(-omb-np.sqrt(hreal/0.5)*omb/omm)

def PSlin(x,al):  
    x=x
    q=x/(hreal*gam)
    q234=2.34*q
    if (q234 < 1e-6):
        CDM_cos=x**(1+tn)*(1-0.5*q234)**2/np.sqrt(1+q*(3.89+q*(259.21+q*(162.771336+q*2027.16958081))))
        #print(CDM_cos)
    else:
        CDM_cos=x**(1+tn)*(np.log(1+q234)/q234)**2/np.sqrt(1+q*(3.89+q*(259.21+q*(162.771336+q*2027.16958081))))
        #print(CDM_cos)
    return CDM_cos*al

#%%
#CDM power spectrum (PS) using an array as input
def PSl(k,al):   
    PE=np.zeros(k.size)
    for i in range(0,k.size,1):
        PE[i]=PSlin(k[i],al)
    return PE

#%%
lrr=np.logspace(-8,3)
plt.plot(lrr,PSl(lrr,al))
plt.xscale('log')
plt.yscale('log')
#%%
#Delta (no units PS) for CDM
def Delta(k,al):
    D=np.zeros(k.size)
    for i in range(1,k.size,1):
        D[i]=k[i]**3*PSlin(k[i],al)/(2*np.pi**2)
    return D

#Delta (no units PS) for WDM
def DeltaW(k,al):
    D=np.zeros(k.size)
    #matriu de transefrència Matteo Leo thermal WDM
    alpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
    beta=2*1.12
    gamma=-5/1.12
    
    for i in range(1,k.size,1):
        D[i]=k[i]**3*PSlin(k[i],al)*(1+(alpha*k[i])**beta)**(2*gamma)/(2*np.pi**2)
    return D

link3=np.logspace(-3,3,1000)
plt.plot(np.log10(link3),np.log10(Delta(link3*hreal,1.0e3)))
plt.plot(np.log10(link3),np.log10(DeltaW(link3*hreal,1.0e3)))

plt.xlim([0,2.5])
plt.ylim([(-4),(2)])

plt.xlabel('k ($h/Mpc$)')
plt.ylabel('$\Delta^2$')

#%%
#CDM variance using k-sharp filter
def sigma2linCDM(y,al):
    sigmaquadrat=integrate.quad(lambda x: PSlin(x,al)*x**2/(2*np.pi**2),1e-6,y)
    return sigmaquadrat[0]   
#%%
#WDM variance using k-sharp filter
def sigma2linW(y,mWDM,al):
    #PS WDM
    omWDM=omm-omb
    #WDM transference matrix Viel et. al (2005)
    alpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
    betar=2*1.12
    gamma=-5/1.12

    sigmaquadrat=integrate.quad(lambda x: PSlin(x,al)*x**2*(1+(alpha*x)**betar)**(2*gamma)/(2*np.pi**2),1e-8,y)
    return sigmaquadrat[0]   
#%%
#variance plot for CDM and WDM
lk=np.logspace(-4,3,1000)
lm=4*np.pi*(2.5/lk)**3/3*rho0*omm
lr=1/lk

lsig2CDM=np.zeros(lk.size)
lsig2WDM=np.zeros(lk.size)

for i in range(0,lk.size):
    lsig2CDM[i]=sigma2linCDM(lk[i],2e8)
    lsig2WDM[i]=sigma2linW(lk[i],3.3,2e8)
    
plt.plot(lm,np.sqrt(lsig2CDM))
plt.plot(lm,np.sqrt(lsig2WDM))

plt.xlabel('M ($M_o$)')
plt.ylabel('$\sigma$')
plt.xlim([10**6,10**15])
#plt.ylim([0,9])
plt.xscale('log')

#%%
#variance derivative
liniay=np.logspace(-3,1.6988,num=1000)
h=0.0001

#CDM
def dersig2CDM(k,al):
    deriv=np.zeros(k.size)   
    for l in range(1,k.size,1):
        deriv[l]=PSlin(k[l], al)*k[l]**2/(2*np.pi**2)
    return deriv

#WDM
def dersig2W(k,mWDM,al):
    alpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
    bbeta=2*1.12
    gamma=-5/1.12
    
    deriv=np.zeros(k.size)
    for l in range(0,k.size,1):
        deriv[l]=PSlin(k[l], al)*(1+(alpha*k[l])**bbeta)**(2*gamma)*k[l]**2/(2*np.pi**2)
    return deriv
#%%
#variance derivative plot for CDM and WDM
plt.plot(liniay,dersig2CDM(liniay,3e6))
plt.plot(liniay,dersig2W(liniay,1,3e6))

plt.xlim([0,10])
plt.ylim([0,5])
plt.xlabel('k ($h/Mpc$)')
plt.ylabel('d$\sigma ^2$/d$k$')
#%%
#HMF for CDM
def FCDM(logmh,al):
    #mass and variance derivative
    M=10**logmh/hreal
    k=(4*np.pi*rho0*omm/(3*M))**(1/3)*c
    dsig2=dersig2CDM(k,al)

    Fm=np.zeros(k.size)
    for l in range(1,k.size,1):
        sig2=sigma2linCDM(k[l],al)
        
        #mass fraction Sheth and Tormen (2002)
        nu=deltac**2/sig2
        A=0.3222
        p=0.3
        q=0.707 
        fracc=A*np.sqrt(2*q*nu/(np.pi))*(1+(q*nu)**(-p))*np.exp(-q*nu/2)
        
        Fm[l]=k[l]*rho0*omm*fracc*dsig2[l]/(6*M[l]*sig2*hreal**3)
    return Fm

#%%
#HMF for WDM
def FW(logmh,al,c,mWDM):
    #mass and variance derivative
    M=10**logmh/hreal
    k=(4*np.pi*rho0*omm/(3*M))**(1/3)*c
    dsig2=dersig2W(k,mWDM,al)

    Fm=np.zeros(k.size)
    for l in range(1,k.size,1):
        sig2=sigma2linW(k[l],mWDM,al)
        
        #mass fraction Sheth and Tormen (2002)
        nu=deltac**2/sig2
        A=0.3222
        p=0.3
        q=0.707 
        fracc=A*np.sqrt(2*q*nu/(np.pi))*(1+(q*nu)**(-p))*np.exp(-q*nu/2)
        
        #print(k,rho0,fracc,I[0],M,sig2)
        #funcio de massa
        Fm[l]=k[l]*rho0*omm*fracc*dsig2[l]/(6*M[l]*sig2*hreal**3)
    return Fm
#%%
#simulation data from COCO, Bose et. al (2016)
#%%
#CDM HMF points 
logmCDM_COCO=np.array([7.455,8.024,8.46,8.825,9.233,9.683,10.076,10.456,10.849,11.229,11.608,11.995])
logHMFCDM_COCO=np.array([1.25,0.768,0.379,0.045,-0.317,-0.759,-1.107,-1.429,-1.75,-2.112,-2.46,-2.795])

#3.3keV WDM HMF points
logm33keV=np.array([np.log10(1.692e6),np.log10(4.108e6),7.012,7.237,7.392,7.771,8.151,8.53,8.924,9.317,9.697,10.076,10.456,10.849,11.229,11.608,11.995])
logHMF33keV=np.array([np.log10(1.333e-2),np.log10(4.64e-2),-0.933,-0.397,-0.143,0.045,0.004,-0.089,-0.263,-0.518,-0.799,-1.121,-1.429,-1.75,-2.112,-2.46,-2.795])


plt.plot(10**logmCDM_COCO,10**logHMFCDM_COCO,'o')
plt.plot(10**logm33keV,10**logHMF33keV,'o')

plt.xlim([10**6,10**13])
plt.ylim([10**(-4),100])


plt.xscale('log')
plt.yscale('log')

plt.xlabel('M ($M_o/h$)')
plt.ylabel('$dn/d log M$ [$h^3Mpc^{-3}$]')

#%%
#fitted HMF for the CDM simualtion Bose et.al (2016)
#parameters from Bose et. al (2016)
mWDM=3.3 
c=2.5
al=2e8
m=1

logarmh=np.linspace(5.5,15,1000)
plt.plot(10**logarmh,FCDM(logarmh, al))
plt.plot(10**logarmh,FW(logarmh, al, c,mWDM))

plt.plot(10**logmCDM_COCO,10**logHMFCDM_COCO,'o')
plt.plot(10**logm33keV,10**logHMF33keV,'o')

plt.xlim([10**6,10**13])
plt.ylim([10**(-4),100])


plt.xscale('log')
plt.yscale('log')

plt.xlabel('M ($M_o/h$)')
plt.ylabel('$dn/d log M$ [$h^3Mpc^{-3}$]')

#%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#variance, variance derivative and HMF using several filters: smooth-k, double box and four boxes
#%%
#smooth-k filter from Leo et.al (2018)
def smooth(k,kc,beta):
    R=1/kc
    filtre=(1+(k*R)**beta)**(-1)
    return filtre

#%%
#smooth-k filtered variance
def sigma2s(y,beta):
    #WDM transefr matrix from Viel et. al (2005)
    alpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
    bbeta=2*1.12
    gamma=-5/1.12
    
    
    sigmaquadrat=integrate.quad(lambda x: smooth(x,y,beta)**2*PSlin(x,al)*x**2*(1+(alpha*x)**bbeta)**(2*gamma)/(2*np.pi**2),1e-10,np.infty, epsabs=1e-25, epsrel=1e-25)[0]
    
    #to correct an integration error
    #if sigmaquadrat<1e-5:
        #sigmaquadrat=sigma2linW(y,mWDM, al)
        
    return sigmaquadrat

#%%
#variance for the double box filter
def sigma2caixes2(y,c,a1,a2):

    alpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
    beta=2*1.12
    gamma=-5/1.12
    
    #to avoid complex numbers appearing when fitting
    if a1<-0.999 or a2<-0.999:
        sigmaquadrat=0
    else:
        sigmaquadrat=integrate.quad(lambda x: PSlin(x,al)*x**2*(1+(alpha*x)**beta)**(2*gamma)/(4*np.pi**2),1e-4,(1+a1)*y)[0]+integrate.quad(lambda x: PSlin(x,al)*x**2*(1+(alpha*x)**beta)**(2*gamma)/(4*np.pi**2),1e-4,(1+a2)*y)[0]
    return sigmaquadrat   

#%%
#variance for the four boxes filter
def sigma2caixes4(y,c,a1,a2,a3,a4):

    alpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
    beta=2*1.12
    gamma=-5/1.12
    
    if a1<-0.999 or a2<-0.999 or a3<-0.999 or a4<-0.999:
        sigmaquadrat=0
    else:
        sigmaquadrat=integrate.quad(lambda x: PSlin(x,al)*x**2*(1+(alpha*x)**beta)**(2*gamma)/(8*np.pi**2),1e-4,(1+a1)*y)[0]+integrate.quad(lambda x: PSlin(x,al)*x**2*(1+(alpha*x)**beta)**(2*gamma)/(8*np.pi**2),1e-4,(1+a2)*y)[0]+integrate.quad(lambda x: PSlin(x,al)*x**2*(1+(alpha*x)**beta)**(2*gamma)/(8*np.pi**2),1e-4,(1+a3)*y)[0]+integrate.quad(lambda x: PSlin(x,al)*x**2*(1+(alpha*x)**beta)**(2*gamma)/(8*np.pi**2),1e-4,(1+a4)*y)[0]
    return sigmaquadrat  

#%%
#variance for the smooth Double Box filter
def sigma2caixes2s(y,beta):
    #PS WDM
    #matriu de transefrència Matteo Leo thermal WDM
    alpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
    bbeta=2*1.12
    gamma=-5/1.12
        
    sigmaquadrat=0.75*integrate.quad(lambda x: PSlin(x,al)*x**2*(1+(alpha*x)**bbeta)**(2*gamma)/(2*np.pi**2),1e-6,y)[0]+0.25*integrate.quad(lambda x: PSlin(x,al)*x**2*(1+(alpha*x)**bbeta)**(2*gamma)/(2*np.pi**2),0,y*(1+beta))[0]#+0.5*integrate.quad(lambda x: (y/x)**(2*beta)*PSlin(x,al)*x**2*(1+(alpha*x)**bbeta)**(2*gamma)/(2*np.pi**2),y*2**(1/beta),np.infty)[0]
    return sigmaquadrat   

#%%
def sigma2saprox(y,beta):
    #WDM transefr matrix from Viel et. al (2005)
    alpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
    bbeta=2*1.12
    gamma=-5/1.12
    
    
    sigmaquadrat=integrate.quad(lambda x: (1-(x/y)**beta)**(2)*PSlin(x,al)*x**2*(1+(alpha*x)**bbeta)**(2*gamma)/(2*np.pi**2),1e-8,y)[0]+integrate.quad(lambda x: (y/x)**(2*beta)*PSlin(x,al)*x**2*(1+(alpha*x)**bbeta)**(2*gamma)/(2*np.pi**2),y,1000)[0]
    
    #to correct an integration error
    #if sigmaquadrat<1e-5:
     #   sigmaquadrat=sigma2linW(y,mWDM, al)
        
    return sigmaquadrat

#%%
def sigma2saprox2(y,beta):
    #WDM transefr matrix from Viel et. al (2005)
    alpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
    bbeta=2*1.12
    gamma=-5/1.12
    
    
    sigmaquadrat=integrate.quad(lambda x: (1-2*(x/y)**beta)*PSlin(x,al)*x**2*(1+(alpha*x)**bbeta)**(2*gamma)/(2*np.pi**2),1e-6,y)[0]+integrate.quad(lambda x: (y/x)**(2*beta)*PSlin(x,al)*x**2*(1+(alpha*x)**bbeta)**(2*gamma)/(2*np.pi**2),y,1000)[0]
    
    #to correct an integration error
    #if sigmaquadrat<1e-5:
        #sigmaquadrat=sigma2linW(y,mWDM, al)
        
    return sigmaquadrat

#%%
def sigma2exp(y,beta):
    #WDM transefr matrix from Viel et. al (2005)
    alpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
    bbeta=2*1.12
    gamma=-5/1.12
        
    sigmaquadrat=integrate.quad(lambda x: (np.exp(-(x/y)**beta))**2*PSlin(x,al)*x**2*(1+(alpha*x)**bbeta)**(2*gamma)/(2*np.pi**2),1e-6,np.infty)[0]
        
    return sigmaquadrat

#%%
def sigma2tgh(y,beta):
    #WDM transefr matrix from Viel et. al (2005)
    alpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
    bbeta=2*1.12
    gamma=-5/1.12
        
    sigmaquadrat=integrate.quad(lambda x: (1-np.tanh((x/y)**beta))**2*PSlin(x,al)*x**2*(1+(alpha*x)**bbeta)**(2*gamma)/(2*np.pi**2),1e-6,np.infty)[0]
        
    return sigmaquadrat
#%%
#1/k filter
def sigma2inv(y,beta):
    #WDM transefr matrix from Viel et. al (2005)
    alpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
    bbeta=2*1.12
    gamma=-5/1.12
    
    
    sigmaquadrat=integrate.quad(lambda x: (1.5*(5/(3*np.cosh(x/(y*beta)))-np.sinh(x/(y*beta))/(x/(y*beta)*(np.cosh(x/(y*beta)))**2)))**2*PSlin(x,al)*x**2*(1+(alpha*x)**bbeta)**(2*gamma)/(2*np.pi**2),1e-10,np.infty, epsabs=1e-30, epsrel=1e-30)[0]
    
    #to correct an integration error
    #if sigmaquadrat<1e-5:
        #sigmaquadrat=sigma2linW(y,mWDM, al)
        
    return sigmaquadrat

#%%
#variance plots using several filters

lk=np.logspace(-5,5,1000)
lm=4*np.pi*(c/lk)**3/3*rho0*omm
lr=1/lk

lsig2CDM=np.zeros(lk.size)
lsig2WDM=np.zeros(lk.size)
lsig2WDMs=np.zeros(lk.size)
lsig2WDMsaprox=np.zeros(lk.size)
lsig2WDMsaprox2=np.zeros(lk.size)
lsig2WDMc2=np.zeros(lk.size)
lsig2WDMc2s=np.zeros(lk.size)
lsig2exp=np.zeros(lk.size)
lsig2inv=np.zeros(lk.size)

for i in range(0,lk.size):
    #lsig2CDM[i]=sigma2linCDM(lk[i],al)
    #lsig2WDM[i]=sigma2linW(lk[i],mWDM,al)#*(1-(-15+6)/(-15+3-2*6)/2)*2**((-15+3)/6)
    lsig2WDMs[i]=sigma2s(lk[i],2)
    #lsig2WDMc2[i]=sigma2caixes2(lk[i],c,-0.19,0.48)
    #lsig2WDMc2s[i]=sigma2caixes2s(lk[i],-0.403)
    #lsig2WDMsaprox[i]=sigma2saprox(lk[i],6.35)
    #lsig2WDMsaprox2[i]=sigma2saprox2(lk[i],6.35)
    #lsig2exp[i]=sigma2exp(lk[i])
    lsig2inv[i]=sigma2inv(lk[i],1)
    
#plt.plot(np.log10(lm),np.log10(lsig2CDM))
#plt.plot(np.log10(lm),np.log10(lsig2WDM))
plt.plot(np.log10(lm),np.log10(lsig2WDMs))
plt.plot(np.log10(lm),np.log10(lsig2inv))
#plt.plot(np.log10(lm),np.log10(lsig2WDMsaprox))
#plt.plot(np.log10(lm),np.log10(lsig2WDMsaprox2))
#plt.plot(np.log10(lm),np.log10(lsig2WDMc2))
#plt.plot(np.log10(lm),np.log10(lsig2WDMc2s))

plt.xlabel('M ($M_o$)')
plt.ylabel('$\sigma ^2$')

plt.xlim([4,15])
plt.ylim([-1,4])

#%%
print(lsig2WDM[0:850]-lsig2WDMs[0:850])
#%%
#variance derivative for the studied filters apart of k-sharp
liniay=np.logspace(-8,4,num=1000)
h=0.0001

def dersig2s(k,h,beta):
    deriv=np.zeros(k.size)
    for l in range(1,k.size,1):
        if k[l]<510:
            deriv[l]=(sigma2s(k[l]+h,beta)-sigma2s(k[l]-h,beta))/(2*h)
            indd=l
            
    alpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
    bbeta=2*1.12
    gamma=-5/1.12
    
    for l in range(indd,k.size,1):
        if k[l]>510:
            #deriv[l]=PSlin(k[l], al)*(1+(alpha*k[l])**bbeta)**(2*gamma)*k[l]**2*beta*(-1/(72*(1-2/beta)**(16))+(1/2+beta/4)/(19*(1-2/beta)**(17)))/(4*np.pi**2)
            #deriv[l]=integrate.quad(lambda x: 2*beta*(x/k[l])**(beta)*(1/k[l])*PSlin(x,al)*x**2*(1+(alpha*x)**bbeta)**(2*gamma)/(2*np.pi**2*k[l]),0, np.inf, epsabs=1e-25, epsrel=1e-25)[0]
            deriv[l]=integrate.quad(lambda x: 2*beta*(x**beta/k[l]**(beta+1))/(1+(x/k[l])**beta)**3*PSlin(x,al)*x**2*(1+(alpha*x)**bbeta)**(2*gamma)/(2*np.pi**2*k[l]),0, np.inf, epsabs=1e-25, epsrel=1e-25)[0]
    return deriv


 
def dersig2c2(k,c,a1,a2):
    params=np.array([a1,a2])
    deriv=np.zeros(k.size)
    
    alpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
    beta=2*1.12
    gamma=-5/1.12
       
    for l in range(0,k.size,1):
        for i in range(0,params.size,1):
            deriv[l]=deriv[l]+PSlin(k[l]*(1+params[i]), al)*(1+(alpha*k[l]*(1+params[i]))**beta)**(2*gamma)*(k[l]*(1+params[i]))**2/(2*params.size*np.pi**2) #+PSlin(k[l]*(1+a2), al, m)*(1+(alpha*(k[l]*(1+a2)))**beta)**(2*gamma)*(k[l]*(1+a2))**2/(4*np.pi**2)
    return deriv

def dersig2c4(k,c,a1,a2,a3,a4):
    params=np.array([a1,a2,a3,a4])
    deriv=np.zeros(k.size)
    
    #matriu de transefrència Matteo Leo thermal WDM
    alpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
    beta=2*1.12
    gamma=-5/1.12
        
    for l in range(0,k.size,1):
        for i in range(0,params.size,1):
            deriv[l]=deriv[l]+PSlin(k[l]*(1+params[i]), al)*(1+(alpha*k[l]*(1+params[i]))**beta)**(2*gamma)*(k[l]*(1+params[i]))**2/(2*params.size*np.pi**2) #+PSlin(k[l]*(1+a2), al, m)*(1+(alpha*(k[l]*(1+a2)))**beta)**(2*gamma)*(k[l]*(1+a2))**2/(4*np.pi**2)
    return deriv

def dersig2c2s(k,h,beta):
    #deriv=np.zeros(k.size)
    #for l in range(1,k.size,1):
    #    deriv[l]=(sigma2caixes2s(k[l]+h,beta)-sigma2caixes2s(k[l]-h,beta))/(2*h)
    #return deriv
    
    alpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
    bbeta=2*1.12
    gamma=-5/1.12
    
    deriv=np.zeros(k.size)
    for l in range(0,k.size,1):
        q=k[l]*(1+beta)
        deriv[l]=(0.75*PSlin(k[l], al)*(1+(alpha*k[l])**bbeta)**(2*gamma)*k[l]**2+0.25*PSlin(q,al)*(1+(alpha*q)**bbeta)**(2*gamma)*k[l]**2*(1+beta)**3)/(2*np.pi**2)
    return deriv

def dersig2exp(k,beta):
    deriv=np.zeros(k.size)
    for l in range(1,k.size,1):
        if k[l]<510:
            deriv[l]=(sigma2exp(k[l]+h,beta)-sigma2exp(k[l]-h,beta))/(2*h)
            indd=l
            
    alpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
    bbeta=2*1.12
    gamma=-5/1.12
    
    for l in range(indd,k.size,1):
        if k[l]>510:
            #deriv[l]=integrate.quad(lambda x: 2*beta*(x/k[l])**(beta)*(1-1/k[l])*PSlin(x,al)*x**2*(1+(alpha*x)**bbeta)**(2*gamma)/(2*np.pi**2*k[l]),0, np.inf, epsabs=1e-25, epsrel=1e-25)[0]
            deriv[l]=integrate.quad(lambda x: 2*beta*(x**beta/k[l]**(beta+1))/(np.exp(-(x/k[l])**beta))**2*PSlin(x,al)*x**2*(1+(alpha*x)**bbeta)**(2*gamma)/(2*np.pi**2*k[l]),0, np.inf, epsabs=1e-25, epsrel=1e-25)[0]
    return deriv

def dersig2tgh(k,beta):
    deriv=np.zeros(k.size)
    for l in range(1,k.size,1):
        if k[l]<510:
            deriv[l]=(sigma2tgh(k[l]+h,beta)-sigma2tgh(k[l]-h,beta))/(2*h)
            indd=l
            
    alpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
    bbeta=2*1.12
    gamma=-5/1.12
    
    for l in range(indd,k.size,1):
        if k[l]>510:
            #deriv[l]=integrate.quad(lambda x: 2*beta*(x/k[l])**(beta)*(1-1/k[l])*PSlin(x,al)*x**2*(1+(alpha*x)**bbeta)**(2*gamma)/(2*np.pi**2*k[l]),0, np.inf, epsabs=1e-25, epsrel=1e-25)[0]
            deriv[l]=integrate.quad(lambda x: 2*beta*(x**beta/k[l]**(beta+1))*(1-np.tanh((x/k[l])**beta))/(np.cosh((x/k[l])**beta))**2*PSlin(x,al)*x**2*(1+(alpha*x)**bbeta)**(2*gamma)/(2*np.pi**2*k[l]),0, np.inf, epsabs=1e-25, epsrel=1e-25)[0]
    return deriv

def dersig2inv(k,beta):
    deriv=np.zeros(k.size)
            
    alpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
    bbeta=2*1.12
    gamma=-5/1.12
    
    for l in range(0,k.size,1):
        deriv[l]=integrate.quad(lambda x: (x/k[l]**2)/np.cosh(x/(k[l]*beta))*PSlin(x,al)*x**2*(1+(alpha*x)**bbeta)**(2*gamma)/(2*np.pi**2*k[l]),0, np.inf, epsabs=1e-30, epsrel=1e-30)[0]
    return deriv

#%%
#variance derivative plot for CDM and WDM k-sharp and smooth-k and Double Box
#plt.plot(liniay,dersig2CDM(liniay,al))
#plt.plot(liniay,dersig2W(liniay,mWDM,al))
plt.plot(liniay,dersig2s(liniay,h,6.1))
plt.plot(liniay,dersig2inv(liniay,3))
#plt.plot(liniay,dersig2c2(liniay,c,0,0))
#plt.plot(liniay,dersig2c2s(liniay,h,-0.404)/dersig2W(liniay,mWDM,al))

plt.xlabel('k ($h/Mpc$)')
plt.ylabel('d$\sigma ^2$/d$k$')

#plt.xscale('log')
#plt.yscale('log')

plt.xlim([0,100])
plt.ylim([0,80])
#%%
#The following cells are the fitting functions for different filters
#%%
#interpolation using cubic spline to have a greater set of points to work which fit with simulations
def logHMF33COCO(logm):
    logHMF=np.zeros(logm.size)
    for i in range(1,logm.size,1):
        cs=CubicSpline(logm33keV,logHMF33keV)
        logHMF[i]=cs(logm[i])
    return logHMF

logmlinia=np.linspace(7,12,1000)
plt.plot(10**logmlinia,10**logHMF33COCO(logmlinia))
plt.plot(10**logmCDM_COCO,10**logHMFCDM_COCO,'o')
plt.plot(10**logm33keV,10**logHMF33keV,'o')
plt.xlim([10**7,10**12])
plt.ylim([10**(-4),100])

plt.xscale('log')
plt.yscale('log')
plt.xlabel('M ($M_o/h$)')
plt.ylabel('$dn/d log M$ [$h^3Mpc^{-3}$]')
#%%
#HMF function for a smooth-k filter
def FWs(logmh,c,beta):
    #beta=8
    #mass, wavenumber and variance derivative
    M=10**logmh/hreal
    k=(4*np.pi*rho0*omm/(3*M))**(1/3)*c
    dsig2=dersig2s(k,0.0001,beta)

    Fm=np.zeros(k.size)
    for l in range(1,k.size,1):
        sig2=sigma2s(k[l],beta)
        
        #mass fraction from Viel  et. al (2005)
        nu=deltac**2/sig2
        A=0.3222
        p=0.3
        q=0.707 
        fracc=A*np.sqrt(2*q*nu/(np.pi))*(1+(q*nu)**(-p))*np.exp(-q*nu/2)
        
        #HMF
        Fm[l]=k[l]*rho0*omm*fracc*dsig2[l]/(6*M[l]*sig2*hreal**3)
    return Fm
#%%
#fitting code for the smooth-k filter
ff=10**logHMF33keV

x_data=logm33keV
y_data=ff

popt, pcovariance = curve_fit(FWs, x_data, y_data, p0=[2.5,5])

plt.plot(logm33keV,10**logHMF33keV,'o')

logarm=np.linspace(4,13,1000)
plt.plot(logarm,FWs(logarm, popt[0], popt[1]))
#plt.plot(logarm,FWs(logarm, 3.4, 6.35))

#quan es fa l'ajust per als paràmetres de localització, cal guardar-los per tal que no s'acumulin en els ajustos posteriors 
cs=popt[0]
betas=popt[1]

plt.xlabel('M ($M_o/h$)')
plt.ylabel('$dn/d log M$ [$h^3Mpc^{-3}$]')
plt.yscale('log')

#%%
#HMF function for a smooth-k filter fixing the beta=20
def FWss(logmh,c):
    #beta=9
    #mass, wavenumber and variance derivative
    M=10**logmh/hreal
    k=(4*np.pi*rho0*omm/(3*M))**(1/3)*c
    dsig2=dersig2s(k,0.0001,beta)

    Fm=np.zeros(k.size)
    for l in range(1,k.size,1):
        sig2=sigma2s(k[l],beta)
        
        #if sig2==0:
            #print(k[l])
            
        #mass fraction from Viel  et. al (2005)
        nu=deltac**2/sig2
        
        A=0.3222
        p=0.3
        q=0.707 
        fracc=A*np.sqrt(2*q*nu/(np.pi))*(1+(q*nu)**(-p))*np.exp(-q*nu/2)
        
        #HMF
        Fm[l]=k[l]*rho0*omm*fracc*dsig2[l]/(6*M[l]*sig2*hreal**3)
        #Fm[l]=k[l]*rho0*omm*fracc/(6*M[l]*sig2*hreal**3)
    return Fm
#%%
#fitting code for the smooth-k filter
ff=10**logHMF33keV

x_data=logm33keV
y_data=ff

popt, pcovariance = curve_fit(FWss, x_data, y_data, p0=[3])

plt.plot(logm33keV,10**logHMF33keV,'o')

logarm=np.linspace(6,13,1000)
plt.plot(logarm,FWss(logarm,3))
#plt.plot(logarm,FWs(logarm, 3.4, 6.35))
#plt.plot(logarm,FW(logarm*(1), al, 2.5, mWDM))

#quan es fa l'ajust per als paràmetres de localització, cal guardar-los per tal que no s'acumulin en els ajustos posteriors 
print(popt[0])

plt.xlabel('M ($M_o/h$)')
plt.ylabel('$dn/d log M$ [$h^3Mpc^{-3}$]')
plt.yscale('log')

#%%
#HMF for fitting Double Box filter
def FWc2(logmh,a1,a2,c):
    #mass, wavenumber and variance derivative
    M=10**logmh/hreal
    k=(4*np.pi*rho0*omm/(3*M))**(1/3)*(c)
    dsig2=dersig2c2(k, c, a1, a2)

    Fm=np.zeros(k.size)
    for l in range(1,k.size,1):
        sig2=sigma2caixes2(k[l], c, a1, a2)
        if sig2 <1e-8:
            fracc=0
        else:
            #mass fraction from Viel  et. al (2005)
            nu=deltac**2/sig2
            A=0.3222
            p=0.3
            q=0.707 
            fracc=A*np.sqrt(2*q*nu/(np.pi))*(1+(q*nu)**(-p))*np.exp(-q*nu/2)
        
        
        #HMF
        Fm[l]=k[l]*rho0*omm*fracc*dsig2[l]/(6*M[l]*sig2*hreal**3)
    return Fm
#%%
#fitting code for the Double Box filter
ff=10**logHMF33keV

x_data=logm33keV
y_data=ff

popt, pcovariance = curve_fit(FWc2, x_data, y_data, p0=[-0.22,0.44,2.7])

plt.plot(logm33keV,logHMF33keV,'o')

logarm=np.linspace(6,13,1000)
plt.plot(logarm,np.log10(FWc2(logarm, popt[0], popt[1], popt[2])))
#plt.plot(logarm,np.log10(FWc2(logarm, 0, 0)))
#quan es fa l'ajust per als paràmetres de localització, cal guardar-los per tal que no s'acumulin en els ajustos posteriors 
a1=popt[0]
a2=popt[1]
cc2=popt[2]
plt.yscale('log')
#%%
#HMF for the four boxes filter
def FWc4(logmh,a1,a2,a3,a4,c):
    #mass, wavenumber and variance derivative
    M=10**logmh/hreal
    k=(4*np.pi*rho0*omm/(3*M))**(1/3)*c
    dsig2=dersig2c4(k, c, a1, a2, a3, a4)

    Fm=np.zeros(k.size)
    for l in range(1,k.size,1):
        sig2=sigma2caixes4(k[l], c, a1, a2, a3, a4)
        
        if sig2<1e-10:
            fracc=0
        else:
            #mass fraction from Viel  et. al (2005)
            nu=deltac**2/sig2    
            A=0.3222
            p=0.3
            q=0.707 
            fracc=A*np.sqrt(2*q*nu/(np.pi))*(1+(q*nu)**(-p))*np.exp(-q*nu/2)
        
        #HMF
        Fm[l]=k[l]*rho0*omm*fracc*dsig2[l]/(6*M[l]*sig2*hreal**3)
    return Fm
#%%
#fitting code for the four boxes filter
ff=10**logHMF33keV

x_data=logm33keV
y_data=ff

popt, pcovariance = curve_fit(FWc4, x_data, y_data, p0=[0.1,0.1,0.1,-0.1,2.5])

plt.plot(logm33keV,logHMF33keV,'o')

logarm=np.linspace(6,13,1000)
plt.plot(logarm,np.log10(FWc4(logarm, popt[0], popt[1], popt[2], popt[3], popt[4])))

#saving the boxes size
b1=popt[0]
b2=popt[1]
b3=popt[2]
b4=popt[3]

cc4=popt[4]

#%%
#smooth double box filter
def FWc2s(logmh,c,beta):
    #mass, wavenumber and variance derivative
    M=10**logmh/hreal
    k=(4*np.pi*rho0*omm/(3*M))**(1/3)*c
    dsig2=dersig2c2s(k,h,beta)

    Fm=np.zeros(k.size)
    for l in range(1,k.size,1):
        sig2=sigma2caixes2s(k[l],beta)
        
       #mass fraction from Viel  et. al (2005)
        nu=deltac**2/sig2
        A=0.3222
        p=0.3
        q=0.707 #0.707 si es vol corregir Matteo Leo pg 4 ajusta com Warren
        fracc=A*np.sqrt(2*q*nu/(np.pi))*(1+(q*nu)**(-p))*np.exp(-q*nu/2)
        
        #HMF
        Fm[l]=k[l]*rho0*omm*fracc*dsig2[l]/(6*M[l]*sig2*hreal**3)
    return Fm

#%%
#fit for the smooth Double Box filter
ff=10**logHMF33keV

x_data=logm33keV
y_data=ff

popt, pcovariance = curve_fit(FWc2s, x_data, y_data, p0=[3.33,-0.4])
plt.plot(logm33keV,10**logHMF33keV,'o')

logarm=np.linspace(4,13,2000)
plt.plot(logarm,FWc2s(logarm, popt[0], popt[1]))
#plt.plot(logarm,dersig2c2s((3.33 * (4 * np.pi * rho0 * omm / (3 * 10**logarm / hreal))**(1/3)), h, -0.404) /dersig2W((c * (4 * np.pi * rho0 * omm / (3 * 10**logarm / hreal))**(1/3)),mWDM,al))
plt.plot(logarm,FW(logarm*(1), al, 2.5, mWDM))
#when the fit is done the parameters c and beta are saved in
c2s=popt[0]
beta2s=popt[1]

plt.xlabel('M ($M_o/h$)')
plt.ylabel('$dn/d log M$ [$h^3Mpc^{-3}$]')
plt.yscale('log')

#%%
#HMF function for a smooth-k filter
def FWexp(logmh,c,beta):
    #beta=8
    #mass, wavenumber and variance derivative
    M=10**logmh/hreal
    k=(4*np.pi*rho0*omm/(3*M))**(1/3)*c
    dsig2=dersig2exp(k,beta)

    Fm=np.zeros(k.size)
    for l in range(1,k.size,1):
        sig2=sigma2exp(k[l],beta)
        
        #mass fraction from Viel  et. al (2005)
        nu=deltac**2/sig2
        A=0.3222
        p=0.3
        q=0.707 
        fracc=A*np.sqrt(2*q*nu/(np.pi))*(1+(q*nu)**(-p))*np.exp(-q*nu/2)
        
        #HMF
        Fm[l]=k[l]*rho0*omm*fracc*dsig2[l]/(6*M[l]*sig2*hreal**3)
    return Fm

#%%
#fit for the smooth Double Box filter
ff=10**logHMF33keV

x_data=logm33keV
y_data=ff

popt, pcovariance = curve_fit(FWexp, x_data, y_data, p0=[3.33,6])
plt.plot(logm33keV,10**logHMF33keV,'o')

logarm=np.linspace(4,13,2000)
plt.plot(logarm,FWexp(logarm, popt[0], popt[1]))
#plt.plot(logarm,dersig2c2s((3.33 * (4 * np.pi * rho0 * omm / (3 * 10**logarm / hreal))**(1/3)), h, -0.404) /dersig2W((c * (4 * np.pi * rho0 * omm / (3 * 10**logarm / hreal))**(1/3)),mWDM,al))
#plt.plot(logarm,FW(logarm*(1), al, 2.5, mWDM))
#when the fit is done the parameters c and beta are saved in
cexp=popt[0]
betaexp=popt[1]

plt.xlabel('M ($M_o/h$)')
plt.ylabel('$dn/d log M$ [$h^3Mpc^{-3}$]')
plt.yscale('log')

#%%
#HMF function for a smooth-k filter
def FWtgh(logmh,c,beta):
    #mass, wavenumber and variance derivative
    M=10**logmh/hreal
    k=(4*np.pi*rho0*omm/(3*M))**(1/3)*c
    dsig2=dersig2tgh(k,beta)

    Fm=np.zeros(k.size)
    for l in range(1,k.size,1):
        sig2=sigma2tgh(k[l],beta)
        
        #mass fraction from Viel  et. al (2005)
        nu=deltac**2/sig2
        A=0.3222
        p=0.3
        q=0.707 
        fracc=A*np.sqrt(2*q*nu/(np.pi))*(1+(q*nu)**(-p))*np.exp(-q*nu/2)
        
        #HMF
        Fm[l]=k[l]*rho0*omm*fracc*dsig2[l]/(6*M[l]*sig2*hreal**3)
    return Fm

#%%
#fit for the smooth Double Box filter
ff=10**logHMF33keV

x_data=logm33keV
y_data=ff

popt, pcovariance = curve_fit(FWtgh, x_data, y_data, p0=[3.33,6])
plt.plot(logm33keV,10**logHMF33keV,'o')

logarm=np.linspace(4,13,2000)
plt.plot(logarm,FWtgh(logarm, popt[0], popt[1]))
#plt.plot(logarm,dersig2c2s((3.33 * (4 * np.pi * rho0 * omm / (3 * 10**logarm / hreal))**(1/3)), h, -0.404) /dersig2W((c * (4 * np.pi * rho0 * omm / (3 * 10**logarm / hreal))**(1/3)),mWDM,al))
#plt.plot(logarm,FW(logarm*(1), al, 2.5, mWDM))
#when the fit is done the parameters c and beta are saved in
ctgh=popt[0]
betatgh=popt[1]

plt.xlabel('M ($M_o/h$)')
plt.ylabel('$dn/d log M$ [$h^3Mpc^{-3}$]')
plt.yscale('log')
#%%
#coparació entre filtre SK i exponencial
plt.plot(logm33keV,10**logHMF33keV,'o')
plt.plot(logarm,FWexp(logarm, 2.5, 50))
plt.plot(logarm,FWs(logarm, 2.8, 22))
#plt.plot(logarm,FWtgh(logarm, 2.5, 100))
plt.plot(logarm,FW(logarm,al,c,mWDM))

plt.xlabel('M ($M_o/h$)')
plt.ylabel('$dn/d log M$ [$h^3Mpc^{-3}$]')
plt.yscale('log')
#%%
def FWfit(logmh,c):
    #mass and variance derivative
    M=10**logmh/hreal
    k=(4*np.pi*rho0*omm/(3*M))**(1/3)*c
    dsig2=dersig2W(k,mWDM,al)

    Fm=np.zeros(k.size)
    for l in range(1,k.size,1):
        sig2=sigma2linW(k[l],mWDM,al)
        
        #mass fraction Sheth and Tormen (2002)
        nu=deltac**2/sig2
        A=0.3222
        p=0.3
        q=0.707 
        fracc=A*np.sqrt(2*q*nu/(np.pi))*(1+(q*nu)**(-p))*np.exp(-q*nu/2)
        
        #print(k,rho0,fracc,I[0],M,sig2)
        #funcio de massa
        Fm[l]=k[l]*rho0*omm*fracc*dsig2[l]/(6*M[l]*sig2*hreal**3)
    return Fm
#%%
#fitting code for the smooth-k filter
ff=10**logHMF33keV

x_data=logm33keV
y_data=ff

popt, pcovariance = curve_fit(FWfit, x_data, y_data, p0=[2.5])

plt.plot(logm33keV,10**logHMF33keV,'o')

logarm=np.linspace(6,13,1000)
plt.plot(logarm,FWfit(logarm,popt[0]))
#plt.plot(logarm,FWs(logarm, 3.4, 6.35))
plt.plot(logarm,FW(logarm, al, 2.5, mWDM))

#quan es fa l'ajust per als paràmetres de localització, cal guardar-los per tal que no s'acumulin en els ajustos posteriors 
print(popt[0])

cfit=popt[0]

plt.xlabel('M ($M_o/h$)')
plt.ylabel('$dn/d log M$ [$h^3Mpc^{-3}$]')
plt.yscale('log')

#%% 
plt.plot((3.33 * (4 * np.pi * rho0 * omm / (3 * 10**logarm / hreal))**(1/3)),dersig2c2s((3.33 * (4 * np.pi * rho0 * omm / (3 * 10**logarm / hreal))**(1/3)), h, -0.404) /dersig2W((2.5 * (4 * np.pi * rho0 * omm / (3 * 10**logarm / hreal))**(1/3)),mWDM,al))

plt.plot(lk, np.abs(dersig2c2s(lk,h, -0.402)-dersig2W(lk, mWDM, al)) /(dersig2W(lk, mWDM, al)), color=coolwarm(1.0))
plt.xscale('log')
plt.yscale('log')
#%%
#in this cell the deviation from the interpolation of simulations is calculated for
#the five used filters
err_s=0 
err_c2=0 
err_c4=0
err_c2s=0
err_W=0

#using the function defined above to obtain a big set of interpolated poits from simulations
logmd=np.linspace(7.4,12,1000)
logHMFd=logHMF33COCO(logmd)

#initialyzing the arrays for the deviation
lerr_s=np.zeros(logmd.size)
#lerr_c2=np.zeros(logmd.size)
#lerr_c4=np.zeros(logmd.size)
lerr_W=np.zeros(logmd.size)
lerr_c2s=np.zeros(logmd.size)

llerr_s=np.zeros(logmd.size)
#lerr_c2=np.zeros(logmd.size)
#lerr_c4=np.zeros(logmd.size)
llerr_W=np.zeros(logmd.size)
llerr_c2s=np.zeros(logmd.size)


#initializing the arrays for the deviation with respect to the K-sharp filter
lerr_s_W=np.zeros(logmd.size)
lerr_W_W=np.zeros(logmd.size)
lerr_c2s_W=np.zeros(logmd.size)

#initialyzing the values for the mean squared error (SE)
mse_s=0
mse_W=0
mse_c2s=0

#accumulated deviation
lacum_W=np.zeros(logmd.size)
lacum_s=np.zeros(logmd.size)
lacum_c2s=np.zeros(logmd.size)

#generatinng the HMF arrays for each filter
FWsl=FWs(logmd, cs, betas)
#FWc2l=FWc2(logmd, a1, a2, cc2)
#FWc4l=FWc4(logmd, b1,b2,b3,b4,cc4)
FWl=FW(logmd,al,c,mWDM)
FWc2sl=FWc2s(logmd, c2s, beta2s)

#sum of each point deviation
for i in range(1,logmd.size,1):
    err_W=err_W+((10**logHMFd[i]-FWl[i])/(10**logHMFd[i]))**2
    err_s=err_s+((10**logHMFd[i]-FWsl[i])/(10**logHMFd[i]))**2
    #err_c2=err_c2+((10**logHMFd[i]-FWc2l[i])/(10**logHMFd[i]))**2
    #err_c4=err_c4+((10**logHMFd[i]-FWc4l[i])/(10**logHMFd[i]))**2
    err_c2s=err_c2s+((10**logHMFd[i]-FWc2sl[i])/(10**logHMFd[i]))**2
    
    mse_W=mse_W+(10**logHMFd[i]-FWl[i])**2/logmd.size
    mse_s=mse_s+(10**logHMFd[i]-FWsl[i])**2/logmd.size
    mse_c2s=mse_c2s+(10**logHMFd[i]-FWc2sl[i])**2/logmd.size
    
    lerr_s[i]=np.abs((10**logHMFd[i]-FWsl[i])/(10**logHMFd[i]))
    #lerr_c2[i]=((10**logHMFd[i]-FWc2l[i])/(10**logHMFd[i]))**2
    #lerr_c4[i]=((10**logHMFd[i]-FWc4l[i])/(10**logHMFd[i]))**2
    lerr_W[i]=np.abs((10**logHMFd[i]-FWl[i])/(10**logHMFd[i]))
    lerr_c2s[i]=np.abs((10**logHMFd[i]-FWc2sl[i])/(10**logHMFd[i]))
    
    llerr_s[i]=((10**logHMFd[i]-FWsl[i])/(10**logHMFd[i]))
    #lerr_c2[i]=((10**logHMFd[i]-FWc2l[i])/(10**logHMFd[i]))**2
    #lerr_c4[i]=((10**logHMFd[i]-FWc4l[i])/(10**logHMFd[i]))**2
    llerr_W[i]=((10**logHMFd[i]-FWl[i])/(10**logHMFd[i]))
    llerr_c2s[i]=((10**logHMFd[i]-FWc2sl[i])/(10**logHMFd[i]))
    
    lerr_s_W[i]=((FWl[i]-FWsl[i])/FWl[i])**2
    #lerr_c2[i]=((10**logHMFd[i]-FWc2l[i])/(10**logHMFd[i]))**2
    #lerr_c4[i]=((10**logHMFd[i]-FWc4l[i])/(10**logHMFd[i]))**2
    lerr_W_W[i]=((FWl[i]-FWl[i])/FWl[i])**2
    lerr_c2s_W[i]=((FWl[i]-FWc2sl[i])/FWl[i])**2
    
    lacum_W[i]=err_W
    lacum_s[i]=err_s
    lacum_c2s[i]=err_c2s
#%%  
print(err_W/err_s,err_s/err_s,err_c2s/err_s,err_W/logmd.size,err_s/logmd.size,err_c2s/logmd.size)    

#%%
print(mse_W,mse_s,mse_c2s,np.average(lerr_s),np.average(lerr_c2s))
#%%
#HMF (Halo Mass Function plot) for different filters
logmd2=np.linspace(6.95,12,100)
#logmd2=np.linspace(5,8,30)

logHMFd2=logHMF33COCO(logmd2)

#plot dels diferents filtres
plt.plot(10**logmd2,10**logHMFd2,'o',color='violet')
plt.plot(10**logmd2,FWs(logmd2, cs, betas))
#plt.plot(10**logmd,FWc2(logmd, a1, a2, cc2))
plt.plot(10**logmd2,FW(logmd2,al,c,mWDM))
#plt.plot(10**logmd,FWc4(logmd, b1,b2,b3,b4, cc4))
plt.plot(10**logmd2,FWc2s(logmd2, c2s, beta2s))
plt.plot(10**logmd2, (1-5e-16*(10**logmd2)**(betas/3))*(10**logmd2)**1.1*1e-8)
plt.xlim([10**6,10**12])
plt.ylim([10**(-4),10**(2)])

plt.xscale('log')
plt.yscale('log')

plt.xlabel('M ($M_o/h$)')
plt.ylabel('$dn/d log M$ [$h^3Mpc^{-3}$]')

#%%
#cinquena figura article
#the following two cells are a combined figure for the deviation and the values of HMF
#including the SK and for a wider range
logmd=np.linspace(4,12,1000)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

# Import necessary libraries
from matplotlib import cm

# Get the 'coolwarm' colormap for consistent styling
coolwarm = cm.get_cmap('coolwarm')
magma = cm.get_cmap('YlGnBu')

fig = plt.figure(figsize=(5.8, 16))
# Set height ratios for subplots
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

# === First Subplot ===
ax0 = plt.subplot(gs[0])

# Set axis limits and ticks
ax0.set_xlim([5 * 10**4, 10**9])
ax0.set_ylim([8 * 10**(-9), 2.5])
#ax0.set_yticks([0.1, 0.5, 1.0, 1.5])

# Add shaded region with lower z-order than the grid
ax0.axvspan(10**4, 1.8*10**7, color=(0.9, 0.9, 0.9, 0.6), zorder=1)

# Add gridlines for improved readability
ax0.grid(visible=True, which='major', linestyle='-', linewidth=0.5, color=(0.75, 0.75, 0.75, 0.6), zorder=2)

# Logarithmic x-scale
ax0.set_xscale("log")
ax0.set_yscale('log')

# Plot data with higher z-order to appear above the grid
#line0, = ax0.plot(10**logm33keV[2:],10**logHMF33keV[2:],'x', color='orange', linewidth='2', markersize=6, markeredgewidth=2, zorder=3)
line1, = ax0.plot(10**logmd, FWs(logmd, cs, 41), color=coolwarm(0.0), label='Smooth-k', linewidth='2', zorder=3)
line2, = ax0.plot(10**logmd, FWc2s(logmd, c2s, beta2s), color=coolwarm(1.0), label='Double Box', linewidth='2', zorder=3)
line3, = ax0.plot(10**logmd, FW(logmd, al, c, mWDM), color=magma(0.35), label='K-sharp', linewidth='2', zorder=3)
line0, = ax0.plot(10**logm33keV[2:],10**logHMF33keV[2:],'x', color='orange', linewidth='2', markersize=6, markeredgewidth=2, label='COCO', zorder=3)
# === Custom Tick Formatting for Logarithmic Scale ===
# Specify the ticks you want to show
y_ticks = [1e-7, 1e-5, 1e-3, 1e-1]  
ax0.set_yticks(y_ticks)      # Set the ticks explicitly

# Optionally format the tick labels
ax0.set_yticklabels([f'$10^{{{int(np.log10(y))}}}$' for y in y_ticks])

# Remove minor ticks
ax0.yaxis.set_minor_locator(plt.NullLocator())
ax0.xaxis.set_minor_locator(plt.NullLocator())

# Major tick settings
ax0.tick_params(axis='both', labelsize=14, direction='in', which='major', length=6, width=1)

ax0.set_ylabel('d$n$ $/$ d$\\log M$ [$h^3 \\mathrm{Mpc}^{-3}$]', fontsize=14)

# Hide x-axis labels for this subplot
plt.setp(ax0.get_xticklabels(), visible=False)

# Add legend
ax0.legend(fontsize=13, loc='lower right')

# === Adjust spacing between subplots ===
plt.subplots_adjust(hspace=0.1)

# ========== Second Subplot ===========
ax1 = plt.subplot(gs[1], sharex=ax0)

# Set y-axis scale and limits
ax1.set_yscale('log')
ax1.set_ylim([1e-3, 5e9])

# Add shaded region with lower z-order
ax1.axvspan(10**4, 1.8*10**7, color=(0.9, 0.9, 0.9, 0.6), zorder=1)

# Add gridlines for this subplot
ax1.grid(visible=True, which='major', linestyle='-', linewidth=0.5, color=(0.75, 0.75, 0.75, 0.6), zorder=2)

# Plot lines with updated colors
u = 0
line5, = ax1.plot(10**logmd[u:1000], np.abs(FWs(logmd[u:1000], cs, betas) / FW(logmd[u:1000], al, 2.5, mWDM) - 1), color=coolwarm(0.0), linewidth='2', zorder=3)
line6, = ax1.plot(10**logmd[u:1000], np.abs(FWc2s(logmd[u:1000], c2s, beta2s) / FW(logmd[u:1000], al, 2.5, mWDM) - 1), color=coolwarm(1.0), linewidth='2', zorder=3)
line9, = ax1.plot(10**logmd[u:1000], (c * (4 * np.pi * rho0 * omm / (3 * 10**logmd[u:1000] / hreal))**(1/3))**(21-1-betas)*0.64 / 10**26, color=coolwarm(0.25), linewidth='2', linestyle='--', label='D$_{SK}$/D$_{KS}$', zorder=3)
line10, = ax1.plot(10**logmd[u:1000], 0.00293*np.ones(logmd[u:1000].size)*(3/4+1/(4*(1-0.404)**(20))), color=coolwarm(0.7), linestyle='--', linewidth='2', label='D$_{DB}$/D$_{KS}$', zorder=3)

# === Custom Tick Formatting for Logarithmic Scale ===
# Specify the ticks you want to show
y_ticks = [1e0, 1e4, 1e8]  # Define the desired ticks
ax1.set_yticks(y_ticks)      # Set the ticks explicitly

# Optionally format the tick labels
ax1.set_yticklabels([f'$10^{{{int(np.log10(y))}}}$' for y in y_ticks])

# Remove minor ticks (optional)
ax1.yaxis.set_minor_locator(plt.NullLocator())
ax1.xaxis.set_minor_locator(plt.NullLocator())

# Customize the appearance of the ticks (optional)
ax1.tick_params(axis='x', which='both', direction='in', length=6, width=1, labelsize=14)
ax1.tick_params(axis='y', which='both', direction='in', length=6, width=1, labelsize=14)

ax1.set_xlabel('$M$ [$h^{-1} M_\\odot$]', fontsize=14)
ax1.set_ylabel('$\\Delta_i^{KS}$', fontsize=14)

# Add legend
ax1.legend(fontsize=13, loc='upper right')
# === Save and Show ===
plt.savefig('FHMF_KS.eps', format='eps', bbox_inches='tight', pad_inches=0.1)
plt.show()
#%%
import matplotlib.pyplot as plt
import numpy as np

# Sample data for demonstration
x = np.logspace(0.1, 2, 100)  # X values
y = x ** 2                    # Y values (quadratic relationship)

# Create the plot
fig, ax = plt.subplots()
ax.plot(x, y)

# Set logarithmic scale for the Y-axis
ax.set_yscale('log')

# Specify the ticks you want to show
y_ticks = [1e1, 1e3, 1e5]  # Define the desired ticks
ax.set_yticks(y_ticks)      # Set the ticks explicitly

# Optionally format the tick labels
ax.set_yticklabels([f'$10^{{{int(np.log10(y))}}}$' for y in y_ticks])

# Remove minor ticks (optional)
ax.yaxis.set_minor_locator(plt.NullLocator())

# Customize the appearance of the ticks (optional)
ax.tick_params(axis='y', which='both', direction='in', length=6, labelsize=12)

# Add labels and title
ax.set_xlabel('X-axis (Log scale)', fontsize=14)
ax.set_ylabel('Y-axis (Log scale)', fontsize=14)
ax.set_title('Custom Logarithmic Ticks', fontsize=16)

# Show the plot
plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Get the 'cividis' colormap
cividis = cm.get_cmap('viridis')

# Create an array of numbers to represent the range of the colormap
x = np.linspace(0, 1, 256)  # 256 values to cover the range from 0 to 1

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 2))

# Create a horizontal gradient using the 'cividis' colormap
ax.imshow([x], aspect='auto', cmap=cividis)
ax.set_axis_off()  # Turn off the axis for a cleaner display

# Add a colorbar
plt.colorbar(ax.imshow([x], aspect='auto', cmap=cividis), ax=ax, orientation='horizontal')

# Show the plot
plt.show()

#%%
plt.plot(10**logmd[u:1000],(c*(4*np.pi*rho0*omm/(3*10**logmd[u:1000]/hreal))**(1/3))**(13.65)/10**26)
#plt.plot(10**logmd[u:1000],4*10**56*(10**logmd[u:1000])**(-8.5))
plt.plot(10**logmd[u:1000], lerr_s_W[u:1000], color='green')
plt.plot(10**logmd[u:1000], np.abs(FWs(logmd[u:1000], cs, betas)/FW(logmd[u:1000],al,c,mWDM)-1))
plt.xscale('log')
plt.yscale('log')

#%%
print(c*(4*np.pi*rho0*omm/(5*1e4/hreal))**(1/3))

#%%
#segona figura de l'article
lk=np.logspace(-3,6,2000)
# Your data and plotting code
plt.figure(figsize=(7, 6))  # Adjust width and height here

# Get the 'coolwarm' colormap
coolwarm = cm.get_cmap('coolwarm')

plt.plot(lk, np.abs(dersig2s(lk,h, betas)-dersig2W(lk, mWDM, al))/(dersig2W(lk, mWDM, al)), color=coolwarm(0.0))
plt.plot(lk, np.abs(dersig2c2s(lk,h, beta2s)-dersig2W(lk, mWDM, al)) /(dersig2W(lk, mWDM, al)), color=coolwarm(1.0))
#plt.plot(np.ones(1000)*103, np.logspace(-2,16,1000), color='black', linestyle='-', linewidth='0.5')
# Shade between x = 0 and x =103 
plt.axvspan(47, 500, color=(0.9, 0.9, 0.9, 0.6), label="Shaded Region") # alpha=0.2, hatch="\\", edgecolor="black",
# extrems teòrics

plt.plot(lk,np.ones(lk.size)*(3/4+1/(4*(1-0.403)**(20))),color=coolwarm(0.7),linestyle='--')
plt.plot(lk,lk**(20-betas)*4/10**(26),color=coolwarm(0.25),linestyle='--')
#plt.plot(lk, np.zeros(lk.size), color='royalblue')
#plt.plot(lk, np.abs(dersig2varconst(lk, -0.5)-dersig2nWDM(lk))/(dersig2nWDM(lk)), color='orange')
#plt.plot(1/lk,dersig2s(lk, 6.35)/dersig2nWDM(lk))


plt.xticks(fontsize = '17')
plt.yticks(fontsize = '17')
plt.tick_params(axis='x',which='major', direction='in', length=7, width=1.7)
plt.tick_params(axis='x', which='minor', direction='in', length=4, width=1)
plt.tick_params(axis='y',which='major', direction='in', length=7, width=1.7)
plt.tick_params(axis='y', which='minor', direction='in', length=4, width=1)

plt.xlim([15, 500])
plt.ylim([1e-4,2e12])
plt.xlabel('k ($1/Mpc$)', fontsize = '17')
plt.ylabel('$\epsilon_i$', fontsize = '17')
plt.legend(('Smooth-k','Double Box'), loc='upper left',fontsize = '15')
plt.xscale('log')
plt.yscale('log')
# Explicitly set ticks in log scale

ax = plt.gca()  # Get current axes
ax.set_xscale('log')
ax.set_yscale('log')

# Define custom ticks for log scale
x_ticks = [100]
y_ticks = [1e-4, 1e0 ,1e4, 1e8, 1e12]
ax.yaxis.set_major_formatter(plt.NullFormatter())  # Remove labels for major ticks
#ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[], numticks=1))  # Major ticks only at 100
ax.set_xticks(x_ticks)

ax.set_yticks(y_ticks)
# Remove labels for any ticks or format them as desired
#ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '' if x not in  else str(x)))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '' if y not in [1e-4, 1e0,1e4,  1e8, 1e12] else f'$10^{{{int(np.log10(y))}}}$'))



ax.xaxis.set_minor_formatter(plt.NullFormatter())

# Save as EPS
plt.savefig('FVD.eps', format='eps', bbox_inches='tight', pad_inches=0.1)

plt.show()
#%%
#quarta figura de l'article
#the following two cells are a combined figure for the deviation and the values of HMF
logmd=np.linspace(7,12,1000)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

# Import necessary libraries
from matplotlib import cm

# Get the 'coolwarm' colormap for consistent styling
coolwarm = cm.get_cmap('coolwarm')
magma = cm.get_cmap('YlGnBu')

fig = plt.figure(figsize=(5.8, 16))
# Set height ratios for subplots
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

# === First Subplot ===
ax0 = plt.subplot(gs[0])

# Set axis limits and ticks
ax0.set_xlim([2.05 * 10**7, 10**12])
ax0.set_ylim([1.1 * 10**(-3), 2.5])

# Add gridlines for improved readability
ax0.grid(visible=True, which='major', linestyle='-', linewidth=0.5, color=(0.75, 0.75, 0.75, 0.6), zorder=2)

# Logarithmic x-scale
ax0.set_xscale("log")
ax0.set_yscale('log')

# Plot lines with updated colors
line0, = ax0.plot(10**logmd2[1:100], 10**logHMFd2[1:100], '-', color='orange', linewidth='2', zorder=3)
line1, = ax0.plot(10**logmd, FWs(logmd, cs, betas), color=coolwarm(0.0), linewidth='2', label='Smooth-k', zorder=3)
line2, = ax0.plot(10**logmd, FWc2s(logmd, c2s, beta2s), color=coolwarm(1.0), linewidth='2', label='Double Box', zorder=3)
line3, = ax0.plot(10**logmd, FW(logmd, al, c, mWDM), color=magma(0.4), linewidth='2', label='K-sharp', zorder=3)
line4, = ax0.plot(10**logm33keV,10**logHMF33keV,'x', color='orange', linewidth='2', markersize=8, markeredgewidth=2, label='COCO', zorder=3)

# === Custom Tick Formatting for Logarithmic Scale ===
# Specify the ticks you want to show
y_ticks = [1e-2, 1e-1, 1]  
ax0.set_yticks(y_ticks)      # Set the ticks explicitly

# Optionally format the tick labels
ax0.set_yticklabels([f'$10^{{{int(np.log10(y))}}}$' for y in y_ticks])

# Remove minor ticks
ax0.yaxis.set_minor_locator(plt.NullLocator())
ax0.xaxis.set_minor_locator(plt.NullLocator())

# Major tick settings
ax0.tick_params(axis='both', labelsize=14, direction='in', which='major', length=6, width=1)

ax0.set_ylabel('d$n$ $/$ d$\\log M$ [$h^3 \\mathrm{Mpc}^{-3}$]', fontsize=14)

# Hide x-axis labels for this subplot
plt.setp(ax0.get_xticklabels(), visible=False)

# Add legend
ax0.legend(fontsize=13, loc='lower left')

# === Adjust spacing between subplots ===
plt.subplots_adjust(hspace=0.1)


# === Second Subplot ===
ax1 = plt.subplot(gs[1], sharex=ax0)

# Set axis limits and ticks
ax1.set_xlim([2.05 * 10**7, 10**12])
ax1.set_ylim([0, 0.65])

# Add gridlines for improved readability
ax1.grid(visible=True, which='major', linestyle='-', linewidth=0.5, color=(0.75, 0.75, 0.75, 0.6), zorder=2)

# Logarithmic x-scale
ax0.set_xscale("log")

# Plot lines with updated colors
u = 0
line4, = ax1.plot(10**logmd[u:1000], lerr_s, color=coolwarm(0.0), linewidth='2', zorder=3)
line5, = ax1.plot(10**logmd[u:1000], lerr_c2s, color=coolwarm(1.0), linewidth='2', zorder=3)
#line6, = ax1.plot(10**logmd[u:1000], np.zeros(logmd[u:1000].size), color='orange', linewidth='2', linestyle='--', zorder=3)
line7, = ax1.plot(10**logmd[u:1000], lerr_W, color=magma(0.4), linewidth='2', zorder=3)

# Specify the ticks you want to show
y_ticks = [0.0, 0.2, 0.4, 0.6]  
ax1.set_yticks(y_ticks)      # Set the ticks explicitly

# Optionally format the tick labels
#ax0.set_yticklabels([f'$10^{{{int(np.log10(y))}}}$' for y in y_ticks])

# Remove minor ticks
ax1.xaxis.set_minor_locator(plt.NullLocator())

# Major tick settings
ax1.tick_params(axis='both', labelsize=14, direction='in', which='major', length=6, width=1)

ax1.set_xlabel('$M$ [$h^{-1} M_\\odot$]', fontsize=14)
ax1.set_ylabel('$\\Delta_i^{COCO}$', fontsize=14)

# === Save and Show ===
plt.savefig('FHMF.eps', format='eps', bbox_inches='tight', pad_inches=0.1)
plt.show()
#%%
from scipy.stats import skewnorm
plt.hist(llerr_W,
         bins=7,         # number of bins, you can also pass an array of bin edges
         density=True,   # if True, normalize to form a probability density
         alpha=0.75,      # transparency
         edgecolor='black')
# Parameters for skew-normal distribution
a = 5      # positive skew parameter (a > 0 for right skew, a < 0 for left skew)
loc = -0.03    # mean of the distribution
scale = 0.1  # standard deviation

# Define x-range
x = np.linspace(-0.1, 0.15, 1000)

# Compute the skew-normal PDF
pdf_skewed = skewnorm.pdf(x, a, loc=loc, scale=scale)

plt.plot(x, pdf_skewed, label=f'Skew-normal (a={a})')
plt.title("Skewed Gaussian (Skew-normal) Distribution")
plt.xlabel("x")
plt.ylabel("PDF")
plt.legend()
plt.grid(True)
plt.show()

#%%
from scipy.stats import skewnorm
from scipy.optimize import curve_fit
nbins=5
data=llerr_W
# Compute histogram (normalized to form a PDF)
counts, bin_edges = np.histogram(data,bins=nbins,density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Define skew-normal PDF for curve fitting
def skewnorm_pdf(x, a, loc, scale):
    return skewnorm.pdf(x, a, loc=loc, scale=scale)

# Initial parameter guesses: shape (a), loc, scale
initial_guess = [3, -0.03, 0.1]

# Fit the skew-normal PDF to the histogram data
params_opt, params_cov = curve_fit(skewnorm_pdf, bin_centers, counts, p0=initial_guess)

# Extract fitted parameters
a_fit, loc_fit, scale_fit = params_opt

# Plot histogram and fitted PDF
x_fit = np.linspace(bin_edges[0], bin_edges[-1], 200)
pdf_fit = skewnorm.pdf(x_fit, a_fit, loc=loc_fit, scale=scale_fit)

print(params_opt)
plt.figure()
plt.hist(data,bins=nbins,density=True, label='Histogram data')
plt.plot(x_fit, pdf_fit, linewidth=2, label=f'Fitted skewnorm\n(a={a_fit:.2f}, loc={loc_fit:.2f}, scale={scale_fit:.2f})')
plt.title("Histogram and Fitted Skew-normal Distribution")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()

#%%
from scipy.stats import skewnorm, kstest, probplot
params = params_opt
a, loc, scale = params

# Kolmogorov-Smirnov Test
statistic, p_value = kstest(data, "skewnorm", args=params)
print(f"Test Statistic: {statistic:.4f}, p-value: {p_value:.4e}")

# Histogram and Fitted Curve
plt.hist(data, bins=nbins, density=True, alpha=0.6, color="skyblue", edgecolor="black", label="Histogram")
x = np.linspace(min(data), max(data), 1000)
pdf = skewnorm.pdf(x, a, loc, scale)
plt.plot(x, pdf, "r-", label="Fitted Skew-Normal PDF")
plt.legend()
plt.title(f"Fit with p-value={p_value:.4e}")
plt.show()

#%%
print(skew_normal_loglik(params,data))
#%%
from scipy.stats import skewnorm
from scipy.optimize import minimize

# 1. Define the log-likelihood function
def skew_normal_loglik(params, data):
    """
    Compute the log-likelihood of the skew-normal distribution
    for data given parameters params = [xi, omega, alpha].
    
    xi    : location (mean) parameter
    omega : scale (std. dev.) parameter, must be > 0
    alpha : shape (skewness) parameter
    """
    alpha, xi, omega = params
    if omega <= 0:
        return -np.inf   # invalid scale

    # scipy.stats.skewnorm takes arguments (a, loc, scale)
    # and has a .logpdf method for log-density
    ll = skewnorm.logpdf(data, a=alpha, loc=xi, scale=omega)
    return np.sum(ll)

# 3. Compute log-likelihood at some parameter guess
initial_guess = [a_fit, loc_fit, scale_fit]
print("Log-likelihood @ initial guess:",
      skew_normal_loglik(initial_guess, data))

# 4. Find the MLE by maximizing the log-likelihood
#    (we minimize the negative log-likelihood)
def neg_loglik(params):
    return -skew_normal_loglik(params, data)

bounds = [(-np.inf, np.inf),     # xi ∈ ℝ
          (1e-6, np.inf),        # omega > 0
          (-np.inf, np.inf)]     # alpha ∈ ℝ

res = minimize(neg_loglik, initial_guess, bounds=bounds)

if res.success:
    xi_mle, omega_mle, alpha_mle = res.x
    print(f"MLE estimates:\n xi = {xi_mle:.4f}\n omega = {omega_mle:.4f}\n alpha = {alpha_mle:.4f}")
    print("Maximized log-likelihood:", -res.fun)
else:
    print("Optimization failed:", res.message)
#%%
#in this cell the deviation from the interpolation of simulations is calculated for
#the five used filters

#using the function defined above to obtain a big set of interpolated poits from simulations
logmd=np.linspace(7.4,12,12)
#logmd=logHMF33keV
logHMFd=logHMF33COCO(logmd)

llerr_s=np.zeros(logmd.size)
#lerr_c2=np.zeros(logmd.size)
#lerr_c4=np.zeros(logmd.size)
llerr_W=np.zeros(logmd.size)
llerr_c2s=np.zeros(logmd.size)


#generatinng the HMF arrays for each filter
FWsl=FWs(logmd, cs, betas)
#FWc2l=FWc2(logmd, a1, a2, cc2)
#FWc4l=FWc4(logmd, b1,b2,b3,b4,cc4)
FWl=FW(logmd,al,2.6744,mWDM)
FWc2sl=FWc2s(logmd, c2s, beta2s)

#sum of each point deviation
for i in range(1,logmd.size,1):  
    llerr_s[i]=((10**logHMFd[i]-FWsl[i])/(10**logHMFd[i]))
    #lerr_c2[i]=((10**logHMFd[i]-FWc2l[i])/(10**logHMFd[i]))**2
    #lerr_c4[i]=((10**logHMFd[i]-FWc4l[i])/(10**logHMFd[i]))**2
    llerr_W[i]=((10**logHMFd[i]-FWl[i])/(10**logHMFd[i]))
    llerr_c2s[i]=((10**logHMFd[i]-FWc2sl[i])/(10**logHMFd[i]))

    
#%%                            
#this cell is just for the k-sharp filtered HMF as a difference from the previous cell
#quarta figura de l'article
#the following two cells are a combined figure for the deviation and the values of HMF
logmd=np.linspace(7,12,1000)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Get the 'coolwarm' colormap for consistent styling
magma = cm.get_cmap('YlGnBu')

# Set the figure size
fig = plt.figure(figsize=(7, 6))

# Create a single axis (no subplots)
ax0 = fig.add_subplot(111)

# Set labels
ax0.set_ylabel('$dn/d \\log M$ [$h^3 \\mathrm{Mpc}^{-3}$]', fontsize=17)
ax0.set_xlabel('$M$ [$h^{-1} M_\\odot$]', fontsize=17)

# Add gridlines for improved readability
ax0.grid(visible=True, which='major', linestyle='-', linewidth=0.5, color=(0.75, 0.75, 0.75, 0.6), zorder=2)

# Logarithmic x-scale
ax0.set_xscale("log")
#ax0.set_yscale("log")  # Uncomment this if you need to set a log scale for the y-axis

# Plot the FW function with the colormap
line1, = ax0.plot(10**logmd2[0::1], 10**logHMFd2[0::1], 'x', color='orange', linewidth=2, label='COCO', markersize=7, markeredgewidth=2, zorder=3)
line2, = ax0.plot(10**logmd, FW(logmd, al, c, mWDM), color=magma(0.45), linewidth=2, label='K-sharp', zorder=3)

# Set axis limits and ticks
ax0.set_xlim([2.05 * 10**7, 10**12])
ax0.set_yticks([0.0, 0.5, 1.0, 1.5])

# Customize tick parameters: font size, ticks pointing inwards, and padding for labels
ax0.tick_params(axis='both', labelsize=17, direction='in', which='both')  # Added label padding

# Add legend with font size of 15
ax0.legend(fontsize=15, loc='upper right')

# Save the plot
plt.savefig('FW_plot.eps', format='eps', bbox_inches='tight', pad_inches=0.1)

# Show the plot
plt.show()


#%%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
#primera figura de l'article
#the three studied filters with different parameters in addition to the top-hat filterç
# Get the 'coolwarm' colormap for consistent styling
coolwarm = cm.get_cmap('coolwarm')
yl = cm.get_cmap('YlGnBu')
magma = cm.get_cmap('magma')

linkR=np.logspace(-2,2,5000)
linkRfall=np.linspace(2**(1/beta2s),100,10000)
linkRbox=np.linspace(0,2**(1/beta2s)-1e-6,10000)
zeros=np.zeros(10000)

linkRtotal=np.concatenate((zeros,1/linkRfall))
linkRtotal2=np.concatenate((1/linkRbox,1/linkRfall))

fig = plt.figure(figsize=(7, 5.8))
plt.plot(np.log10(linkR*cs),(1+(linkR)**betas)**(-1),color=coolwarm(0.0))
plt.plot(np.log10(linkR*c2s),0.75*(1+(linkR)**1e5)**(-1)+0.25*(1+(linkR/(1-0.403))**1e5)**(-1),color=coolwarm(1.0))
#plt.plot(np.log10(1/linkRtotal2*c2s),(1+(1/linkRtotal2/2**(1/beta2s))**1e10)**(-1)+0.5*(linkRtotal)**beta2s)
#plt.plot(np.log10(1/linkRtotal*c2s),(linkRtotal)**beta2s)
#plt.plot(np.log10(linkR*1),(0.25*(1+(linkR*(1-0.52))**1e5)**(-1)+0.75*(1+(linkR*(1-0.049))**1e5)**(-1)))
plt.plot(np.log10(linkR*0.75),(3*(np.sin(linkR)-linkR*np.cos(linkR)))/linkR**3, color=magma(0.8))
plt.plot(np.log10(linkR*0.9),(1+(linkR)**1e5)**(-1), color=yl(0.45))

plt.xlabel('$log(kcR)$',fontsize = '16')
plt.ylabel('$W(kR)$',fontsize = '16')
plt.xticks([-0.5,0.0,0.5,1.0,1.5],fontsize = '16')
plt.yticks([0.0,0.5,1.0],fontsize = '16')

# Adjust the position of x-ticks inside the plot
plt.tick_params(axis='both', direction='in', length=6, width=1)  # axis='x' for x-axis, 'in' for ticks inside


plt.legend(('SK ($\hat{\\beta} =6.32$)','DB ($\\beta=-0.404$)', 'TH', 'KS'), fontsize='15')
plt.xlim([-0.5,1.62])

# Save as EPS
plt.savefig('filtres.eps', format='eps', bbox_inches='tight', pad_inches=0.1)

plt.show()

#%%
#obtaining the rough values of the spectral indexs for the used CDM and WDM power spectrum
link4=np.logspace(-8,8,1000)

#transfer matrix from Viel et.al (2005)
aaalpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
aabeta=2*1.12
aagamma=-5/1.12

#plt.plot(link4,PSl(link4,al)) #CDM power spectrum
plt.plot(link4,link4**2*PSl(link4,al)*(1+(aaalpha*link4)**aabeta)**(2*aagamma)) #WDM power spectrum
#plt.plot(link4,PSl(link4,3e6)*(1+(aaalpha*link4)**aabeta)**(2*aagamma)) #WDM power spectrum
#plt.plot(link4,PSl(link4,al)*(1+(aaalpha*link4)**aabeta)**(2*aagamma)*(1+(0.02*link4)**4)**(-5))  #an additional function is added to obtain a more negative slope

#rough fits for the different regimes
plt.plot(link4,link4**(-0.5)*4.5e3) 
plt.plot(link4,link4**(2.97)*2.5e8)
plt.plot(link4,link4**(-24+0.966)*8e43)
#plt.plot(link4,link4**(-12)*8e16)

plt.ylim([10**(-17),10**(7)])

plt.xlabel('k ($h/Mpc$)')
plt.ylabel('P(k) ($Mpc^3$)')
plt.xscale('log')
plt.yscale('log')

   
#%%
#trends of the variance behaviour function for slopes of small masses (spectral power for R \to 0)
linian=np.linspace(-23,-3,10000)
betass=3.67
betasmoothk=3.2
aj=0.25
bb=0.4
ns=-0.067#spectral index variation
plt.plot(linian,(251*(linian+23.01)**(-2/3))**ns*(linian+3)/(linian+ns+3))
plt.plot(linian,(12.59*(-(linian+2.5))**(1/1.65))**ns*(linian+3)/(linian+ns+3))
plt.plot(linian,2**((linian+3)/betass)*(1-(linian+3)/(2*(linian+3-2*betass))))
plt.plot(linian,1-(linian+3)/(linian+3-2*betasmoothk)-(2*(linian+3))/(linian+3+betasmoothk)+((linian+3))/(linian+3+2*betasmoothk))
#plt.plot(linian,3/4+2**((linian+3)/betass)*(1/4)-2**((linian+3)/betass-2*betass)*(linian+3)/(2*(linian+3-2*betass)))
plt.plot(linian,(1-aj)+(aj)*(1+bb)**(linian+3))

plt.xlabel('$n (spectral index)$')
plt.ylabel('$p(n)$')

plt.ylim([-1,1.5])

#%%
#the relative deviation of the p functions for the SK and sDB filters
#consiering a sDB filter
#plt.plot(linian,np.abs(((251*(linian+23.1)**(-2/3))**ns*(linian+3)/(linian-ns+3)-2**((linian+3)/betass)*(1-(linian+3)/(linian+3-2*betass)/2))/((251*(linian+23.1)**(-2/3))**ns*(linian+3)/(linian-ns+3)))) #for the  interval [-22,-12]
#plt.plot(linian,np.abs(((12.59*(-(linian+2.5))**(1/1.65))**ns*(linian+3)/(linian-ns+3)-2**((linian+3)/betass)*(1-(linian+3)/(linian+3-2*betass)/2))/((12.59*(-(linian+2.5))**(1/1.65))**ns*(linian+3)/(linian-ns+3))))

#considering an SK filter
#plt.plot(linian,(((251*(linian+23.1)**(-2/3))**ns*(linian+3)/(linian-ns+3)-1+(linian+3)/(linian+3-2*betasmoothk)+(2*(linian+3))/(linian+3+betasmoothk)-((linian+3))/(linian+3+2*betasmoothk))/((251*(linian+23.1)**(-2/3))**ns*(linian+3)/(linian-ns+3)))**2)

#considering a DB non uniform
plt.plot(linian,(((12.59*(-(linian+2.5))**(1/1.65))**ns*(linian+3)/(linian-ns+3)-(1-aj)-(aj)*(1+bb)**(linian+3))/((12.59*(-(linian+2.5))**(1/1.65))**ns*(linian+3)/(linian-ns+3)))**2)
#%%
#power spectrum logarithm derivative
loglink5=np.linspace(-4,5,10000)
#derivlogP=np.zeros(loglink5.size)
h=0.0000001

#transfer matrix from Viel et.al (2005)
aaalpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
aabeta=2*1.12
aagamma=-5/1.12

derivlogP=(np.log10(PSl(10**(loglink5+h),al)*(1+(aaalpha*(10**(loglink5+h)))**aabeta)**(2*aagamma))-np.log10(PSl(10**(loglink5-h),al)*(1+(aaalpha*(10**(loglink5-h)))**aabeta)**(2*aagamma)))/(2*h)
#derivlogP=(np.log10(PSl(10**(loglink5+h),al))-np.log10(PSl(10**(loglink5-h),al)))/(2*h)

plt.plot(10**loglink5,derivlogP)  

plt.xlim([10**(-4),10**(5)])
plt.ylim([-24,2])
#plt.plot(10**loglink5,(1/(10**(loglink5-2.4)))**1.5-22.9)
plt.plot(10**loglink5,(1/(10**(loglink5-2.4)))**1.5-22.9)
plt.plot(10**loglink5,-24*(loglink5-2.1)-23)
plt.plot(10**loglink5,1-(10**(loglink5-1.1))**1.65-3.5)

plt.xlabel('k (1/Mpc)')
plt.ylabel('n')
plt.xscale('log')


#%%
def dlogP(logk):
    derlogP=(np.log10(PSlin(10**(logk+h),al)*(1+(aaalpha*(10**(logk+h)))**aabeta)**(2*aagamma))-np.log10(PSlin(10**(logk-h),al)*(1+(aaalpha*(10**(logk-h)))**aabeta)**(2*aagamma)))/(2*h)
    return derlogP



l1=1e-6
l2=
mitjana=(integrate.quad(lambda x: dlogP(np.log10(x)),l1,l2)[0])/(l2-l1)

print(dlogP(1),mitjana)


#%%
#analisi de l'integrant per a smooth k
lk=np.logspace(-3,8,10000)
integrand=np.zeros(10000)
integrand1=np.zeros(10000)
integrand2=np.zeros(10000)
integrand3=np.zeros(10000)
kk=1e8
beta=25
alpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
bbeta=2*1.12
gamma=-5/1.12
b=1

for i in range(0,10000):   
    #integrand[i]=2*beta*(lk[i]/kk)**(beta)*smooth(lk[i],kk,beta)**3*PSlin(lk[i],al)*lk[i]**2*(1+(alpha*lk[i])**bbeta)**(2*gamma)/(2*np.pi**2*kk)
    integrand1[i]=(1-(lk[i]/kk)**beta)*(lk[i]**beta/kk**(beta+1))*PSlin(lk[i],al)*lk[i]**2*(1+(alpha*lk[i])**bbeta)**(2*gamma)/(2*np.pi**2*kk)
    #integrand2[i]=np.exp(-(lk[i]/kk)**b/2)*lk[i]**b/kk**(b+1)*PSlin(lk[i],al)*lk[i]**2*(1+(alpha*lk[i])**bbeta)**(2*gamma)/(2*np.pi**2)
    integrand3[i]=(lk[i]/kk**2)/np.cosh(lk[i]/(kk*beta))*PSlin(lk[i],al)*lk[i]**2*(1+(alpha*lk[i])**bbeta)**(2*gamma)/(2*np.pi**2)
plt.plot(lk,integrand)   
plt.plot(lk,integrand1)
plt.plot(lk,integrand3)  

#plt.xscale('log')
plt.yscale('log') 
plt.xlim(0,10*kk)
#%%
km=10
b=10
plt.plot(lk,np.exp(-(lk/km)**b/2)+((lk+10*km)/km)**(-2))
plt.plot(lk,lk**(-2))
plt.xlim(10**(-4),10000*km)

plt.xscale('log')
plt.yscale('log') 


#%%
rl=np.linspace(1e-4,1e2)
plt.plot(rl,1/(2*(1-np.cos(rl)))**(3/2))
plt.plot(rl,1/np.sin(rl)**3)
plt.plot(rl,1/rl**3)

plt.xscale('log')
plt.yscale('log') 


#%%
plt.plot(lk,(1+(lk/kk)**beta)**(-1))
plt.plot(lk,1-(lk/kk)**beta)

plt.ylim(0,1.1)
plt.xscale('log')

