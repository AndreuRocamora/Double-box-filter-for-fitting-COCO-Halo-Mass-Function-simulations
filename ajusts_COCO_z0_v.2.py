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
#deltac=(rhoc-rho0)/rho0 #crytical density contrast
deltac=1.6865 #de moment considero aquest valor de deltac
omWDM=omm-omb
mWDM=3.3
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
plt.plot(np.log10(link3),np.log10(Delta(link3*hreal,1.3e7)))
plt.plot(np.log10(link3),np.log10(DeltaW(link3*hreal,1.3e7)))

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
    deriv=np.zeros(k.size)
    omWDM=omm-omb
    #matriu de transefrència Matteo Leo thermal WDM
    alpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
    beta=2*1.12
    gamma=-5/1.12
    
    for l in range(1,k.size,1):
        deriv[l]=PSlin(k[l], al)*(1+(alpha*k[l])**beta)**(2*gamma)*k[l]**2/(2*np.pi**2)
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
logm33keV=np.array([7.012,7.237,7.392,7.771,8.151,8.53,8.924,9.317,9.697,10.076,10.456,10.849,11.229,11.608,11.995])
logHMF33keV=np.array([-0.933,-0.397,-0.143,0.045,0.004,-0.089,-0.263,-0.518,-0.799,-1.121,-1.429,-1.75,-2.112,-2.46,-2.795])


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
    
    
    sigmaquadrat=integrate.quad(lambda x: smooth(x,y,beta)**2*PSlin(x,al)*x**2*(1+(alpha*x)**bbeta)**(2*gamma)/(2*np.pi**2),1e-6,1000)[0]
    
    #to correct an integration error
    if sigmaquadrat<1e-5:
        sigmaquadrat=sigma2linW(y,mWDM, al)
        
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
        
    sigmaquadrat=0.75*integrate.quad(lambda x: PSlin(x,al)*x**2*(1+(alpha*x)**bbeta)**(2*gamma)/(2*np.pi**2),1e-6,y)[0]+0.25*integrate.quad(lambda x: PSlin(x,al)*x**2*(1+(alpha*x)**bbeta)**(2*gamma)/(2*np.pi**2),1e-6,y*(1+beta))[0]#+0.5*integrate.quad(lambda x: (y/x)**(2*beta)*PSlin(x,al)*x**2*(1+(alpha*x)**bbeta)**(2*gamma)/(2*np.pi**2),y*2**(1/beta),np.infty)[0]
    return sigmaquadrat   

#%%
def sigma2saprox(y,beta):
    #WDM transefr matrix from Viel et. al (2005)
    alpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
    bbeta=2*1.12
    gamma=-5/1.12
    
    
    sigmaquadrat=integrate.quad(lambda x: (1-(x/y)**beta)**(2)*PSlin(x,al)*x**2*(1+(alpha*x)**bbeta)**(2*gamma)/(2*np.pi**2),1e-6,y)[0]+integrate.quad(lambda x: (y/x)**(2*beta)*PSlin(x,al)*x**2*(1+(alpha*x)**bbeta)**(2*gamma)/(2*np.pi**2),y,1000)[0]
    
    #to correct an integration error
    if sigmaquadrat<1e-5:
        sigmaquadrat=sigma2linW(y,mWDM, al)
        
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
lsig2WDMc2=np.zeros(lk.size)
lsig2WDMc2s=np.zeros(lk.size)

for i in range(0,lk.size):
    lsig2CDM[i]=sigma2linCDM(lk[i],al)
    lsig2WDM[i]=sigma2linW(lk[i],mWDM,al)#*(1-(-15+6)/(-15+3-2*6)/2)*2**((-15+3)/6)
    lsig2WDMs[i]=sigma2s(lk[i],2)
    lsig2WDMc2[i]=sigma2caixes2(lk[i],c,-0.19,0.48)
    lsig2WDMc2s[i]=sigma2caixes2s(lk[i],0)
    lsig2WDMsaprox[i]=sigma2saprox(lk[i],2)

plt.plot(np.log10(lm),np.log10(lsig2CDM))
plt.plot(np.log10(lm),np.log10(lsig2WDM))
plt.plot(np.log10(lm),np.log10(lsig2WDMs))
#plt.plot(np.log10(lm),np.log10(lsig2WDMsaprox))
#plt.plot(np.log10(lm),np.log10(lsig2WDMc2))
plt.plot(np.log10(lm),np.log10(lsig2WDMc2s))

plt.xlabel('M ($M_o$)')
plt.ylabel('$\sigma ^2$')

plt.xlim([6,15])
plt.ylim([-1,4])

#%%
#variance derivative for the studied filters apart of k-sharp
liniay=np.logspace(-4,2,num=1000)
h=0.0001

def dersig2s(k,h,beta):
    deriv=np.zeros(k.size)
    for l in range(1,k.size,1):
        deriv[l]=(sigma2s(k[l]+h,beta)-sigma2s(k[l]-h,beta))/(2*h)
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
    
    #thermal WDM transference matrix
    alpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
    beta=2*1.12
    gamma=-5/1.12
        
    for l in range(0,k.size,1):
        for i in range(0,params.size,1):
            deriv[l]=deriv[l]+PSlin(k[l]*(1+params[i]), al)*(1+(alpha*k[l]*(1+params[i]))**beta)**(2*gamma)*(k[l]*(1+params[i]))**2/(2*params.size*np.pi**2) #+PSlin(k[l]*(1+a2), al, m)*(1+(alpha*(k[l]*(1+a2)))**beta)**(2*gamma)*(k[l]*(1+a2))**2/(4*np.pi**2)
    return deriv

def dersig2c2s(k,h,beta):
    deriv=np.zeros(k.size)
    for l in range(1,k.size,1):
        deriv[l]=(sigma2caixes2s(k[l]+h,beta)-sigma2caixes2s(k[l]-h,beta))/(2*h)
    return deriv
#%%
#variance derivative plot for CDM and WDM k-sharp and smooth-k and Double Box
plt.plot(liniay,dersig2CDM(liniay,al))
plt.plot(liniay,dersig2W(liniay,mWDM,al))
plt.plot(liniay,dersig2s(liniay,h,2))
plt.plot(liniay,dersig2c2(liniay,c,0,0))
plt.plot(liniay,dersig2c2s(liniay,h,1000))

plt.xlabel('k ($h/Mpc$)')
plt.ylabel('d$\sigma ^2$/d$k$')

plt.xlim([0,40])
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

plt.plot(logm33keV,logHMF33keV,'o')

logarm=np.linspace(6,13,1000)
plt.plot(logarm,np.log10(FWs(logarm, popt[0], popt[1])))

#quan es fa l'ajust per als paràmetres de localització, cal guardar-los per tal que no s'acumulin en els ajustos posteriors 
cs=popt[0]
betas=popt[1]

plt.xlabel('M ($M_o/h$)')
plt.ylabel('$dn/d log M$ [$h^3Mpc^{-3}$]')

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

popt, pcovariance = curve_fit(FWc2s, x_data, y_data, p0=[3,0])

plt.plot(logm33keV,logHMF33keV,'o')

logarm=np.linspace(6,13,1000)
plt.plot(logarm,np.log10(FWc2s(logarm, popt[0], popt[1])))

#when the fit is done the parameters c and beta are saved in
c2s=popt[0]
beta2s=popt[1]

plt.xlabel('M ($M_o/h$)')
plt.ylabel('$dn/d log M$ [$h^3Mpc^{-3}$]')
#%%
#in this cell the deviation from the interpolation of simulations is calculated for
#the five used filters
err_s=0 
err_c2=0 
err_c4=0
err_c2s=0
err_W=0

#using the function defined above to obtain a big set of interpolated poits from simulations
logmd=np.linspace(7.3,12,1000)
logHMFd=logHMF33COCO(logmd)

#initialyzing the arrays for the deviation
lerr_s=np.zeros(logmd.size)
lerr_c2=np.zeros(logmd.size)
lerr_c4=np.zeros(logmd.size)
lerr_W=np.zeros(logmd.size)
lerr_c2s=np.zeros(logmd.size)

#accumulated deviation
lacum_W=np.zeros(logmd.size)
lacum_s=np.zeros(logmd.size)
lacum_c2s=np.zeros(logmd.size)

#generatinng the HMF arrays for each filter
FWsl=FWs(logmd, cs, betas)
FWc2l=FWc2(logmd, a1, a2, cc2)
FWc4l=FWc4(logmd, b1,b2,b3,b4,cc4)
FWl=FW(logmd,al,c,mWDM)
FWc2sl=FWc2s(logmd, c2s, beta2s)

#sum of each point deviation
for i in range(1,logmd.size,1):
    err_W=err_W+((10**logHMFd[i]-FWl[i])/(10**logHMFd[i]))**2
    err_s=err_s+((10**logHMFd[i]-FWsl[i])/(10**logHMFd[i]))**2
    err_c2=err_c2+((10**logHMFd[i]-FWc2l[i])/(10**logHMFd[i]))**2
    err_c4=err_c4+((10**logHMFd[i]-FWc4l[i])/(10**logHMFd[i]))**2
    err_c2s=err_c2s+((10**logHMFd[i]-FWc2sl[i])/(10**logHMFd[i]))**2
    
    lerr_s[i]=((10**logHMFd[i]-FWsl[i])/(10**logHMFd[i]))**2
    lerr_c2[i]=((10**logHMFd[i]-FWc2l[i])/(10**logHMFd[i]))**2
    lerr_c4[i]=((10**logHMFd[i]-FWc4l[i])/(10**logHMFd[i]))**2
    lerr_W[i]=((10**logHMFd[i]-FWl[i])/(10**logHMFd[i]))**2
    lerr_c2s[i]=((10**logHMFd[i]-FWc2sl[i])/(10**logHMFd[i]))**2
    
    lacum_W[i]=err_W
    lacum_s[i]=err_s
    lacum_c2s[i]=err_c2s
    
print(err_W/err_s,err_s/err_s,err_c2/err_s,err_c4/err_s,err_c2s/err_s)    

#%%
#HMF (Halo Mass Function plot) for different filters
logmd2=np.linspace(6.95,12,30)
logHMFd2=logHMF33COCO(logmd2)

#plot dels diferents filtres
plt.plot(10**logmd2,10**logHMFd2,'o',color='violet')
plt.plot(10**logmd,FWs(logmd, cs, betas))
#plt.plot(10**logmd,FWc2(logmd, a1, a2, cc2))
plt.plot(10**logmd,FW(logmd,al,c,mWDM))
#plt.plot(10**logmd,FWc4(logmd, b1,b2,b3,b4, cc4))
plt.plot(10**logmd,FWc2s(logmd, c2s, beta2s))
plt.xlim([10**7,10**12])
plt.ylim([10**(-4),2])

plt.xscale('log')
plt.yscale('log')

plt.xlabel('M ($M_o/h$)')
plt.ylabel('$dn/d log M$ [$h^3Mpc^{-3}$]')
#%%
#the following two cells are a combined figure for the deviation and the values of HMF
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

# Simple data to display in various forms
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

fig = plt.figure()
# set height ratios for subplots
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 

# the first subplot
ax0 = plt.subplot(gs[0])
ax0.set_ylabel('$dn/d log M$ [$h^3Mpc^{-3}$]')
# log scale for axis Y of the first subplot
ax0.set_xscale("log")
#ax0.set_yscale("log")

line0, = ax0.plot(10**logmd2, 10**logHMFd2,'x', color='orangered')
line1, = ax0.plot(10**logmd,FWs(logmd, cs, betas),color='green')
line2, = ax0.plot(10**logmd,FWc2s(logmd, c2s, beta2s), color='darkviolet')
#line3, = ax0.plot(10**logmd,FW(logmd,al,c,mWDM), color='royalblue')

plt.xlim([2.05*10**7,10**12])
plt.ylim([8*10**(-4),1.2])

# the second subplot
# shared axis X
ax1 = plt.subplot(gs[1], sharex = ax0)
ax1.set_ylabel('$\Delta^2_i$')
line4, = ax1.plot(10**logmd, lerr_s, color='green')#, linestyle='--')
line5, = ax1.plot(10**logmd, lerr_c2s, color='darkviolet')#, linestyle='--')
line6, = ax1.plot(10**logmd, np.zeros(logmd.size), color='orangered', linestyle='--')
#line7, = ax1.plot(10**logmd, lerr_W, color='royalblue')
plt.setp(ax0.get_xticklabels(), visible=False)
# remove last tick label for the second subplot
yticks = ax1.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

plt.ylim([-0.005,0.035])
# put legend on first subplot
ax0.legend((line0, line1, line2, line4, line5), ('COCO', 'Smooth-k','Double Box'), loc='upper right')

fig.text(0.5, 0.04, '$M$ [$h^{-1}M_o$]', ha='center')
#fig.text(0.04, 0.5, '$dn/d log M$ [$h^3Mpc^{-3}$]', va='center', rotation='vertical')
# remove vertical gap between subplots
plt.subplots_adjust(hspace=.0)
plt.show()

#%%
#this cell includes the k-sharp filtered HMF as a difference from the previous cell
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

# Simple data to display in various forms
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

fig = plt.figure()
# set height ratios for subplots
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 

# the first subplot
ax0 = plt.subplot(gs[0])
ax0.set_ylabel('$dn/d log M$ [$h^3Mpc^{-3}$]')
# log scale for axis Y of the first subplot
ax0.set_xscale("log")
#ax0.set_yscale("log")

line0, = ax0.plot(10**logmd2, 10**logHMFd2,'x', color='orangered')
line1, = ax0.plot(10**logmd,FWs(logmd, cs, betas),color='green')
line2, = ax0.plot(10**logmd,FWc2s(logmd, c2s, beta2s), color='darkviolet')
line3, = ax0.plot(10**logmd,FW(logmd,al,c,mWDM), color='royalblue')

plt.xlim([2.05*10**7,10**12])
plt.ylim([10**(-4),2])

# the second subplot
# shared axis X
ax1 = plt.subplot(gs[1], sharex = ax0)
ax1.set_ylabel('$\sum \Delta^2_i/\sum \Delta^2_{SK}$')
line4, = ax1.plot(10**logmd, lacum_s/err_s, color='green')#, linestyle='--')
line5, = ax1.plot(10**logmd, lacum_c2s/err_s, color='darkviolet')#, linestyle='--')
line6, = ax1.plot(10**logmd, np.zeros(logmd.size), color='orangered', linestyle='--')
line7, = ax1.plot(10**logmd, lacum_W/err_s, color='royalblue')#, linestyle='--')
plt.setp(ax0.get_xticklabels(), visible=False)
# remove last tick label for the second subplot
yticks = ax1.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

# put legend on first subplot
ax0.legend((line0, line1, line2, line3, line4, line5), ('COCO', 'Smooth-k','Smooth Double Box', 'K-sharp'), loc='upper right')

fig.text(0.5, 0.04, '$M$ [$h^{-1}M_o$]', ha='center')
#fig.text(0.04, 0.5, '$dn/d log M$ [$h^3Mpc^{-3}$]', va='center', rotation='vertical')
# remove vertical gap between subplots
plt.subplots_adjust(hspace=.0)
plt.show()
#%%
#the three studied filters with different parameters in addition to the top-hat filter
linkR=np.logspace(-2,2,5000)
linkRfall=np.linspace(2**(1/beta2s),100,10000)
linkRbox=np.linspace(0,2**(1/beta2s)-1e-6,10000)
zeros=np.zeros(10000)

linkRtotal=np.concatenate((zeros,1/linkRfall))
linkRtotal2=np.concatenate((1/linkRbox,1/linkRfall))

plt.plot(np.log10(linkR*cs),(1+(linkR)**betas)**(-1))
plt.plot(np.log10(linkR*c2s),0.75*(1+(linkR)**1e5)**(-1)+0.25*(1+(linkR*(1-0.403))**1e5)**(-1))
#plt.plot(np.log10(1/linkRtotal2*c2s),(1+(1/linkRtotal2/2**(1/beta2s))**1e10)**(-1)+0.5*(linkRtotal)**beta2s)
#plt.plot(np.log10(1/linkRtotal*c2s),(linkRtotal)**beta2s)
#plt.plot(np.log10(linkR*1),(0.25*(1+(linkR*(1-0.52))**1e5)**(-1)+0.75*(1+(linkR*(1-0.049))**1e5)**(-1)))
plt.plot(np.log10(linkR*0.75),(3*(np.sin(linkR)-linkR*np.cos(linkR)))/linkR**3)
plt.plot(np.log10(linkR*0.9),(1+(linkR)**1e5)**(-1))

plt.xlabel('$log(kcR)$')
plt.ylabel('$W(kR)$')

plt.legend(('SK ($\hat{\\beta} =6.35$, $c=3.42$)','DB ($\\beta =-0.403$, $c=3.33$)', 'TH', 'KS (c=0.9)'))
plt.xlim([-0.5,1.5])

#%%
#obtaining the rough values of the spectral indexs for the used CDM and WDM power spectrum
link4=np.logspace(-8,4,1000)

#transfer matrix from Viel et.al (2005)
aaalpha=0.049*(omWDM/0.25)**(0.11)*(hreal/0.7)**(1.22)*(mWDM)**(-1.11)/hreal
aabeta=2*1.12
aagamma=-5/1.12

plt.plot(link4,PSl(link4,al)) #CDM power spectrum
plt.plot(link4,PSl(link4,al)*(1+(aaalpha*link4)**aabeta)**(2*aagamma)) #WDM power spectrum
#plt.plot(link4,PSl(link4,al)*(1+(aaalpha*link4)**aabeta)**(2*aagamma)*(1+(0.02*link4)**4)**(-5))  #an additional function is added to obtain a more negative slope

#rough fits for the different regimes
plt.plot(link4,link4**(-2.8)*2.5e3) 
plt.plot(link4,link4**(0.966)*2.5e8)
plt.plot(link4,link4**(-24+0.966)*6e38)
plt.plot(link4,link4**(-12)*8e16)

plt.ylim([10**(-25),10**(10)])

plt.xlabel('k ($h/Mpc$)')
plt.ylabel('P(k) ($1/Mpc^6$)')
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
loglink5=np.linspace(-4,5,1000)
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



