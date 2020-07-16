# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 12:40:29 2017

@author: wb56
"""

#
import numpy as np
from scipy.integrate import quad;
import scipy as sp
from scipy import interpolate
import scipy.special
import matplotlib.pyplot as plt
import time
from matplotlib.colors import SymLogNorm

t0=time.clock()

#Nnz = grid points along z axis
#Nnk = grid points along k axis
#elp = \ell
elp=2; Nnz=2000; Nnk=200
#
kmin=-3.0; kmax=-0.3

znew = np.linspace(0.,4.0,Nnz); 
knew = np.logspace(kmin,kmax,Nnk)

#M = matter density
#r0 = cut off for selection function radius
M=0.25; r0=150

#
def factor(z,M): return (M*(1+z)**3 + (1-M))
    
#Computes comoving r as a function of z
def drdz(z,M): return 3000*factor(z,M)**(-0.5)
    
#Defines Bessell functions; 
#inputs: position(x), spherical \ell (el); outputs: spherical Bessell function    
def bessel(x,el):
    a=(np.pi/2.0/x)**.5
    b=a*scipy.special.jv(el+1./2,x) 
    return b
    
#Defines derivate of Bessel Functions
#inputs: position(x), spherical \ell(el); outputs: derivative of spherical Bessell function    
def dbessel(x,el):
    a=(np.pi/2.0/x)**.5
    b= a*((el*sp.special.jv(el-1./2,x)-(el+1)*sp.special.jv(el+3./2,x))/(2*el+1)) 
    return b 
 
#Establishes a read function to read correct columns from file for def pkz
#Converts linear power \Delta_L^{2}(k) and \Delta^{2}(k) to power spectrums
def read_ps(infile):
    ps= np.loadtxt(infile)
    c = np.size(ps,0);
    ak = ps[0:c-1,0];
    lnps=ps[0:c-1,1]
    nlnps=ps[0:c-1,2]
    
    for ik in range(0,c-1):
        lnps[ik]=2.0*np.pi**2*lnps[ik]/ak[ik]**3
        nlnps[ik]=2.0*np.pi**2*nlnps[ik]/ak[ik]**3
    return ak,lnps,nlnps
    
#Reads files using read_ps function
#inputs: linear power spectrum(lnps), non-linear power spectrum(nlnps),
#Interpolates linear and non-linear P(k), with respect to z, k.
#Calculates linear and non linear growth factor (growthL, growthN) from interpolated P(k) and P_L(k)
#Returns interpolated k, non-linear P(k), linear P(k), and non interpolated values
#Also returns linear and non-linear growth factors
def pkz(dir_name):

    z     =[0,0.25,0.5,0.75,1,1.25,1.50,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4]
    nz    =np.size(z)
    file0 =dir_name+Dkfilez0
    
    k,lnps0,nlnps0=read_ps(file0);nk=np.size(k);nk= 697;pkL=np.empty((nk,nz));pkN=np.empty((nk,nz))
    
    filez=dir_name+Dkfilename
    for ik in range(0,nz):
        file_name=filez+str(z[ik])+".dat"
        x1,lnps1,nlnps1=read_ps(file_name)
        nk= np.size(lnps1)
        for jk in range(0,nk):
            pkL[jk][ik]=lnps1[jk]
            pkN[jk][ik]=nlnps1[jk] 
        
    fL=interpolate.interp2d(z,k,pkL,kind="cubic")
    fN=interpolate.interp2d(z,k,pkN,kind='cubic')
    
    growthL=np.empty((Nnk,Nnz));growthN=np.empty((Nnk,Nnz));pknewL=fL(znew,knew);pknewN=fN(znew,knew)

    for ik in range(0,Nnz):
        for jk in range(0,Nnk):
            growthL[jk][ik] =(pknewL[jk][ik]/pknewL[jk][0])
            growthN[jk][ik] =(pknewN[jk][ik]/pknewN[jk][0])
    
    for ik in range(0,Nnz): 
        for jk in range(0,Nnk):
            growthL[jk][ik] =(growthL[jk][ik])**(0.5)
            growthN[jk][ik] =(growthN[jk][ik])**(0.5)

    return k, knew, pkL, pkN, pknewL, pknewN, growthL, growthN
    
    
#Establishes a read function to read correct columns from file for def dlnD
def read_dlnD(infile):
    lnD= np.loadtxt(infile)
    c = np.size(lnD,0);
    ak = lnD[0:c-1,0];
    dlnD=lnD[0:c-1,4]
    return ak,dlnD  

#Reads files using read_dlnD function
#Interpolates dlnD with repsect to z, k
#Returns interolated z, k, linear growing mode and pre-interpolated values 
def dlnD(dir_name):
    z=[0,0.25,0.5,0.75,1,1.25,1.50,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4]
    nz=np.size(z)    
    file0=dir_name+Dlinfilez0
    k,dlnD=read_dlnD(file0) 
    nk=np.size(k)

    DdlnD=np.empty((nk,nz))
    filez=dir_name+Dlinfilename
    for ik in range(0,nz-1):
        file_name=filez+str(z[ik])+".dat"
        x1,dlnD=read_dlnD(file_name)
        for jk in range(0,nk-1):
            DdlnD[jk][ik]=dlnD[jk]
    dL = sp.interpolate.interp2d(z,k,DdlnD,kind="cubic")
    newDdlnD = dL(znew,knew)
    return z, k,DdlnD, znew, knew, newDdlnD

#Establishes a read function to read correct columns from file for def nddlnD
def read_ndlnD(infile):
    nlnD= np.loadtxt(infile)
    c = np.size(nlnD,0);
    ak = nlnD[0:c-1,0];
    ndlnD=nlnD[0:c-1,1]
    return ak, ndlnD
    
#Reads files using read_ndlnD function
#Interpolates ndlnD with repsect to z, k
#Returns interolated z, k, non-linear growing mode and pre-interpolated values
def ndlnD(dir_name):
    z      =[0,0.25,0.5,0.75,1,1.25,1.50,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4]
    nz     =np.size(z)
    file0  =dir_name+Dnlinfilez0
    
    k,ndlnD=read_ndlnD(file0);nk=np.size(k)
    
    filez=dir_name+Dnlinfilename
    nDdlnD=np.empty((nk,nz))
    for ik in range(0,nz-1):
        file_name=filez+str(z[ik])+".dat"
        x1,ndlnD=read_ndlnD(file_name)
        for jk in range(0,nk-1):
            nDdlnD[jk][ik]=ndlnD[jk]
    
    ndL = interpolate.interp2d(z,k,nDdlnD,kind="cubic")
    newndlnD = ndL(znew,knew)
    
    return z, k, ndlnD, znew, knew, newndlnD


###############################################################################

#Calls linear f_P, non-linear f_P and linear and non-linear P(k)
def kernels(el):
    tz, tk, tf,tznew, tknew, tnew_f = dlnD(dir_name2)
    z, k, f, znew, knew, new_f = ndlnD(dir_name3)
    k, knew, pkL, pkN, pknewL, pknewN, growthL, growthN = pkz(dir_name1)
    
    Nnk=200; ka=np.logspace(kmin,kmax,Nnk)
    
    Nnz=np.size(znew);Nnk=np.size(knew);r=np.empty((Nnz)) 
    varphi=np.empty((Nnz));dvarphi=np.empty((Nnz))

#Creates selection function \varphi(r) and also finds its derivative
#Integrates comoving r as func. of , along interpolated z axis (znew) to find distance r
    for i in range(0,Nnz):
        r[i]       =quad(drdz,0,znew[i],args=(M))[0]
        varphi[i]  =np.exp(-r[i]**2/r0**2)
        dvarphi[i] =(-2*r[i]/r0**2)*np.exp(-r[i]**2/r0**2)

#Creates empty arrays for linear and non-linear C_\ell's
    NI0= np.empty((Nnz));LI0= np.empty((Nnz));NLI0=np.empty((Nnz))
    
    NI1= np.empty((Nnz));LI1= np.empty((Nnz));NLI1=np.empty((Nnz))
    
    NIl0=np.empty((Nnk,Nnk));LIl0=np.empty((Nnk,Nnk));NLIl0=np.empty((Nnk,Nnk));
    
    NIl1=np.empty((Nnk,Nnk));LIl1=np.empty((Nnk,Nnk));NLIl1=np.empty((Nnk,Nnk));
    
    S=np.empty((Nnk,Nnz));
    dS=np.empty((Nnk,Nnz))
    
# Calculates the spherical Bessell function as a func. of k1, and distnace r
    for k1 in range(0,Nnk):
        for i in range(0,Nnz):
            Y=ka[k1]*(r[i]+1.e-8)
            S[k1][i]=bessel(Y,el)  
            dS[k1][i]=dbessel(Y,el)  
   
    for k1 in range(0,Nnk): 
        print (k1, Nnk)
        for k2 in range(0,Nnk):
            for i in range(0,Nnz):
                
                S1= S[k1,i]; S2= S[k2,i]
                dS1= dS[k1,i]; dS2= dS[k2,i]
        
        #Calculates window functions pre-integration
        #Eqs. (3.51, 3.53)
                W0      =r[i]**2*ka[k1]*S1*S2*varphi[i]*drdz(znew[i],M)
                NI0[i]  = W0*growthN[k2,i]  
                LI0[i]  =W0*growthL[k2,i]
                NLI0[i] =W0*(growthN[k2,i]*growthL[k2,i])**(0.5)
                
                W1      =r[i]**2*drdz(znew[i],M)*tnew_f[k2,i]
                newW1      =W1*(ka[k1]**2*dS1*dS2*varphi[i]+ka[k1]*S1*dS2*dvarphi[i])
                NI1[i]  =W1*growthN[k2,i] 
                LI1[i]  =W1*growthL[k2,i]
                NLI1[i] =W1*(growthN[k2,i]*growthL[k2,i])**(0.5)
                
                
                np.save('W0.npy',W0)
                np.save('W1.npy',newW1)
                
        #Performs the integration in (3.51, 3.53)   
            NIy   =sp.integrate.simps(NI0,znew);  NIl0[k1,k2]=NIy            
            LIy   =sp.integrate.simps(LI0,znew);  LIl0[k1,k2]=LIy
            NLIy  =sp.integrate.simps(NLI0,znew); NLIl0[k1,k2]=NLIy
            NI1y  =sp.integrate.simps(NI1,znew);  NIl1[k1,k2]=NI1y
            LI1y  =sp.integrate.simps(LI1,znew);  LIl1[k1,k2]=LI1y
            NLI1y =sp.integrate.simps(NLI1,znew); NLIl1[k1,k2]=NLI1y
 
            np.save('NIy.npy', NIy)
            np.save('LIy.npy', LIy)
            np.save('NLIy.npy', NLIy)
            np.save('NI1y.npy', NI1y)
            np.save('LI1y.npy', LI1y)
            np.save('NLI1y.npy', NLI1y)
#                      
    for k1 in range(0,Nnk):
        for k2 in range(0,Nnk):
            
            NIl0[k1,k2]  =(pknewN[k2,0])**(0.5)*NIl0[k1,k2]
            LIl0[k1,k2]  =(pknewL[k2,0])**(0.5)*LIl0[k1,k2]
            NLIl0[k1,k2] =(pknewN[k2,0]*pknewL[k2,0])**(0.25)*NLIl0[k1,k2]
            
            NIl1[k1,k2]  =(pknewN[k2,0])**(0.5)*NIl1[k1,k2]/ka[k2]           
            LIl1[k1,k2]  =(pknewL[k2,0])**(0.5)*LIl1[k1,k2]/ka[k2]
            NLIl1[k1,k2] =(pknewN[k2,0]*pknewL[k2,0])**(0.25)*NLIl1[k1,k2]/ka[k2]        

    for k1 in range(0,Nnk):
        print(k1, NIl0[k1,k1],LIl0[k1,k1],NLIl0[k1,k1],NIl1[k1,k1],LIl1[k1,k1],NLIl1[k1,k1])
        return NIl0,LIl0,NLIl0, NIl1,LIl1,NLIl1
        
        
def only_cls(NIl0,LIl0,NLIl0,NIl1,LIl1,NLIl1):

    NInt0= np.empty((Nnk));LInt0= np.empty((Nnk));NLInt0=np.empty((Nnk))
    NInt1= np.empty((Nnk));LInt1= np.empty((Nnk));NLInt1=np.empty((Nnk))
    
    Ncls1=np.empty((Nnk));Lcls1=np.empty((Nnk));NLcls1=np.empty((Nnk))   
    Ncls0=np.empty((Nnk));Lcls0=np.empty((Nnk));NLcls0=np.empty((Nnk))
    kf=np.logspace(kmin,kmax,Nnk)
    
    for k1 in range(0,Nnk):
        for k in range(0,Nnk):
            NInt0[k]   =NIl0[k1,k]*  NIl0[k1,k]*kf[k]*kf[k]
            LInt0[k]   =LIl0[k1,k]*  LIl0[k1,k]*kf[k]*kf[k]
            NLInt0[k]  =NLIl0[k1,k]* NLIl0[k1,k]*kf[k]*kf[k]
            NInt1[k]   =NIl1[k1,k]*  NIl0[k1,k]*kf[k]*kf[k]
            LInt1[k]   =LIl1[k1,k]*  LIl0[k1,k]*kf[k]*kf[k]
            NLInt1[k]  =NLIl1[k1,k]* NLIl0[k1,k]*kf[k]*kf[k]

#Integrates window functions over k to find C_\ell's            
        Ncls0[k1]  = sp.integrate.simps(NInt0,kf)
        Lcls0[k1]  = sp.integrate.simps(LInt0,kf)
        NLcls0[k1] = sp.integrate.simps(NLInt0,kf)
        Ncls1[k1]  = sp.integrate.simps(NInt1,kf)
        Lcls1[k1]  = sp.integrate.simps(LInt1,kf)
        NLcls1[k1] = sp.integrate.simps(NLInt1,kf)

    return kf,Ncls0,Lcls0,NLcls0,Ncls1,Lcls1,NLcls1


def cls_cov(NIl0,LIl0,NLIl0,NIl1,LIl1,NLIl1):

    NInt0= np.empty((Nnk));LInt0= np.empty((Nnk));NLInt0=np.empty((Nnk))
    NInt1= np.empty((Nnk));LInt1= np.empty((Nnk));NLInt1=np.empty((Nnk))   
    NInt2= np.empty((Nnk))  ;LInt2= np.empty((Nnk));NLInt2=np.empty((Nnk))
    
    Ncov1=np.empty((Nnk,Nnk));Lcov1=np.empty((Nnk,Nnk));NLcov1=np.empty((Nnk,Nnk));
    Ncov0=np.empty((Nnk,Nnk));Lcov0=np.empty((Nnk,Nnk));NLcov0=np.empty((Nnk,Nnk));
    Ncov2=np.empty((Nnk,Nnk));Lcov2=np.empty((Nnk,Nnk));NLcov2=np.empty((Nnk,Nnk));
    kf=np.logspace(kmin,kmax,Nnk);
    
    for k1 in range(0,Nnk):
        for k2 in range(0,Nnk):
            for k in range(0,Nnk):
                
                NInt0[k]=NIl0[k1,k]*NIl0[k2,k]*kf[k]*kf[k]
                LInt0[k]=LIl0[k1,k]*LIl0[k2,k]*kf[k]*kf[k]
                NLInt0[k]=NLIl0[k1,k]*NLIl0[k2,k]*kf[k]*kf[k]
                
                NInt1[k]=NIl1[k1,k]*NIl0[k2,k]*kf[k]*kf[k]
                LInt1[k]=LIl1[k1,k]*LIl0[k2,k]*kf[k]*kf[k]
                NLInt1[k]=NLIl1[k1,k]*NLIl0[k2,k]*kf[k]*kf[k]
            
                NInt2[k]=NIl1[k1,k]*NIl1[k2,k]*kf[k]*kf[k]
                LInt2[k]=LIl1[k1,k]*LIl1[k2,k]*kf[k]*kf[k]
                NLInt2[k]=NLIl1[k1,k]*NLIl1[k2,k]*kf[k]*kf[k]
            
            Ncov0[k1,k2]  =sp.integrate.simps(NInt0,kf)
            Lcov0[k1,k2]  =sp.integrate.simps(LInt0,kf)
            NLcov0[k1,k2] =sp.integrate.simps(NLInt0,kf)
            
            Ncov1[k1,k2]  =sp.integrate.simps(NInt1,kf)
            Lcov1[k1,k2]  =sp.integrate.simps(LInt1,kf)
            NLcov1[k1,k2] =sp.integrate.simps(NLInt1,kf)
    
            Ncov2[k1,k2]  =sp.integrate.simps(NInt2,kf)
            Lcov2[k1,k2]  =sp.integrate.simps(LInt2,kf)
            NLcov2[k1,k2] =sp.integrate.simps(LInt2,kf)
        
    
    return kf,Ncov0,Lcov0,NLcov0,Ncov1,Lcov1,NLcov1,Ncov2,Lcov2,NLcov2

def do_lcdm():
    
    NIl0,LIl0,NLIl0, NIl1,LIl1,NLIl1=kernels(elp)
    
    kf,Ncov0,Lcov0,NLcov0,Ncov1,Lcov1,NLcov1,Ncov2,Lcov2,NLcov2=cls_cov(NIl0,LIl0,NLIl0,NIl1,LIl1,NLIl1)

    return
    
for choice1 in range(1,2):
    for choice2 in range(1,4):
        if(choice1==1):
            if(choice2==1):
                cosmology="LCDM"
                datadir=""
                dir_name3=datadir+"fP/"
                dir_name2=datadir+"Dlin/" 
                dir_name1=datadir+"Dk/"
                #
                Dkfilename="/Dk_LCDM_z" 
                Dlinfilename="/Dlin_LCDM_z"
                Dnlinfilename="/fP_LCDM_z"
                #
                Dkfilez0="Dk_LCDM_z0.dat"
                Dlinfilez0="Dlin_LCDM_z0.dat"
                Dnlinfilez0="fP_LCDM_z0.dat"
                #
                do_lcdm()
                
def do_all():
       
    NIl0,LIl0,NLIl0, NIl1,LIl1,NLIl1=kernels(elp)
    
    kf,Ncov0,Lcov0,NLcov0,Ncov1,Lcov1,NLcov1,Ncov2,Lcov2,NLcov2=cls_cov(NIl0,LIl0,NLIl0,NIl1,LIl1,NLIl1)
    
    kf,Ncls0,Lcls0,NLcls0,Ncls1,Lcls1,NLcls1=only_cls(NIl0,LIl0,NLIl0,NIl1,LIl1,NLIl1)

    Ncls=np.empty((Nnk))
    Lcls=np.empty((Nnk))
    NLcls=np.empty((Nnk))
    for ik in range(0,Nnk):
        Ncls[ik]=Ncls0[ik]+2*Ncls1[ik]
        Lcls[ik]=Lcls0[ik]+2*Lcls1[ik]
        NLcls[ik]=NLcls0[ik]+2*NLcls1[ik]

tf=time.clock()
t=tf-t0
print(t)