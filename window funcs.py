# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 05:22:05 2017

@author: Wayne
"""

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
                