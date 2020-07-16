#
# utility module for various modified gravity codes
# (1) Cosmology definitions
# (2) -xsph1- and  -dsph1- provide spherical Bessel function and it's derivative
# (3) pk(z), dlnD, ndlnD etc reads Patrick's files
# (4) read_ps, read_dlnD, read_nDlnD supports  pk(z), dlnD, ndlnD respectively.
#
import numpy as np; from numpy import *
import inputs as inputs
import xcls as xcls
#
from scipy import interpolate
import scipy as sp ; from scipy import *
from scipy.misc import *
from scipy.special import *
#
#####
def a(z,M): return 1/(1+z) # scalefactor
def factor(z,M): return (M*(1+z)**3 + (1-M))  # useful fcator
def h(z,M): return factor(z,M)**.5  # hubble
def f(z,M): return (M*(1+z)**3/factor(z,M))**0.6  # dlnD+/dln a
def ps_prefact(z,M): return h(z,M)**2*f(z,M)**2*a(z,M)**2 # preftor in P_\perp
def drdz(z,M): return 3000*factor(z,M)**(-0.5)  # dr/dz
def rInt(z,M): return 3000*factor(z,M)**(-0.5)  # comoving r as function z

################
def xsph1(x,el):
################
    # Computes the spherical Bessel functions
    pfactor=(pi/2.0/x)**.5
    y=pfactor*jv(el+1./2,x) 
    return y

################
def dsph1(x,el):
################
    # Computes the derivative of sph. Bessel function.
    pfactor=(pi/2.0/x)**.5
    dy= el*jv(el-1./2,x)-(el+1)*jv(el+3./2,x)
    dy=pfactor*dy/(2*el+1)  
    return dy 


####################
def read_ps(infile):
####################
    # reads columns from Patrick's power spectrum files 
    ps0= loadtxt(infile)
    nps = size(ps0,0);
    ak = ps0[0:nps-1,0];
    lnps=ps0[0:nps-1,1]
    nlnps=ps0[0:nps-1,2]
    for ik in range(0,nps-1):
        lnps[ik]=2.0*pi**2*lnps[ik]/ak[ik]/ak[ik]/ak[ik]
        nlnps[ik]=2.0*pi**2*nlnps[ik]/ak[ik]/ak[ik]/ak[ik]
    return ak,lnps,nlnps

######################
def read_dlnD(infile):
######################
    # read columns from Patrick's f = dln D/ dln a files 
    ps1= loadtxt(infile)
    nps = size(ps1,0);
    ak = ps1[0:nps-1,0];
    dlnD=ps1[0:nps-1,4]
    #print dlnD
    return ak,dlnD

#######################
def read_ndlnD(infile):
#######################
    # reads colums from nonlinear f(k,z) files
    ps1= loadtxt(infile)
    nps = size(ps1,0);
    ak = ps1[0:nps-1,0];
    ndlnD=ps1[0:nps-1,1]
    return ak, ndlnD

##################
def pkz(dir_name):
##################
    #Reads files and then returns p(k,z)
    #
    z=[0,0.25, 0.5,0.75, 1, 1.25, 1.50,1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]
    nz=np.size(z)
    #
    file0=dir_name+inputs.Dkfilez0  #"Dk_LCDM_z0.dat"
    k,lnps0,nlnps0=read_ps(file0); nk=size(k)
    nk= 697;
    pkL=np.empty((nk,nz))
    pkN=np.empty((nk,nz))
    #
    print("from Patrick's file:")
    #
    filez=dir_name+inputs.Dkfilename  #"/Dk_LCDM_z"
    for ik in range(0,nz):
        file_name=filez+str(z[ik])+".dat"
        x1,lnps1,nlnps1=read_ps(file_name)
        nk= size(lnps1)
        for jk in range(0,nk):
            pkL[jk][ik]=lnps1[jk]
            pkN[jk][ik]=nlnps1[jk] 
    #    
    fL=interpolate.interp2d(z,k,pkL,kind="cubic")
    fN=interpolate.interp2d(z,k,pkN,kind='cubic')
    #
    growthL = np.empty((inputs.Nnk,inputs.Nnz))
    growthN = np.empty((inputs.Nnk,inputs.Nnz))
    #
    pknewL = fL(inputs.znew,inputs.knew); pknewN = fN(inputs.znew,inputs.knew)
    #
    for ik in range(0,inputs.Nnz):
        for jk in range(0,inputs.Nnk):
            growthL[jk][ik]=(pknewL[jk][ik]/pknewL[jk][0])
            growthN[jk][ik]=(pknewN[jk][ik]/pknewN[jk][0])
    #
    for ik in range(0,inputs.Nnz): 
        for jk in range(0,inputs.Nnk):
            if (growthL[jk][ik] < 0.0 ): print ik, jk, growthL[jk][ik] 
            if (growthN[jk][ik] < 0.0 ): print ik, jk, growthN[jk][ik] 
            growthL[jk][ik] = (growthL[jk][ik])**.5
            growthN[jk][ik] = (growthN[jk][ik])**.5

    return k, inputs.knew, pkL, pkN, pknewL, pknewN, growthL, growthN
###################
def dlnD(dir_name):
###################
    #
    # Reads f(k,z)=dlnD_+/dln a from Patricks file and
    # interpolates them in a prescribed (k,z) grid.
    #
    z=[0,0.25,0.5,0.75,1,1.25,1.50,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4]
    nz=np.size(z)
    #
    print "dir_name=", dir_name
    #
    file0=dir_name+inputs.Dlinfilez0 #"Dlin_LCDM_z0.dat"
    k,dlnD=read_dlnD(file0); 
    nk=np.size(k)
    #
    DdlnD=np.empty((nk,nz))
    filez=dir_name+inputs.Dlinfilename #"/Dlin_LCDM_z"
    #
    for ik in range(0,nz-1):
        file_name=filez+str(z[ik])+".dat"
        x1,dlnD=read_dlnD(file_name)
        for jk in range(0,nk-1):
            DdlnD[jk][ik]=dlnD[jk]
    #
    dL = interpolate.interp2d(z,k,DdlnD,kind="cubic")
    #
    new_DdlnD = dL(inputs.znew,inputs.knew)
    #
    return z, k,DdlnD, inputs.znew, inputs.knew, new_DdlnD

####################
def ndlnD(dir_name):
####################
    #
    # nonlinear version of dlnD
    # Reads the Nonlinear f(k,z)=dlnD_+/dln a from Patricks file and
    # interpolates them in a prescribed (k,z) grid.
    #
    z=[0,0.25,0.5,0.75,1,1.25,1.50,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4]
    nz=size(z)
    #
    file0=dir_name+inputs.Dnlinfilez0 #"fP_LCDM_z0.dat"
    k,ndlnD=read_ndlnD(file0); 
    nk=size(k)
    #
    print "doneA"
    #
    filez=dir_name+inputs.Dnlinfilename
    nDdlnD=np.empty((nk,nz))
    for ik in range(0,nz-1):
        file_name=filez+str(z[ik])+".dat"
        x1,ndlnD=read_ndlnD(file_name)
        for jk in range(0,nk-1):
            nDdlnD[jk][ik]=ndlnD[jk]
    #
    ndL = interpolate.interp2d(z,k,nDdlnD,kind="cubic")
    new_ndlnD = ndL(inputs.znew,inputs.knew)
    #
    return z,k, ndlnD, inputs.znew, inputs.knew, new_ndlnD

###############
