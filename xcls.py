#
# Module to compute linear redshift space distortions
# kernels(el) compute the two kernels and cls3D compute the cls
#
import xcosmo as xcosmo
import inputs as inputs
#
import numpy as np; from numpy import *
from scipy.integrate import quad;
import scipy as sp; from scipy import *
import matplotlib.pyplot as plt; from pylab import *

#python inputs.py xcls.py xcosmo.py 

#from scipy.special import *  # spherical Bessel
#from scipy.misc import *
#
#################
def kernels(el):
#################
    ##
    print "dn1=", inputs.dir_name1
    print "dn2=", inputs.dir_name2
    print "dn3=", inputs.dir_name3
    #
    #
    ### linear fp
    tz,tk,tf,tznew,tknew,tnew_f = xcosmo.dlnD(inputs.dir_name2)
    print "doneA"
    ### nonlinear fp
    z,k,f,znew, knew, new_f = xcosmo.ndlnD(inputs.dir_name3)
    print "doneB"
    #### linear and nonlinear ps
    k,knew,pkL, pkN,pknewL, pknewN,growthL,growthN = xcosmo.pkz(inputs.dir_name1)
    print "doneC"
    ##############
    #
    ka=logspace(inputs.kmin,inputs.kmax,inputs.Nnk)
    #
    nz=np.size(z); nk=np.size(k)
    Nnz=np.size(znew); Nnk=np.size(knew)
    r= np.empty((Nnz)) 
    phi = np.empty((Nnz)) # selction function
    dphi= np.empty((Nnz))
    #
    for i in range(0,Nnz):
        r[i]=quad(xcosmo.rInt,0, znew[i],args=(inputs.M))[0]
        phi[i] = exp(-r[i]**2/inputs.r0**2)
        dphi[i]= (-2*r[i]/inputs.r0**2)*exp(-r[i]**2/inputs.r0**2)
    #
    #for k1 in range(0,inputs.Nnk):
    #    for i in range(0,inputs.Nnz):
    #        argu1=ka[k1]*(r[i]+1.e-8)
    #        xa1[k1][i]=xcosmo.xsph1(argu1,el)  
    #        dxa1[k1][i]=xcosmo.dsph1(argu1,el)  
    #
    #
    #
    # The three sets of kernels are for D_Nl,
    #
    NtestI0= np.empty((inputs.Nnz)); 
    LtestI0= np.empty((inputs.Nnz));
    NLtestI0=np.empty((inputs.Nnz))
    #
    NtestI1= np.empty((inputs.Nnz));
    LtestI1= np.empty((inputs.Nnz))
    NLtestI1=np.empty((inputs.Nnz))
    #
    NIl0=np.empty((inputs.Nnk,inputs.Nnk)); 
    LIl0=np.empty((inputs.Nnk,inputs.Nnk));
    NLIl0=np.empty((inputs.Nnk,inputs.Nnk));
    NIl1=np.empty((inputs.Nnk,inputs.Nnk)); 
    LIl1=np.empty((inputs.Nnk,inputs.Nnk));
    NLIl1=np.empty((inputs.Nnk,inputs.Nnk));
    #
    xa1=np.empty((inputs.Nnk,inputs.Nnz));
    dxa1=np.empty((inputs.Nnk,inputs.Nnz))
    #
    for k1 in range(0,inputs.Nnk):
        for i in range(0,inputs.Nnz):
            argu1=ka[k1]*(r[i]+1.e-8)
            xa1[k1][i]=xcosmo.xsph1(argu1,el)  
            dxa1[k1][i]=xcosmo.dsph1(argu1,el)  
    #
    for k1 in range(0,inputs.Nnk): 
        print k1, inputs.Nnk
        for k2 in range(0,inputs.Nnk):
            for i in range(0,inputs.Nnz):
                #
                temp1= xa1[k1,i]; temp2= xa1[k2,i]
                dtemp1= dxa1[k1,i]; dtemp2= dxa1[k2,i]
                # Computes the 0th window
                xx0=r[i]**2*ka[k1]*temp1*temp2*phi[i]*xcosmo.drdz(znew[i],inputs.M)
                NtestI0[i]= xx0*growthN[k2,i];  LtestI0[i]=xx0*growthL[k2,i]
                NLtestI0[i]=xx0*(growthN[k2,i]*growthL[k2,i])**.5
                # computes the 1st window
                xx1=r[i]**2*xcosmo.drdz(znew[i],inputs.M)*tnew_f[k2,i]
                xx1=xx1*(ka[k1]**2*dtemp1*dtemp2*phi[i]+ ka[k1]*temp1*dtemp2*dphi[i]) ##(d\phi term) 
                NtestI1[i]=xx1*growthN[k2,i]; LtestI1[i]=xx1*growthL[k2,i]
                NLtestI1[i]=xx1*(growthN[k2,i]*growthL[k2,i])**.5
                #
            NtestIy=sp.integrate.simps(NtestI0,znew)
            LtestIy=sp.integrate.simps(LtestI0,znew)
            NLtestIy=sp.integrate.simps(NLtestI0,znew)
            #
            NtestI1y=sp.integrate.simps(NtestI1,znew)
            LtestI1y=sp.integrate.simps(LtestI1,znew)
            NLtestI1y=sp.integrate.simps(NLtestI1,znew)
            #
            #
            NIl0[k1,k2]= NtestIy; LIl0[k1,k2]= LtestIy;  NLIl0[k1,k2]=NLtestIy
            NIl1[k1,k2]= NtestI1y; LIl1[k1,k2]= LtestI1y; NLIl1[k1,k2]=NLtestI1y           
            #
    #for k1 in range(0,Nnk):
    #    print NIl0[k1,k1], LIl0[k1,k1], NLIl0[k1,k1], NIl1[k1,k1], LIl1[k1,k1], NLIl1[k1,k1]
    #    #
    #exit()
    for k1 in range(0,Nnk):
        for k2 in range(0,Nnk):
            #
            NIl0[k1,k2]=(pknewN[k2,0])**.5*NIl0[k1,k2]
            LIl0[k1,k2]=(pknewL[k2,0])**.5*LIl0[k1,k2]
            NLIl0[k1,k2]=(pknewN[k2,0]*pknewL[k2,0])**.25*NLIl0[k1,k2]
            #
            NIl1[k1,k2]=(pknewN[k2,0])**.5*NIl1[k1,k2]/ka[k2]           
            LIl1[k1,k2]=(pknewL[k2,0])**.5*LIl1[k1,k2]/ka[k2]
            NLIl1[k1,k2]=(pknewN[k2,0]*pknewL[k2,0])**.25*NLIl1[k1,k2]/ka[k2]

    for k1 in range(0,Nnk):
        print k1, NIl0[k1,k1], LIl0[k1,k1], NLIl0[k1,k1], NIl1[k1,k1], LIl1[k1,k1], NLIl1[k1,k1]
    #
    #print "kernel done"
    #
    #exit()

    #print "a=", NIl0[2,2], NIl0[3,3], NIl0[4,4]
    #print "from kernel", NIl0[70,70],LIl0[70,70],NLIl0[70,70], NIl1[70,70],LIl1[70,70],NLIl1[70,70]   
    #print "from kernel", NIl0[70,23],LIl0[70,23],NLIl0[70,23], NIl1[70,23],LIl1[20,23],NLIl1[70,23]        

    return NIl0,LIl0,NLIl0, NIl1,LIl1,NLIl1
#
#############################################
def cls_cov(NIl0,LIl0,NLIl0,NIl1,LIl1,NLIl1):
#############################################
#
    NIntegral0= np.empty((inputs.Nnk)); 
    LIntegral0= np.empty((inputs.Nnk));
    NLIntegral0=np.empty((inputs.Nnk));
    #
    NIntegral1= np.empty((inputs.Nnk)); 
    LIntegral1= np.empty((inputs.Nnk));
    NLIntegral1=np.empty((inputs.Nnk));
    
    NIntegral2= np.empty((inputs.Nnk)); 
    LIntegral2= np.empty((inputs.Nnk));
    NLIntegral2=np.empty((inputs.Nnk))
    #
    Ncov1=np.empty((inputs.Nnk,inputs.Nnk)); 
    Lcov1=np.empty((inputs.Nnk,inputs.Nnk));
    NLcov1=np.empty((inputs.Nnk,inputs.Nnk));
    #
    Ncov0=np.empty((inputs.Nnk,inputs.Nnk)); 
    Lcov0=np.empty((inputs.Nnk,inputs.Nnk));
    NLcov0=np.empty((inputs.Nnk,inputs.Nnk));
    #
    Ncov2=np.empty((inputs.Nnk,inputs.Nnk)); 
    Lcov2=np.empty((inputs.Nnk,inputs.Nnk));
    NLcov2=np.empty((inputs.Nnk,inputs.Nnk));
    #
    kdummy=logspace(inputs.kmin,inputs.kmax,inputs.Nnk);
    #
    for k1 in range(0,inputs.Nnk):
        print "k1=", k1
        for k2 in range(0,inputs.Nnk):
            for k in range(0,inputs.Nnk):
                #
                NIntegral0[k]=NIl0[k1,k]*NIl0[k2,k]*kdummy[k]*kdummy[k]
                LIntegral0[k]=LIl0[k1,k]*LIl0[k2,k]*kdummy[k]*kdummy[k]
                NLIntegral0[k]=NLIl0[k1,k]*NLIl0[k2,k]*kdummy[k]*kdummy[k]
                #
                NIntegral1[k]=NIl1[k1,k]*NIl0[k2,k]*kdummy[k]*kdummy[k]
                LIntegral1[k]=LIl1[k1,k]*LIl0[k2,k]*kdummy[k]*kdummy[k]
                NLIntegral1[k]=NLIl1[k1,k]*NLIl0[k2,k]*kdummy[k]*kdummy[k]
            #
                NIntegral2[k]=NIl1[k1,k]*NIl1[k2,k]*kdummy[k]*kdummy[k]
                LIntegral2[k]=LIl1[k1,k]*LIl1[k2,k]*kdummy[k]*kdummy[k]
                NLIntegral2[k]=NLIl1[k1,k]*NLIl1[k2,k]*kdummy[k]*kdummy[k]
            #
            Ncov0[k1,k2] = sp.integrate.simps(NIntegral0,kdummy)
            Lcov0[k1,k2] = sp.integrate.simps(LIntegral0,kdummy)
            NLcov0[k1,k2]= sp.integrate.simps(NLIntegral0,kdummy)
        #
            Ncov1[k1,k2]= sp.integrate.simps(NIntegral1,kdummy)
            Lcov1[k1,k2]= sp.integrate.simps(LIntegral1,kdummy)
            NLcov1[k1,k2]= sp.integrate.simps(NLIntegral1,kdummy)
        #
            Ncov2[k1,k2]= sp.integrate.simps(NIntegral2,kdummy)
            Lcov2[k1,k2]= sp.integrate.simps(LIntegral2,kdummy)
            NLcov2[k1,k2]=sp.integrate.simps(LIntegral2,kdummy)
        #print Ncov0[k1,k1], Ncov1[k1,k1], NLcov2[k1,k1] 
        #
    print "covariance calculation done"
    #
    print "Ncov0 is 00 part of the covariance"
    print "Ncov1 is 01 part of the covariance"
    print "Ncov2 is 11 part of the covariance"
    #
    print "Lcov0 is linear 00 part of the covariance"
    print "Lcov1 is linear 01 part of the covariance"
    print "Lcov2 is linear 11 part of the covariance"
    #
    print "NLcov0 is non*linear 00 part of the covariance"
    print "NLcov1 is non*linear 01 part of the covariance"
    print "NLcov2 is non*linear 11 part of the covariance"
    #
    return kdummy,Ncov0,Lcov0,NLcov0,Ncov1,Lcov1,NLcov1,Ncov2,Lcov2,NLcov2
#
##################################################################
def cls(Ncov0,Lcov0,NLcov0,Ncov1,Lcov1,NLcov1,Lcov2,Ncov2,NLcov2):
##################################################################
#
    Ncls0=np.diag(Ncov0)
    Lcls0=np.diag(Lcov0)
    NLcls0=np.diag(NLcov0)
    Ncls1=np.diag(Ncov1)
    Lcls1=np.diag(Lcov1)
    NLcls1=np.diag(NLcov1)
    Ncls2=np.diag(Ncov2)
    Lcls2=np.diag(Lcov2)
    NLcls2=np.diag(NLcov2)
    #
    #for ik in range(inputs.Nnk):
    #    print Ncls1[ik],Lcls1[ik],NLcls1[ik], Ncls2[ik],Lcls2[ik],NLcls2[ik]
    #
    #exit()
    #
    return Ncls0,Lcls0,NLcls0,Ncls1,Lcls1,NLcls1,Ncls2,Lcls2,NLcls2
#
#############################################
def only_cls(NIl0,LIl0,NLIl0,NIl1,LIl1,NLIl1):
#############################################
#
    NIntegral0= np.empty((inputs.Nnk)); 
    LIntegral0= np.empty((inputs.Nnk));
    NLIntegral0=np.empty((inputs.Nnk));
    #
    NIntegral1= np.empty((inputs.Nnk)); 
    LIntegral1= np.empty((inputs.Nnk));
    NLIntegral1=np.empty((inputs.Nnk));
    #
    Ncls1=np.empty((inputs.Nnk)); 
    Lcls1=np.empty((inputs.Nnk));
    NLcls1=np.empty((inputs.Nnk));
    #
    Ncls0=np.empty((inputs.Nnk)); 
    Lcls0=np.empty((inputs.Nnk));
    NLcls0=np.empty((inputs.Nnk));
    #
    kdummy=logspace(inputs.kmin,inputs.kmax,inputs.Nnk);
    #
    for k1 in range(0,inputs.Nnk):
        print "k1=", k1
        for k in range(0,inputs.Nnk):
            #
            NIntegral0[k]=NIl0[k1,k]*NIl0[k1,k]*kdummy[k]*kdummy[k]
            LIntegral0[k]=LIl0[k1,k]*LIl0[k1,k]*kdummy[k]*kdummy[k]
            NLIntegral0[k]=NLIl0[k1,k]*NLIl0[k1,k]*kdummy[k]*kdummy[k]
                #
            NIntegral1[k]=NIl1[k1,k]*NIl0[k1,k]*kdummy[k]*kdummy[k]
            LIntegral1[k]=LIl1[k1,k]*LIl0[k1,k]*kdummy[k]*kdummy[k]
            NLIntegral1[k]=NLIl1[k1,k]*NLIl0[k1,k]*kdummy[k]*kdummy[k]
            #
            #
        Ncls0[k1] = sp.integrate.simps(NIntegral0,kdummy)
        Lcls0[k1] = sp.integrate.simps(LIntegral0,kdummy)
        NLcls0[k1]= sp.integrate.simps(NLIntegral0,kdummy)
        #
        Ncls1[k1]= sp.integrate.simps(NIntegral1,kdummy)
        Lcls1[k1]= sp.integrate.simps(LIntegral1,kdummy)
        NLcls1[k1]= sp.integrate.simps(NLIntegral1,kdummy)
        #
        #
    print "covariance calculation done"
    #
    return kdummy,Ncls0,Lcls0,NLcls0,Ncls1,Lcls1,NLcls1
#

##########################################
def xi2(nbar,cls,lambda_cls,cov,bin_size):
##########################################
    acov=np.empty((inputs.Nnk,inputs.Nnk));
    xcls=cls[0:inputs.Nnk:bin_size]
    lambda_xcls=lambda_cls[0:inputs.Nnk:bin_size]
    xcls=xcls-lambda_xcls
    for ik in range(inputs.Nnk):
        for jk in range(inputs.Nnk):
            acov[ik,jk]=(cov[ik,jk]+1.0/nbar)**2
    xcov= acov[0:inputs.Nnk:bin_size,0:inputs.Nnk:bin_size]
    print "start inversion"
    invxcov=inv(xcov)
    #
    identity = invxcov*xcov
    print identity
    #
    print "finished inversion"
    chi2= dot(xcls,dot(invxcov,xcls))/2.
    print chi2
    return chi2
#
###########################################################################################
def data_dump_cls(el,r0,kdummy,Ncls0,Lcls0,NLcls0,Ncls1,Lcls1,NLcls1, Ncls2,Lcls2, NLcls2):
###########################################################################################
# 
   a=str(el); b =str(r0)
   kernel_dat=inputs.cls_dir+inputs.cosmo+"_cls_"+"el="+a+"r_0="+b+".dat"
   print kernel_dat
   f = open(kernel_dat, "w")
   for ik in range(0,inputs.Nnk):
       #print kdummy[ik],Ncls0[ik],Lcls0[ik], NLcls0[ik],\
       #                                            Ncls1[ik],Lcls1[ik], NLcls1[ik], \
       #                                            Ncls2[ik],Lcls2[ik], NLcls2[ik]
       f.write("{} {} {} {} {} {} {} {} {} {}\n".format(kdummy[ik],Ncls0[ik],Lcls0[ik], NLcls0[ik],\
                                                   Ncls1[ik],Lcls1[ik], NLcls1[ik], \
                                                   Ncls2[ik],Lcls2[ik], NLcls2[ik]))
   f.close()
   return
###########################################################################################

























#
################
def old_kernels(el):
################ 
    #
    # Computes the four Kernels for integrals.
    #
    print "dn1=", inputs.dir_name1
    print "dn2=", inputs.dir_name2
    print "dn3=", inputs.dir_name3
    #
    print "el=", el
    #
    ### linear fp
    tz,tk,tf,tznew, tknew, tnew_f = xcosmo.dlnD(inputs.dir_name2)
    ### nonlinear fp
    z,k,f,znew, knew, new_f = xcosmo.ndlnD(inputs.dir_name3)
    #### linear and nonlinear ps
    k,knew,pkL, pkN,pknewL, pknewN,growthL,growthN = xcosmo.pkz(inputs.dir_name1)
    ####
    #figure(101)
    #loglog(k[:],pkN[:,0], "b-"); loglog(k[:],pkL[:,0], "b-")
    #loglog(knew[:],pknewL[:,0], "r--"); loglog(knew[:], pknewN[:,0], "r--")
    #show(101)
    ###
    #figure(102)
    #loglog(knew[:], new_f[:,0])
    #loglog(tknew[:], tnew_f[:,0])
    #show(102)
    ###
    print "done"
    ###
    nz=np.size(z); nk=np.size(k)
    Nnz=np.size(znew); Nnk=np.size(knew)
    #
    r= np.empty((Nnz)) # radial comoving
    phi = np.empty((Nnz)) # selction function
    dphi= np.empty((Nnz))
    #
    for i in range(0,Nnz):
        r[i]=quad(xcosmo.rInt,0, znew[i],args=(inputs.M))[0]
        phi[i] = exp(-r[i]**2/inputs.r0**2)
        dphi[i]= (-2*r[i]/inputs.r0**2)*exp(-r[i]**2/inputs.r0**2)
    #
    ka=logspace(inputs.kmin,inputs.kmax,inputs.Nnk); 
    #print ka[::1]
    #print ka[0], ka[25], ka[37], ka[62], ka[74], ka[99]
    #exit()
    #
    NtestI0= np.empty((inputs.Nnz)); 
    LtestI0= np.empty((inputs.Nnz))
    #
    NtestI1= np.empty((inputs.Nnz)); 
    LtestI1= np.empty((inputs.Nnz))
    #
    xa1=np.empty((inputs.Nnk,inputs.Nnz));
    dxa1=np.empty((inputs.Nnk,inputs.Nnz))
    #
    print inputs.kmin, inputs.kmax,inputs.Nnk
    #
    for k1 in range(0,inputs.Nnk):
        for i in range(0,inputs.Nnz):
            argu1=ka[k1]*(r[i]+1.e-8)
            xa1[k1][i]=xcosmo.xsph1(argu1,el)  
            dxa1[k1][i]=xcosmo.dsph1(argu1,el)

    NIl0=np.empty((inputs.Nnk,inputs.Nnk)); LIl0=np.empty((inputs.Nnk,inputs.Nnk))
    NIl1=np.empty((inputs.Nnk,inputs.Nnk)); LIl1=np.empty((inputs.Nnk,inputs.Nnk))
    #
    for k1 in range(0,inputs.Nnk): 
        print k1, inputs.Nnk
        for k2 in range(0,inputs.Nnk):
            for i in range(0,inputs.Nnz):
                #
                temp1= xa1[k1,i]; temp2= xa1[k2,i]
                dtemp1= dxa1[k1,i]; dtemp2= dxa1[k2,i]
                # Computes the 0th window
                xx0=r[i]**2*ka[k1]*temp1*temp2*phi[i]*xcosmo.drdz(znew[i],inputs.M)
                NtestI0[i]= xx0*growthN[k2,i];  LtestI0[i]=xx0*growthL[k2,i]
                # computes the 1st window
                xx1=r[i]**2*xcosmo.drdz(znew[i],inputs.M)*tnew_f[k2,i]
                xx1=xx1*(ka[k1]**2*dtemp1*dtemp2*phi[i]+ka[k1]*temp1*dtemp2*dphi[i]) 
                NtestI1[i]=xx1*growthN[k2,i]; LtestI1[i]=xx1*growthL[k2,i]
                #
            NtestIy=sp.integrate.simps(NtestI0,znew)
            LtestIy=sp.integrate.simps(LtestI0,znew)
            #
            NtestI1y=sp.integrate.simps(NtestI1,znew)
            LtestI1y=sp.integrate.simps(LtestI1,znew)
            #
            NIl0[k1,k2]= NtestIy; LIl0[k1,k2]= LtestIy; 
            NIl1[k1,k2]= NtestI1y; LIl1[k1,k2]= LtestI1y; 
            #
    for k1 in range(0,Nnk):
        for k2 in range(0,Nnk):
            #
            NIl0[k1,k2]=(pknewN[k2,0])**.5*NIl0[k1,k2]
            LIl0[k1,k2]=(pknewL[k2,0])**.5*LIl0[k1,k2]
            #
            NIl1[k1,k2]=(pknewN[k2,0])**.5*NIl1[k1,k2]/ka[k2]           
            LIl1[k1,k2]=(pknewL[k2,0])**.5*LIl1[k1,k2]/ka[k2]
            #
    print; print; print
    return NIl0,LIl0,NIl1,LIl1
#
##################################
def old_cls_cov(NIl0,LIl0,NIl1,LIl1):
##################################
    #
    NIntegral0= np.empty((inputs.Nnk)); LIntegral0= np.empty((inputs.Nnk))
    NIntegral1= np.empty((inputs.Nnk)); LIntegral1= np.empty((inputs.Nnk))
    NIntegral2= np.empty((inputs.Nnk)); LIntegral2= np.empty((inputs.Nnk))
    #
    Ndia01=np.empty((inputs.Nnk,inputs.Nnk)); Ldia01=np.empty((inputs.Nnk,inputs.Nnk))
    Ndia00=np.empty((inputs.Nnk,inputs.Nnk)); Ldia00=np.empty((inputs.Nnk,inputs.Nnk))
    Ndia11=np.empty((inputs.Nnk,inputs.Nnk)); Ldia11=np.empty((inputs.Nnk,inputs.Nnk))
    #
    kdummy=logspace(inputs.kmin,inputs.kmax,inputs.Nnk);
    for k1 in range(0,inputs.Nnk):
        print "covariance", k1
        for k2 in range(0,inputs.Nnk):
            for k in range(0,inputs.Nnk):
            #
                NIntegral0[k]=NIl0[k1][k]*NIl0[k2][k]*kdummy[k]*kdummy[k]
                LIntegral0[k]=LIl0[k1][k]*LIl0[k2][k]*kdummy[k]*kdummy[k]
            #
                NIntegral1[k]=NIl1[k1][k]*NIl0[k2][k]*kdummy[k]*kdummy[k]
                LIntegral1[k]=LIl1[k1][k]*LIl0[k2][k]*kdummy[k]*kdummy[k]
            #
                NIntegral2[k]=NIl1[k1][k]*NIl1[k2][k]*kdummy[k]*kdummy[k]
                LIntegral2[k]=LIl1[k1][k]*LIl1[k2][k]*kdummy[k]*kdummy[k]
            #
            Ndia00[k1,k2] = sp.integrate.simps(NIntegral0,kdummy)
            Ldia00[k1,k2] = sp.integrate.simps(LIntegral0,kdummy)
        #
            Ndia01[k1,k2] = sp.integrate.simps(NIntegral1,kdummy)
            Ldia01[k1,k2] = sp.integrate.simps(LIntegral1,kdummy)
        #
            Ndia11[k1,k2] = sp.integrate.simps(NIntegral2,kdummy)
            Ldia11[k1,k2] = sp.integrate.simps(LIntegral2,kdummy)
    return Ndia00,Ldia00,Ndia01,Ldia01,Ndia11,Ldia11


#
###################################
def old_cls3D(NIl0,LIl0,NIl1,LIl1):
###################################x
#
    NIntegral0= np.empty((inputs.Nnk)); LIntegral0= np.empty((inputs.Nnk))
    NIntegral1= np.empty((inputs.Nnk)); LIntegral1= np.empty((inputs.Nnk))
    NIntegral2= np.empty((inputs.Nnk)); LIntegral2= np.empty((inputs.Nnk))
    #
    Ndia1=np.empty((inputs.Nnk)); Ldia1=np.empty((inputs.Nnk))
    Ndia0=np.empty((inputs.Nnk)); Ldia0=np.empty((inputs.Nnk))
    Ndia2=np.empty((inputs.Nnk)); Ldia2=np.empty((inputs.Nnk))
    #
    kdummy=logspace(inputs.kmin,inputs.kmax,inputs.Nnk);
    for k1 in range(0,inputs.Nnk):
        for k in range(0,inputs.Nnk):
            #
            NIntegral0[k]=NIl0[k1][k]*NIl0[k1][k]*kdummy[k]*kdummy[k]
            LIntegral0[k]=LIl0[k1][k]*LIl0[k1][k]*kdummy[k]*kdummy[k]
            #
            NIntegral1[k]=NIl1[k1][k]*NIl0[k1][k]*kdummy[k]*kdummy[k]
            LIntegral1[k]=LIl1[k1][k]*LIl0[k1][k]*kdummy[k]*kdummy[k]
            #
            NIntegral2[k]=NIl1[k1][k]*NIl1[k1][k]*kdummy[k]*kdummy[k]
            LIntegral2[k]=LIl1[k1][k]*LIl1[k1][k]*kdummy[k]*kdummy[k]
            #
        Ndia0[k1] = sp.integrate.simps(NIntegral0,kdummy)
        Ldia0[k1] = sp.integrate.simps(LIntegral0,kdummy)
        #
        Ndia1[k1] = sp.integrate.simps(NIntegral1,kdummy)
        Ldia1[k1] = sp.integrate.simps(LIntegral1,kdummy)
        #
        Ndia2[k1] = sp.integrate.simps(NIntegral2,kdummy)
        Ldia2[k1] = sp.integrate.simps(LIntegral2,kdummy)
        #
    return kdummy,Ndia0,Ldia0, Ndia1, Ldia1, Ndia2, Ldia2
#
