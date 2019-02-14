# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 23:06:20 2019

@author: signalprocessing
"""

import numpy as np
import matplotlib.pyplot as plt

r_process=np.random.RandomState()

Nsample=3000

Nscale_gen=1
Nfft_base=64
Ndelay=Nfft_base*Nscale_gen
Nfft=Nfft_base*Nscale_gen
Nds=12
#PDP=np.concatenate((np.arange(2*Nscale_gen)*2/Nscale_gen,25-np.arange(Nds*Nscale_gen)*2/Nscale_gen,-2*np.ones((Ndelay-(Nds+2)*Nscale_gen))),axis=0)
PDP=np.concatenate((np.arange(2*Nscale_gen)*2/Nscale_gen,25-np.arange(Nds*Nscale_gen)*2/Nscale_gen,-800*np.ones((Ndelay-(Nds+2)*Nscale_gen))),axis=0)

fig=plt.figure(1)
plt.plot(PDP)
plt.plot(PDP,'o')
CIRs=(r_process.randn(Nsample,Ndelay)+1j*r_process.randn(Nsample,Ndelay))*np.power(10,PDP/20)
CFRs=np.fft.fft(CIRs)

Cut_idx=[0,24,52,64]
Nband=len(Cut_idx)-1

# cut full band CFR according to Cut_idx
CFRs_perBand=[0]*Nband
Idxs_perBand=[0]*Nband
FFTbase=np.fft.fft(np.eye(Nfft))
for iiBand in range(Nband):
    Idxs_perBand[iiBand]=range(Cut_idx[iiBand],Cut_idx[iiBand+1])
    CFRs_perBand[iiBand]=CFRs[:,Idxs_perBand[iiBand]]
    
    print('sanity check 0:',np.sum(np.abs(CFRs_perBand[iiBand]- np.matmul(FFTbase[Idxs_perBand[iiBand],:],CIRs.T).T)))
    
plt.figure()
#plt.hold()
Nds2=Nds+10
prod=np.zeros((Nband-1,Nsample,1,1),dtype=complex)
for iiBand in range(Nband-1):#[0]:#
    Ny1=len(Idxs_perBand[iiBand+1])
    Ny0=len(Idxs_perBand[iiBand])
    Y1_ct=CFRs_perBand[iiBand+1].reshape(Nsample,1,Ny1).conj()
    Y0=CFRs_perBand[iiBand].reshape(Nsample,Ny0,1)
    M12=np.matmul(FFTbase[Idxs_perBand[iiBand],:Nds2],FFTbase[Idxs_perBand[iiBand+1],:Nds2].conj().T)
    u,s,vh=np.linalg.svd(M12)
    plt.plot(s,label='Band'+str(iiBand))
    plt.yscale('log')
    M12_inv=np.linalg.pinv(M12)
    tmp_diag=np.zeros((Ny1,Ny0))
    Ny_min=np.minimum(Ny0,Ny1)
    tmp_diag[:Ny_min,:Ny_min]=np.diag(1/(s+0.1))
    #M12_inv=np.matmul(np.matmul(vh.conj().T,tmp_diag),u.conj().T)
    if 1:
        a=np.matmul(M12_inv,M12)
        print('sanity check M12_inv 0:',np.max(np.abs(a- np.eye(a.shape[0]))))
 
    prod[iiBand,:,:,:]=np.matmul(np.matmul(Y1_ct,M12_inv),Y0)
plt.legend()

if 1:    
    prod2=prod.reshape(Nband-1,Nsample)
    a=np.angle(prod2)
    a.sort()
    
    plt.figure()
    for iiBand in range(Nband-1):
        plt.plot(a[iiBand,:],label='Band'+str(iiBand))
    
