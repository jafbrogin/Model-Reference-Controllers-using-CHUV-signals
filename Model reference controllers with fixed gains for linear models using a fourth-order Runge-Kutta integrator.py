#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:42:27 2023

@author: joao

"""

import numpy as np
import matplotlib.pyplot as plt
# Please change the name of 'LMIs for designing a controller to linear state-space models' to 'LMI_seizures_CHUV_simple'
from LMI_seizures_CHUV_simple import Ah, As, Bh, Bs, X0h, X0s, states_h, states_s, Gh, Gs, Gg

# %% Parameters of simulation:
Fs = 256
dt = 1/Fs
nd = len(Ah)
N = len(states_h[0, :])

X_h = np.zeros((nd, N))
uf_h = np.zeros((nd, N))

X_s = np.zeros((nd, N))
uf_s = np.zeros((nd, N))
uf_a = np.zeros((nd, N))

t = np.linspace(0, N*dt, N)

from scipy import signal
sqr = signal.square(0.05 * np.pi * 5 * t)
sqr[sqr <= 0] = 0
sqr = sqr*0.4

# %% Set of differential equations:
def __dXdt__(Xdh1, statesh1, Xds1, statess1, switch_g, it):
    switch_h = 1
    switch_s = 1 
    
    # Healthy (switch_h = 1 --> tracks non-seizure activity):
    Xdh2 = Xdh1-statesh1 
    Xdh2 = np.reshape(Xdh2, (-1, 1))
    auxh = Ah.dot(Xdh1.T) 
    UTh = -switch_h*Gh.dot(Xdh2)
    wuh = Bh.dot(UTh)
    solh = np.reshape(auxh, (-1,)) + np.reshape(wuh, (-1,)) 

    # Seizure (switch_s = 1 --> tracks seizure; seizure_g = 1 --> controls seizure):
    Xds2 = Xds1-statess1 
    Xds3 = Xds1-Xdh1
    Xds2 = np.reshape(Xds2, (-1, 1))
    Xds3 = np.reshape(Xds3, (-1, 1))
    auxs = As.dot(Xds1.T) 
    UTs = -switch_s*Gs.dot(Xds2)
    UTa = -switch_g*Gg.dot(Xds3)
    wus = Bs.dot(UTs)
    wua = Bh.dot(UTa)
    sols = np.reshape(auxs, (-1,)) + np.reshape(wus, (-1,)) + np.reshape(wua, (-1,)) 

    return solh, wuh, sols, wus, wua

# %% RK4:
X_h[:, 0] = list(X0h)
X_s[:, 0] = list(X0s)

switch = 1   # control (seizure --> non-seizure) 
Etest = []
for k in range(0, N - 1):
    print('Iteração número: ' + str(k))
    k1, u1, kk1, uu1, ua1 = __dXdt__(X_h[:, k], states_h[:, k], X_s[:, k], states_s[:, k], switch, k)
    k2, u2, kk2, uu2, ua2 = __dXdt__(X_h[:, k] + k1*(dt/2), states_h[:, k], X_s[:, k] + kk1*(dt/2), states_s[:, k], switch, k)
    k3, u3, kk3, uu3, ua3 = __dXdt__(X_h[:, k] + k2*(dt/2), states_h[:, k], X_s[:, k] + kk2*(dt/2), states_s[:, k], switch, k)
    k4, u4, kk4, uu4, ua4 = __dXdt__(X_h[:, k] + k3*dt, states_h[:, k], X_s[:, k] + kk3*dt, states_s[:, k], switch, k)

    # States:
    X_h[:, k+1] = X_h[:, k] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    X_s[:, k+1] = X_s[:, k] + (dt/6)*(kk1 + 2*kk2 + 2*kk3 + kk4)
    
    # Input forces:
    uf_h[:, k+1] = np.array(u4.T)
    uf_s[:, k+1] = np.array(uu4.T)
    uf_a[:, k+1] = np.array(ua4.T)
    
    # if k >= 0:
    #     switch = sqr[k]

# %% Plots:
Eh = (states_h[0] - X_h[0, :]) / (max(states_h[0]) - min(states_h[0]))
Es = (states_s[0] - X_s[0, :]) / (max(states_s[0]) - min(states_s[0]))
    
plt.figure(1)
plt.subplot(411)
plt.plot(t, states_h[0], 'k',linewidth=1)
plt.plot(t, X_h[0, :], 'b--',linewidth=2)  
plt.grid()
plt.ylabel('$x^{{\{h\}}}$ $[\mu V]$', fontsize=15)
plt.xlim(t[0],t[-1])

plt.subplot(412)
plt.plot(t,Eh,'k',linewidth=1)
plt.ylim(-1,1)
plt.grid()
plt.ylabel('$e(t)^{{\{h\}}}$ $[\%]$', fontsize=15)
plt.xlim(t[0],t[-1])

plt.subplot(413)
plt.plot(t,states_s[0],'k',linewidth=1)
plt.plot(t,X_s[0, :], 'r--',linewidth=2)
plt.grid()
plt.ylabel('$x^{{\{s\}}}$ $[\mu V]$', fontsize=15)
plt.xlim(t[0],t[-1])

plt.subplot(414)
plt.plot(t,Es,'k',linewidth=1)
plt.ylim(-1,1)
plt.ylim(-1,1)
plt.grid()  
plt.ylabel('$e(t)^{{\{s\}}}$ $[\%]$', fontsize=15)
plt.xlabel('$t$ $[s]$', fontsize=15, rotation=0)
plt.xlim(t[0],t[-1])
    
plt.figure(2)
# plt.plot( np.sqrt(uf_a[1]**2 + uf_a[0]**2)/1e6, 'r--')
plt.plot(t, uf_a[1]/1e6, 'k', linewidth=1)
plt.grid()
plt.ylabel('$u^{{\{ad\}}}$ $[\mu V]$', fontsize=15)
plt.xlabel('$t$ $[s]$', fontsize=15, rotation=0)
plt.xlim(t[0],t[-1])

adherence = ( 1 - max(abs(Es)) )*100
mean_error = np.mean(abs(Es))
input_force = max(np.sqrt(uf_a[1]**2 + uf_a[0]**2)/1e6)

att = []
n_wind = 1  
wind = len(states_s[0,:])//n_wind

for jj in range(0,n_wind):
    rmsh = np.sqrt( np.mean(states_h[0,  jj*wind : (jj+1)*wind ]**2) )
    rms1 = np.sqrt( np.mean(X_s[0,  jj*wind : (jj+1)*wind ]**2) )
    rms2 = np.sqrt( np.mean(states_s[0][jj*wind : (jj+1)*wind]**2) )
    
    Is = (rms2 - rmsh)/rmsh
    Ia = (rms1 - rmsh)/rmsh
    I_f = (Is - Ia)/Is
    attenuation = (I_f)*100
    att.append(attenuation)
    # print('Attenuation of: ' + str( round(attenuation,4) ) + '%' ) 

mean_att = np.mean(np.array(att))
print('Attenuation of: ' + str( round(mean_att,4) ) + '%' )


#%% Plot 3D: analysis of adherence x attenuation x input force
fig = plt.figure()
ax = plt.axes(projection='3d')

alfa = [5,10,15,20,25,30,35,40,45,50,55]

me = [0.04661656492196227,0.04363880349600848,0.039528229974944,0.028507995652455695,
      0.03271089128963722,0.0293221374099187,0.02621683632879001,0.025221785710461596,
      0.02291655629593286,0.021027078486391137,0.021117697416503545]

adh = [45.805148332217996,49.40433438776343,54.85879889096823,69.76759826026101,
       64.27941054333795,68.2665463284709,72.1491915087963,72.98552085773535,
       74.33958704575059,75.5529495709219,75.32497725131442]

at = [99.18060848793087,99.6823330021721,97.53319360292886,75.86659170845384,
      87.1807998755501,78.79830426505012,69.1818543479646,65.87599550435812,
      57.11866545776435,48.89142483181031,49.747243232901646]

inp = [2.7477961609335906,3.8729236282140804, 6.623179027773752,13.908186169348635,
       9.341913133501953,11.46113524887098,13.689495460066441,15.35686750058467,
       17.977706962148986,19.498370631964857,22.506042653170713]

ax.plot3D(inp, adh, at, 'ko-')

ax.set_ylabel(r'$\gamma$ $[\%]$', fontsize=15, rotation=0)
ax.set_zlabel(r'$\Delta V$ $[\%]$', fontsize=15, rotation=0)
ax.set_xlabel(r'$max(\parallel \textbf{u} \parallel)$ $[V]$', fontsize=15, rotation=0)

# plt.savefig('seizure_ctrl_params_3d_adap.pdf')

plt.figure()
# plt.plot(alfa,np.array(me)*1000,'ko', linewidth=2, label='$mean|e|$ $[\%]10^{-1}$')
# plt.plot(alfa,np.array(me)*1000,'k', linewidth=1, label='$mean|e|$ $[\%]10^{-1}$')
plt.plot(alfa,adh,'bv', linewidth=3, label='$\gamma$ $[\%]$')
plt.plot(alfa,adh,'b', linewidth=1)
plt.plot(alfa,at,'r*', linewidth=5,label='$\Delta V$ $[\%]$')
plt.plot(alfa,at,'r', linewidth=1)
plt.plot(alfa,inp,'ms', linewidth=3,label='$max(u^{{\{sh\}}})$ $[V]$')
plt.plot(alfa,inp,'m', linewidth=1)
plt.legend(loc=1,fontsize=15)
plt.grid()
plt.xlim(5,55)
plt.ylim(0,200)
plt.xlabel(r'$\alpha^{{\{s\}}}$', fontsize=20, rotation=0)
