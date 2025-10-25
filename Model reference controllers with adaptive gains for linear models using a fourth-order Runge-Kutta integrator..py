#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:42:27 2023

@author: joao

"""

import numpy as np
import matplotlib.pyplot as plt
# Please change the name of 'LMIs for designing a controller to linear state-space models' to 'LMI_seizures_CHUV_simple'
from LMI_seizures_CHUV_simple import Ah, As, Bh, Bs, X0h, X0s, states_h, states_s, Gh, Gs, Pg

# %% Parâmetros de simulação:
Fs = 256
dt = 1/Fs
nd = len(Ah)
N = len(states_h[0, :])

X_h = np.zeros((nd, N))
uf_h = np.zeros((nd, N))

X_s = np.zeros((nd, N))
uf_s = np.zeros((nd, N))
uf_a = np.zeros((nd, N))

E = np.zeros((nd, N))
GG = np.zeros((nd//2, nd, N))
LL = np.zeros((nd//2, nd, N))

t = np.linspace(0, N*dt, N)

# from scipy import signal
# sqr = signal.square(0.05 * np.pi * 5 * t)
# sqr[sqr <= 0] = 0
# sqr = sqr*0.2

# %% Conjunto de equações diferenciais:
def __dXdt__(Xdh1, statesh1, Xds1, statess1, gain_G, gain_L, switch_g, error):
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
    Xds3 = Xds1
    Xds4 = Xdh1

    Xds2 = np.reshape(Xds2, (-1, 1))
    Xds3 = np.reshape(Xds3, (-1, 1))
    Xds4 = np.reshape(Xds4, (-1, 1))
    EE = np.reshape(error,(-1,1))
    PP = np.array(Pg)
    
    auxs = As.dot(Xds1.T)     
    UTs = -switch_s*Gs.dot(Xds2)
    wus = Bs.dot(UTs)
    
    UTa1 = -switch_g * gain_G.dot(Xds3)
    UTa2 = +switch_g * gain_L.dot(Xds4) 
    wua = Bh.dot(UTa1) + Bh.dot(UTa2)
    
    sols = np.reshape(auxs, (-1,)) + np.reshape(wus, (-1,)) + np.reshape(wua, (-1,)) 
    
    pp1 = Bh.T.dot(PP)
    pp2 = pp1.dot(EE)
    pp3 = pp2.dot(Xds3.T)
    
    ppp1 = Bh.T.dot(PP)
    ppp2 = ppp1.dot(EE)
    ppp3 = ppp2.dot(Xds4.T)
        
    G_up = +1*pp3
    L_up = -1*ppp3

    return solh, wuh, sols, wus, wua, G_up, L_up

# %% Runge-Kutta de 4a ordem:
X_h[:, 0] = list(X0h)
X_s[:, 0] = list(X0s)

switch = 0.7
Etest = []
for k in range(0, N - 1):
    print('Iteração número: ' + str(k))
    k1, u1, kk1, uu1, ua1, g1, l1 = __dXdt__(X_h[:, k], states_h[:, k], X_s[:, k], states_s[:, k], 
                                     GG[:,:,k], LL[:,:,k], switch, E[:,k])
    k2, u2, kk2, uu2, ua2, g2, l2 = __dXdt__(X_h[:, k] + k1*(dt/2), states_h[:, k], X_s[:, k] + kk1*(dt/2), states_s[:, k], 
                                     GG[:,:,k] + g1*(dt/2), LL[:,:,k] + l1*(dt/2), switch, E[:,k])
    k3, u3, kk3, uu3, ua3, g3, l3 = __dXdt__(X_h[:, k] + k2*(dt/2), states_h[:, k], X_s[:, k] + kk2*(dt/2), states_s[:, k], 
                                     GG[:,:,k] + g2*(dt/2), LL[:,:,k] + l2*(dt/2), switch, E[:,k])
    k4, u4, kk4, uu4, ua4, g4, l4 = __dXdt__(X_h[:, k] + k3*dt, states_h[:, k], X_s[:, k] + kk3*dt, states_s[:, k], 
                                     GG[:,:,k] + g3*dt, LL[:,:,k] + l3*dt, switch, E[:,k])

    # States:
    X_h[:, k+1] = X_h[:, k] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    X_s[:, k+1] = X_s[:, k] + (dt/6)*(kk1 + 2*kk2 + 2*kk3 + kk4)
    
    # Error:
    E[:, k + 1] = X_s[:,k + 1] - X_h[:, k + 1]
    GG[:,:,k + 1] = GG[:,:,k] + (dt/6)*(g1 + 2*g2 + 2*g3 + g4)
    LL[:,:,k + 1] = LL[:,:,k] + (dt/6)*(l1 + 2*l2 + 2*l3 + l4)
    
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
plt.ylabel('$u$ $[V]$', fontsize=15)
plt.xlabel('$t$ $[s]$', fontsize=15, rotation=0)
plt.xlim(t[0],t[-1])

plt.figure(3)
plt.subplot(411)
plt.plot(t,GG[0][0,:],'m', linewidth=2)
plt.grid()
plt.ylabel('$G_{1}^{{\{ad,1\}}}$', fontsize=15)
plt.xlim(t[0],t[-1])

plt.subplot(412)
plt.plot(t,GG[0][1,:],'m', linewidth=2)
plt.grid()
plt.ylabel('$G_{2}^{{\{ad,1\}}}$', fontsize=15)
plt.xlim(t[0],t[-1])

plt.subplot(413)
plt.plot(t,LL[0][0,:],'m', linewidth=2)
plt.grid()
plt.ylabel('$L^{{\{ad,1\}}}$', fontsize=15)
plt.xlim(t[0],t[-1])

plt.subplot(414)
plt.plot(t,LL[0][1,:],'m', linewidth=2)
plt.grid()
plt.ylabel('$L^{{\{ad,2\}}}$', fontsize=15)
plt.xlabel('$t$ $[s]$', fontsize=15, rotation=0)
plt.xlim(t[0],t[-1])


adherence = ( 1 - max(abs(Es)) )*100
mean_error = np.mean(abs(Es))
input_force = max(np.sqrt(uf_a[1]**2 + uf_a[0]**2)/1e6)

att = []
n_wind = 10
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

me = [0.034305977549770446,0.031558852957557254,0.029391732053818686,0.027736336887858263,
        0.026033457642204393,0.024980170605260518,0.0240277905377313,0.02295711263173461,
        0.021809871108380237,0.02067336022870103,0.01959155514885731]

adh = [57.49545588370817,53.827145636283966,53.13925166085076,55.135897737048346,
        59.353495676237465,62.58121850500278,65.46061880424227,68.29127532712847,
        70.84442314182404,72.99403047942292,74.72082017230836]

at = [51.465984731683406,50.99654626244897,43.652276278247584,37.56932339913972,
        30.852785314858465,27.97098103470255,26.57030255927703,24.40893051446097,
        21.860386166734454,19.476601922383132,17.32137928951805]

inp = [1.6954713560826091,7.633939355116413,18.16541089142942,34.06606938044046,
        47.285462980867955,69.98622697422091,104.27982233974409,129.2855748179509,
        147.61888528754199,163.72161795051798,179.06326411472543]

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
plt.legend(loc=2,fontsize=15)
plt.grid()
plt.xlabel(r'$\alpha^{{\{s\}}}$', fontsize=20, rotation=0)
plt.xlim(min(alfa),max(alfa))
plt.ylim(0,200)

# plt.savefig('seizure_ctrl_params_2d_adap.pdf')
