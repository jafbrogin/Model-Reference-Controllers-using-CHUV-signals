#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 08:21:25 2023

@author: joao
"""

import os
import re
import numpy as np
import pandas as pd
from matplotlib import rc
from numpy.linalg import matrix_power
import matplotlib.pyplot as plt

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

#%% Information about source, sampling frequency and channels:
f_DNK = 1024
f_ZCH = 256

place = 'ZCH'
no_pt = '14'
seizure_ID = 1
Fs = f_ZCH

patient_ID = "SEV-" + place + "-" + no_pt
seizure_data = pd.read_parquet(f"{patient_ID}_seizure_{seizure_ID}.parquet")
channels = list(seizure_data.columns)

#%% Import signals:
sig_vec = []
nc = len(channels)    

# Measurable activity:
for ww in range(0,nc):
    data_folder = os.getcwd() + "/SEV-" + place + "-" + no_pt + '_seizure_' + str(seizure_ID) + '/'
    
    if '+' not in channels[ww] and '-' not in channels[ww] and 'Photic' not in channels[ww]:
        filename = r'' + data_folder + "SEV-" + place + "-" + no_pt + '_seizure_' + str(seizure_ID) + '_' + channels[ww] + '.txt'
        fsig = open(filename, 'r')
        signal_aux = re.findall(r"\S+", fsig.read())
        fsig.close()
    
        sig = np.zeros(len(signal_aux))
        for jj in range(0, len(signal_aux)):
            sig[jj] = round(float(signal_aux[jj]),14)

        sig_vec.append(sig)
    
#%% Measurable activity in state-space notation:
window = 17664           # 03: 26368 + 2k, 04: 17408, 14: 17664, 16: 18688, 19: 20992, 23: 15616
start_window = 921600     # 03: 921600, 04: 921600, 14: 921600, 16:  616704, 19: 921600, 23: 921600
end_window   = start_window + window
    
dof = 1#len(sig_vec) - 22      # degrees of freedom = number of channels considered (only x)
nd =  dof*2                  # number of states = number of channels considered * 2 (x and dot x)
Xsol = np.zeros((nd,window))
Xsol_aux = np.zeros((nd,window))    

for ww in range(0,dof):    
    # Definition of states:
    Xsol[ww] = sig_vec[ww][ 0 : window ]
    Xsol_aux[ww] = sig_vec[ww][ start_window : end_window ]
    
    aux_1 = Xsol[ww]  
    aux_11 = Xsol_aux[ww]
    
    aux_2 = np.diff(aux_1,prepend=aux_1[0]) * f_ZCH
    aux_22 = np.diff(aux_11,prepend=aux_11[0]) * f_ZCH 
    
    Xsol[ww + dof] = aux_2
    Xsol_aux[ww + dof] = aux_22

# %% Operational matrix of integration:
t = np.linspace(0,window/f_ZCH,window)
T = window / f_ZCH
s = 100
r = 2*s + 1 
aux = []

for nn in range(0, s+1):
    phin = np.cos(2*np.pi*t*nn/T)
    aux.append(phin)

for mm in range(1, s+1):
    phi_ = np.sin(2*np.pi*t*mm/T)
    aux.append(phi_)

PHI = np.vstack(aux)

c0 = np.reshape(T/2, (-1, 1))
Z1 = np.zeros((1, s))
ZZ = np.zeros((s, s))
II = np.identity(s)
E = np.reshape(1 / np.linspace(1, s, s), (-1, 1))
Iss = II * E

L1 = [           c0,                 Z1,         -T*E.T/np.pi]
L2 = [         Z1.T,                 ZZ,      T*Iss/(2*np.pi)]
L3 = [T*E/(2*np.pi),   -T*Iss/(2*np.pi),                   ZZ]

H1 = np.hstack(L1)
H2 = np.hstack(L2)
H3 = np.hstack(L3)

P = np.vstack((H1, H2, H3))
e = np.zeros((r, 1))
e[0] = 1

# %% Fourier series implementation:
def _Fourier_Series_Expansion_(y, terms):
    a0 = (1/T) * np.sum(y) / f_ZCH
    an = []
    bn = []
    sum_ = 0
    for n in range(1, terms + 1):
        aux1 = (2/T) * np.sum(y*np.cos(2.*np.pi*n*t/T)) / f_ZCH
        an.append(aux1)
        aux2 = (2/T) * np.sum(y*np.sin(2.*np.pi*n*t/T)) / f_ZCH
        bn.append(aux2)
        sum_ = sum_ + aux1*np.cos(2.*np.pi*n*t/T) + aux2*np.sin(2.*np.pi*n*t/T)

    sf = a0 + sum_
    F_coefs = np.hstack((a0, an, bn))
    return sf, F_coefs

# %% Identificação:
u_input = np.zeros((nd//2, window))

for ll in range(0, nd//2):
    # u_input[ll] = np.sin(2*(1 + ll)*np.pi*t)
    u_input[ll] = np.random.normal(0,1,len(t))

# %% Expansion of time series:   
terms = s
X_series_h = []
X_coefs_h = np.zeros((nd, r))

X_series_s = []
X_coefs_s = np.zeros((nd, r))

# Expansion of states (healthy):
for hh in range(0, nd):
    aux1, aux2 = _Fourier_Series_Expansion_(Xsol[hh, :], terms)
    aux2 = np.reshape(aux2, (-1, 1)).T
    X_series_h.append(aux1)
    X_coefs_h[hh] = aux2
    
# Expansion of states (seizure):
for hh in range(0, nd):
    aux11, aux22 = _Fourier_Series_Expansion_(Xsol_aux[hh, :], terms)
    aux22 = np.reshape(aux22, (-1, 1)).T
    X_series_s.append(aux11)
    X_coefs_s[hh] = aux22

# Expandion of all input forces:
U_series = []
U_coefs = np.zeros((nd//2, r))

for gg in range(0, nd//2):
    aux11, aux22 = _Fourier_Series_Expansion_(u_input[gg], terms)
    aux22 = np.reshape(aux22, (-1, 1)).T
    U_series.append(aux11)
    U_coefs[gg] = aux22

P2 = matrix_power(P, 2)

#%% Visualize original vs reconstructed signal:
from sklearn.metrics import mean_squared_error

nmse_h = 100 * mean_squared_error(Xsol[0,:],X_series_h[0] ) / ( max(Xsol[0,:]) - min(Xsol[0,:]) )
nmse_s = 100 * mean_squared_error(Xsol_aux[0,:],X_series_s[0] ) / ( max(Xsol_aux[0,:]) - min(Xsol_aux[0,:]) )

print('NMSE (healthy): ' + str(nmse_h) )
print('NMSE (seizure): ' + str(nmse_s) )

# %% State-space formulation to obtain matrices -K/M, -C/M and B0/M:
# Healthy:
Xb_h = X_coefs_h
Q_h = np.vstack((e.T, Xb_h.dot(P))) # Without B
# Q_h = np.vstack((e.T, Xb_h.dot(P), U_coefs.dot(P))) # With B

QQ_h = Q_h.dot(Q_h.T)
QQinv_h = np.linalg.inv(QQ_h)
QQQ_h = QQinv_h.dot(Q_h)
theta_h = QQQ_h.dot(Xb_h.T)

# Estimated matrices/parameters (with B):
# A_est_h = theta_h.T[0:nd, 1:-nd//2]
# B_est_h = theta_h.T[:, -nd//2:]
# X0_est_h = theta_h.T[:, 0]

# Z_id_h = A_est_h[0:nd//2, 0:nd//2]
# I_id_h = A_est_h[0:nd//2, nd//2:]
# KMinv_id_h = A_est_h[nd//2:, 0:nd//2]
# CMinv_id_h = A_est_h[nd//2:, nd//2:]

# Estimated matrices/parameters (without B):
A_est_h = theta_h.T[0:nd, 1:]
X0_est_h = theta_h.T[:, 0]

Z_id_h = A_est_h[0:nd//2, 0:nd//2]
I_id_h = A_est_h[0:nd//2, nd//2:]
KMinv_id_h = A_est_h[nd//2:, 0:nd//2]
CMinv_id_h = A_est_h[nd//2:, nd//2:]


###############################################################################


# Seizure:
Xb_s = X_coefs_s
Q_s = np.vstack((e.T, Xb_s.dot(P))) # Without B
# Q_s = np.vstack((e.T, Xb_s.dot(P), U_coefs.dot(P))) # With B

QQ_s = Q_s.dot(Q_s.T)
QQinv_s = np.linalg.inv(QQ_s)
QQQ_s = QQinv_s.dot(Q_s)
theta_s = QQQ_s.dot(Xb_s.T)

# Estimated matrices/parameters (with B):
# A_est_s = theta_s.T[0:nd, 1:-nd//2]
# B_est_s = theta_s.T[:, -nd//2:]
# X0_est_s = theta_s.T[:, 0]

# Z_id_s = A_est_s[0:nd//2, 0:nd//2]
# I_id_s = A_est_s[0:nd//2, nd//2:]
# KMinv_id_s = A_est_s[nd//2:, 0:nd//2]
# CMinv_id_s = A_est_s[nd//2:, nd//2:]

# Estimated matrices/parameters (without B):
A_est_s = theta_s.T[0:nd, 1:]
X0_est_s = theta_s.T[:, 0]

Z_id_s = A_est_s[0:nd//2, 0:nd//2]
I_id_s = A_est_s[0:nd//2, nd//2:]
KMinv_id_s = A_est_s[nd//2:, 0:nd//2]
CMinv_id_s = A_est_s[nd//2:, nd//2:]