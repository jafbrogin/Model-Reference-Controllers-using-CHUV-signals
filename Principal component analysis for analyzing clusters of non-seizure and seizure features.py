#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 08:56:38 2023

@author: joao
"""

import os
import re
import numpy as np
import pandas as pd
from matplotlib import rc
import matplotlib.pyplot as plt

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

#%% PCA:
ZCH_patients = ['03','04','14','16','19','23']
seizures     = ['01','02','03','04','05']

lp = len(ZCH_patients)
ls = len(seizures)

max_it = 20
n_feat = 100
ref_NS = np.zeros(( max_it*lp*ls, n_feat ))
ref_S  = np.zeros(( max_it*lp*ls, n_feat ))


for outer_loop in range(0,max_it):

    counter_NS = 0
    for ii in range(0,lp):
        for jj in range(0,ls):
            filename_NS = 'patient_' + ZCH_patients[ii] + '_seizure_' + seizures[jj] + '_NS.txt' 
            try:
                file_name_NS = open(filename_NS, 'r')
                strings_NS = re.findall(r"\S+", file_name_NS.read()) 
                file_name_NS.close()
            
                features_NS = np.zeros(len(strings_NS))
                for kk in range(0, len(strings_NS)):
                    features_NS[kk] = round(float(strings_NS[kk]),14)
            
                ref_NS[ ls*lp*outer_loop + counter_NS] = features_NS[ outer_loop*n_feat : (outer_loop+1)*n_feat ]
            except:
                pass
    
            counter_NS = counter_NS + 1

    counter_S = 0
    for ii in range(0,lp):
        for jj in range(0,ls):
            filename_S = 'patient_' + ZCH_patients[ii] + '_seizure_' + seizures[jj] + '_S.txt'
            try:
                file_name_S = open(filename_S, 'r')
                strings_S = re.findall(r"\S+", file_name_S.read()) 
                file_name_S.close()
            
                features_S = np.zeros(len(strings_S))
                for kk in range(0, len(strings_S)):
                    features_S[kk] = round(float(strings_S[kk]),14)
            
                ref_S[ ls*lp*outer_loop + counter_S] = features_S[ outer_loop*n_feat : (outer_loop+1)*n_feat ]
            
            except:
                pass
        
            counter_S = counter_S + 1
     
        
     
ref_NS = ref_NS[~np.all(ref_NS == 0, axis=1)]
ref_S = ref_S[~np.all(ref_S == 0, axis=1)]

#%% PCA
signals = np.vstack( (ref_NS,ref_S) )

cvt = np.cov(signals.T)
eig_val, eig_vec = np.linalg.eig(cvt)
idx = eig_val.argsort()[::-1]
eig_val = eig_val[idx]
eig_vec = eig_vec[:,idx]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('$PC_{1}$', fontsize=20)
ax.set_ylabel('$PC_{2}$', fontsize=20)
ax.set_zlabel('$PC_{3}$', fontsize=20)
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.zaxis.set_tick_params(labelsize=10)

# Ref NS:
PCA1_test1 = np.real( eig_vec[:,0].dot(ref_NS.T) )
PCA2_test1 = np.real( eig_vec[:,1].dot(ref_NS.T) )
PCA3_test1 = np.real( eig_vec[:,2].dot(ref_NS.T) )
ax.plot(PCA1_test1,PCA2_test1,PCA3_test1,'b*', markersize=10, label='$Non$-$seizure$')
plt.show()

# Ref S:
PCA1_test2 = np.real( eig_vec[:,0].dot(ref_S.T) )
PCA2_test2 = np.real( eig_vec[:,1].dot(ref_S.T) )
PCA3_test2 = np.real( eig_vec[:,2].dot(ref_S.T) )
ax.plot(PCA1_test2,PCA2_test2,PCA3_test2,'r+', markersize=10, label='$Seizure$')
plt.show()

ax.axes.set_xlim3d(left=-10000, right=2000) 
ax.axes.set_ylim3d(bottom=-10000, top=2000) 
ax.axes.set_zlim3d(bottom=-5000, top=2000) 

ax.legend(fontsize=15)

# ax.view_init(-150,170)

#%% Histograms:
# plt.figure()

# maxmax = np.max([PCA1_test1,PCA1_test2])

# plt.hist(PCA1_test1/maxmax, bins=2000, histtype='step', fill=True, color='skyblue')

# plt.hist(PCA1_test2/maxmax, bins=2000, histtype='step', fill=True, color='lightcoral')

# plt.xlim(0,100)


#%% Kruskal-Wallis and Dunn's test
import scipy
import scikit_posthocs as sp

# PCA 1:
KW_stat_1, p_value_KW_1 = scipy.stats.kruskal( np.real(PCA1_test1), np.real(PCA1_test2)  )
dunns_test_1 = [np.real(PCA1_test1), np.real(PCA1_test2)]
p_values_dunn_1 = sp.posthoc_dunn(dunns_test_1, p_adjust = 'bonferroni')

# PCA 2:
KW_stat_2, p_value_KW_2 = scipy.stats.kruskal( np.real(PCA2_test1), np.real(PCA2_test2)  )
dunns_test_2 = [np.real(PCA2_test1), np.real(PCA2_test2)]
p_values_dunn_2 = sp.posthoc_dunn(dunns_test_2, p_adjust = 'bonferroni')

# PCA 3:
KW_stat_3, p_value_KW_3 = scipy.stats.kruskal( np.real(PCA3_test1), np.real(PCA3_test2)  )
dunns_test_3 = [np.real(PCA3_test1), np.real(PCA3_test2)]
p_values_dunn_3 = sp.posthoc_dunn(dunns_test_3, p_adjust = 'bonferroni')









