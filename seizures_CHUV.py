#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:21:28 2023

@author: joao
"""
import os
import re
import scipy
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from scipy import signal
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter,filtfilt

# %% Sampling frequency:
# DNK patients:
# 19 x
# 26 x
# 37 x (issue on the Excel file --> get end time and subtract seizure duration)
# 43 x
# 50 x
# 52 doesnt have end time
# 55 x *
# 60 doesnt have end time

# ZCH patients:
# 03 x
# 04 x
# 14 x
# 16 x
# 19 x
# 23 x
# 26 x
    
f_DNK = 1024
f_ZCH = 256

place = 'ZCH'
no_pt = '14'
Fs = f_ZCH

#%% Filters:
def notch_filter(data_notch, Fs, notch_freq, quality_factor):
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, Fs)
    freq, h = signal.freqz(b_notch, a_notch, fs = Fs)
    notched_y = filtfilt(b_notch, a_notch, data_notch)
    return notched_y

def __notch__filter__( sign, samp_freq ):     
    notch_freq = 50
    quality_factor = 30

    # Notch filter:    
    n_freqs = 1
    aux = sign
    for ii in range(0, n_freqs + 1):
        notched_sig = notch_filter(aux, samp_freq, notch_freq*(ii + 1), quality_factor)
        aux = notched_sig
    
    return notched_sig
    
# %% Import seizures and information:
seizure_info_path = "SEVERITY-" + place + "-AnalyseVideo.xlsx"
seizure_info = pd.read_excel(
    seizure_info_path,
    sheet_name='analyse crises',
    header=1
)

seizure_info.head()

seizure_info = seizure_info[[
    'Patient number',
    'seizure number',
    'Date of the seizure',
    'Start hour of the focal seizure',
    'End hour of the clonic phase (=end of the seizure)',
    'Type of seizure'
]].rename(columns={
    'Patient number': 'patient_ID',
    'seizure number': 'seizure_ID',
    'Date of the seizure': 'date',
    'Start hour of the focal seizure': 'start_time',
    'End hour of the clonic phase (=end of the seizure)': 'end_time',
    'Type of seizure': 'seizure_type'
})

# Filter the non-GTCS seizure
seizure_info = seizure_info[seizure_info.seizure_type.str.contains('GTCS')]

# Remove non-existant patient
seizure_info = seizure_info.dropna(subset='patient_ID')
seizure_info.patient_ID = seizure_info.patient_ID.astype(int)

seizure_info.patient_ID = "SEV-" + place + "-" + \
    seizure_info.patient_ID.astype(str).str.zfill(2)

seizure_info.head()

# %% Load specific seizure from a patient:
patient_ID = "SEV-" + place + "-" + no_pt
seizure_ID = 1

seizure_data = pd.read_parquet(f"{patient_ID}_seizure_{seizure_ID}.parquet")

seizure_data.head()

# %% Visualize the seizure for one channel:
time_before = (60, 'minute')
time_after = (60, 'minute')

this_seizure_info = seizure_info[(seizure_info.patient_ID == patient_ID) & (
    seizure_info.seizure_ID == seizure_ID)]

# get seizure start time
start_time = pd.Timestamp((this_seizure_info.date.astype(
    str) + ' ' + this_seizure_info.start_time.astype(str)).values[0])

end_time = pd.Timestamp((this_seizure_info.date.astype(
    str) + ' ' + this_seizure_info.end_time.astype(str)).values[0])

# get the data around seizure
t0 = start_time - pd.Timedelta(time_before[0], unit=time_before[1])
t1 = end_time   + pd.Timedelta(time_after[0], unit=time_after[1])

# x=seizure_data[t0:t1].index
# y=seizure_data[t0:t1].Fp1

# %% Convert Pandas series to array:
channels = list(seizure_data.columns)

seizures_list = []
nc = len(channels)

for ii in range(0, nc):
    seizure_series = getattr(seizure_data[t0:t1], channels[ii])
    seizure_series_array = np.array(pd.Series.tolist( seizure_series ))
    full_filtered_sig  = __notch__filter__( seizure_series_array, Fs )
    seizures_list.append(full_filtered_sig)

# %% Save signals as .txt files:
# for mm in range(0, nc):
#     folder_name = os.getcwd() + "/SEV-" + place + "-" + no_pt + '_seizure_1/'
#     filename = folder_name + "SEV-" + place + "-" + no_pt + '_seizure_1_' + channels[mm] + '.txt'
#     with open(filename, "a") as file_object:
#         aux_ = seizures_list[mm]
#         for nn in range(0, len(aux_)):
#             file_object.write(str(aux_[nn]) + '\n')

# %% Import signals from .txt files:
sig_vec = []

# Measurable activity:
for ww in range(0,nc):
    data_folder = os.getcwd() + "/SEV-" + place + "-" + no_pt + '_seizure_1/'
    filename = r'' + data_folder + "SEV-" + place + "-" + no_pt + '_seizure_1_' + channels[ww] + '.txt'
    fsig = open(filename, 'r')
    signal_aux = re.findall(r"\S+", fsig.read())
    fsig.close()
    sig = np.zeros(len(signal_aux))
    for jj in range(0, len(signal_aux)):
        sig[jj] = round(float(signal_aux[jj]),14)

    sig_vec.append(sig)

#%% Define windows:
# Reference: 1h before, seizure onset, seizure offset, 1h after
sec_windows = []
current_windows = [ str(t0)[-8:], str(start_time)[-8:], str(end_time)[-8:], str(t1)[-8:] ]
ftr = [3600,60,1]

for kk in range(4): 
    timestr = current_windows[kk]
    sec_window = sum([a*b for a,b in zip(ftr, map(int,timestr.split(':')))])
    sec_windows.append(sec_window)
    
norm_sec_windows = np.array([sec_windows]) - sec_windows[0]
seizure_onset  = Fs * norm_sec_windows[0][1]
seizure_offset = Fs * norm_sec_windows[0][2]
end_recording  = Fs * norm_sec_windows[0][3]
seizure_duration = seizure_offset - seizure_onset

print('Seizure onset: ' +  str(seizure_onset /(Fs * 60)) + 'min')
print('Seizure offset: ' + str(seizure_offset/(Fs * 60)) + 'min')

# %% Analysis in the frequency domain using Power Spectral Density (PDS): whole signal
colors = ['k','r--','b:']

N = seizure_duration
dt = 1/Fs
t = np.linspace(0,len(seizures_list[0])*dt,len(seizures_list[0]))
start_window = seizure_onset
end_window = seizure_offset

for outer_loop in range(0,2):
    if outer_loop == 0:
        gen_color = colors[0]
        start_window = 0
        end_window = N
        
    if outer_loop == 1:
        gen_color = colors[1]
        start_window = seizure_onset
        end_window = seizure_offset
        
    if outer_loop == 2:
        gen_color = colors[2]
        start_window = -N
        end_window = -1

    NpS_w = int(N//1)
    Nlap_w = int(NpS_w*0.1)
    Pxx_w_vec = []
    fft_w_vec = []
    inst_freq_vec = []

    plt.figure(1)
    for nn in range(0,22):
        sig_vec_window = seizures_list[nn][start_window:end_window]
        
        # Hilbert:
        analytic_sig = scipy.signal.hilbert(sig_vec_window)
        amp_env = np.abs(analytic_sig)
        inst_phase = np.unwrap(np.angle(analytic_sig))
        inst_phase_hz = (np.diff(inst_phase) / (2.0*np.pi) * Fs)
        inst_freq_vec.append( inst_phase_hz )
        
        # PSD:
        f_mw, Pxx_w = scipy.signal.welch(sig_vec_window, fs=Fs, window='hann', 
                                      nperseg=NpS_w, noverlap=Nlap_w,
                                      nfft=None, scaling='density', average='mean')
        
        Pxx_w_vec.append(Pxx_w)
    
        # FFT:
        sig_fft = fft(sig_vec_window)
        xfreq = fftfreq(N, dt)[:N//2]
        fft_abs = (2.0/N) * np.abs(sig_fft[0:N//2])
        fft_w_vec.append(fft_abs)

        plt.subplot(221)
        plt.semilogy(f_mw, Pxx_w, gen_color, linewidth=0.25)
        plt.ylabel('$PSD$ $[mV^2/Hz]$', fontsize=10)
        plt.xlim(0, Fs//2)
        plt.tick_params(axis='both', which='major', labelsize=10)
        
        plt.subplot(222)
        plt.plot(xfreq, fft_abs, gen_color, linewidth=0.25)
        plt.xlabel('$f$ $[Hz]$', fontsize=10)
        plt.ylabel('$FFT$ $[mV]$', fontsize=10)
        plt.xlim(0, Fs//2)
        plt.tick_params(axis='both', which='major', labelsize=10)
    
        plt.subplot(223)
        plt.plot(t[start_window + 1:end_window],inst_phase_hz,gen_color,linewidth=0.25)
        plt.ylabel('$f$ $[Hz]$', fontsize=10)
        plt.xlabel('$t$ $[s]$', fontsize=10)
        plt.xlim(t[0],t[-1])
        plt.tick_params(axis='both', which='major', labelsize=10)
    
        plt.subplot(224)
        data = inst_phase_hz
        kde = stats.gaussian_kde(data)
        x = np.linspace(data.min(), data.max(), 200)
        p = kde(x)
        plt.plot(x,p,gen_color)
        plt.ylabel('$f$ $[Hz]$', fontsize=10)
        plt.xlabel('$t$ $[s]$', fontsize=10)
        plt.tick_params(axis='both', which='major', labelsize=10)
    
    #% Plots media
    plt.figure(2)
    plt.subplot(221)
    plt.semilogy(f_mw,np.mean(Pxx_w_vec,axis=0),gen_color,linewidth=0.25)
    plt.ylim(0,np.max(Pxx_w_vec))
    plt.ylabel('$PSD$ $[mV^2/Hz]$', fontsize=10)
    plt.xlim(0, Fs//2)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.grid()
    plt.subplot(222)
    plt.plot(xfreq,np.mean(fft_w_vec,axis=0),gen_color,linewidth=0.25)
    plt.ylim(0,np.max(fft_w_vec))
    plt.xlabel('$f$ $[Hz]$', fontsize=10)
    plt.ylabel('$FFT$ $[mV]$', fontsize=10)
    plt.xlim(0, Fs//2)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.grid()
    plt.subplot(223)
    plt.plot(t[start_window+1:end_window],np.mean(inst_freq_vec,axis=0),gen_color,linewidth=0.25)
    plt.grid()
    plt.xlim(t[0],t[-1])
    plt.ylabel('$f$ $[Hz]$', fontsize=10)
    plt.xlabel('$t$ $[s]$', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.subplot(224)
    data = np.mean(inst_freq_vec,axis=0)
    kde = stats.gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 200)
    p = kde(x)

    plt.plot(x,p,gen_color)
    plt.ylim(0,1)
    plt.xlabel('$f$ $[Hz]$', fontsize=10)
    plt.ylabel('$p(f)$', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)
