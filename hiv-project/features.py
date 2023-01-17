import numpy as np
import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from itertools import pairwise
from sklearn.metrics import auc


def ecg_features(data):
    frame_len = 10_000
    i = 0
    hrvs = []
    while (i + frame_len) <= len(data):
        peaks, _ = nk.ecg_peaks(data[i:i + frame_len], sampling_rate=100)
        try:
            hrv_time = nk.hrv_time(peaks, sampling_rate=100, show=False)
            hrv_freq = nk.hrv_frequency(peaks, sampling_rate=100, show=False, normalize=True)
        
        
            hrvs.append(hrv_time.join(hrv_freq, lsuffix='_time', rsuffix='_freq'))
        except:
            pass 

        i+=frame_len

    # Drop columns with nans
    hrvs = pd.concat(hrvs)
    hrvs.dropna(axis='columns', inplace=True)
    return hrvs.to_numpy(dtype=float)

def get_features_ecg(data, columns):

    ecg_cols = list([True if ('ECG' in name) else False for name in columns])

    ecg_data = np.array(data[:,ecg_cols])
    ecg_columns = np.array(columns[ecg_cols])
    assert ecg_data.shape[1] == len(ecg_columns)
    
    time_cols = [True if el is False else False for el in ecg_cols]

    time_data = np.array(data[:,time_cols])
    time_columns = np.array(columns[time_cols])
    assert time_data.shape[1] == len(time_columns)

    # Obtain hrv features for all channels
    hrvs = []
    for channel in range(len(ecg_data[0,:])):
        ecg_feats = ecg_features(ecg_data[:,channel])
        hrvs.append(ecg_feats)

    return hrvs

def get_features_ppg(data, columns):
    "Retrieving features from ppg data"

    plot_ppg = False
    if plot_ppg is True: plotppg(data, columns)

    # What do SPO2 status jumps mean?

    ############# PPG data ##################
    # Extracting features mentioned in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6426305/

    # There are 4 ppg and 7 ppg second derivative features
    # ppg_features = [Systolic Amplitude, Pulse Area, Peak to Peak Interval, Large Artery Stiffness Index]
    ppg_feats = ['sys_amp', 'pulse_area', 'p2p_int', 'las_index']
    ppg_features = {}

    # for sys_amp detect local maxima
    # for pulse area, detect local minima and integrate under 
    # for p2p interval, detect differences between local maxima in value
    # for las_index p2p time interval

    loc_max = argrelextrema(data[:,4], np.greater)[0]
    loc_min = argrelextrema(data[:,4], np.less)[0]

    ppg_features["sys_time"] = data[loc_max,0]
    ppg_features["sys_amp"] = data[loc_max,4]

    ppg_features["dia_time"] = data[loc_min,0]
    ppg_features["dia_amp"] = data[loc_min,4]

    #
    # ppg_features["pulse_area"] = auc(xx,yy)

    ppg_features["p2p_int"] = np.array([y - x for x, y in pairwise(ppg_features["sys_amp"])])
    print('------------------')
    
    print(ppg_features['sys_amp'])
    print(ppg_features['p2p_int'])

    print('-------------------')

    ppg_features["las_ind"] = np.array([y - x for x, y in pairwise(data[loc_max,0])])
    
    plt.plot(data[:,0], data[:,4])
    plt.plot(data[loc_max,0], data[loc_max,4], 'r*')
    plt.plot(data[loc_min,0], data[loc_min,4], 'r*')
    plt.show()

    plt.hist(ppg_features["p2p_int"])
    plt.title('Peak-to-peak lengths in ms')
    plt.show()

    plt.hist(ppg_features["las_ind"])
    plt.show()

    input()

    return data

def plotppg(data, columns):
    print(columns)

    fig, axs = plt.subplots(3, 2)
    axs[0, 0].plot(data[:,0], data[:,1])
    axs[0, 1].plot(data[:,0], data[:,2], 'tab:orange')
    axs[1, 0].plot(data[:,0], data[:,3], 'tab:green')
    axs[1, 1].plot(data[:,0], data[:,5], 'tab:red')
    axs[2, 0].plot(data[:,0], data[:,4], 'tab:blue')
    axs[2, 1].plot(data[:int(len(data)/8),0], data[:int(len(data)/8),4], 'tab:blue')

    axs[0, 0].set_ylabel(columns[1])
    axs[0, 1].set_ylabel(columns[2])
    axs[1, 0].set_ylabel(columns[3])
    axs[1, 1].set_ylabel(columns[5])
    axs[2, 0].set_ylabel(columns[4])
    axs[2, 1].set_ylabel('Zoomed in ' + columns[4])
    

    for ax in axs.flat:
        ax.set(xlabel=columns[0])

    fig.show()
    input()
