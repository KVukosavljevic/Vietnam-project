import numpy as np
import neurokit2 as nk
import pandas as pd


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

def get_features(data, columns):

    
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
