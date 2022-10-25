import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

def get_file_names(path, print_names=False):
    "Gets all csv files for Shimmer or PPG data. Make sure that the path ends with //."
    
    f_n_1 = glob.glob(path + "*\\*.csv")
    f_n_2 = glob.glob(path + "*\\*\\*.csv")
    
    file_names = []
    file_names.extend(f_n_1)
    file_names.extend(f_n_2)
    
    print(path.split('\\')[-2],'data file names extracted.')
    
    if print_names == True : print(file_names)
    
    return file_names

def get_id(path):
    
    ID = int(file.replace(path,"").split("\\", 1)[0].split("-", 2)[-1])
    
    return ID


def load_data(path):
    
    # Think about how you're gonna implement all the different channels
    df = pd.read_csv(path, delimiter='\t', low_memory=False, header=1)
    
    # Extracting ECG_LL the one with mV
    data_ch1 = df['ECG_LL-RA_24BIT.1']
    data_ch1 = data_ch1.to_numpy()[:-3]
    data_ch1 = data_ch1[2:]
    data_ch1 = np.array([float(el) for el in data_ch1])
    
    plt.rcParams['figure.figsize'] = [15, 5]  # Bigger images
    plt.rcParams['font.size'] = 14
    
    # Load ecg peaks for a specific channel
    peaks, info = nk.ecg_peaks(data_ch1, sampling_rate=100)
    
    # Load HRV basic time stats and frequency stats ---------------------- could separate HRV as a separate
    hrv_time = nk.hrv_time(peaks, sampling_rate=100, show=False)
    hrv_freq = nk.hrv_frequency(peaks, sampling_rate=100, show=False, normalize=True)
    
    # ECG bands
    ulf = hrv_freq['HRV_ULF'][0]
    vlf = hrv_freq['HRV_VLF'][0]
    lf = hrv_freq['HRV_LF'][0]
    hf = hrv_freq['HRV_HF'][0]
    vhf = hrv_freq['HRV_VHF'][0]

    
    
    
    
    
    
    
    
    
    
    
    
    
    