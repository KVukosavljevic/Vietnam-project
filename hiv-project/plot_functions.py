import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import sosfilt

from data import butter_bandpass

def plot_spo2_checks_hist(data):

    indicator = [el for el in range(len(data[:,3])) if data[el,3] > 0.]
    datahist = [indicator[i] - indicator[i-1] for i in range(1,len(indicator))]
    plt.hist(datahist, bins=1000, range=(5, np.max(datahist)))
    plt.title('Distribution of the number of samples between consecutive SpO2 status checks.')
    plt.show()

def plot_pleth(data):
    '''This funtion plots the ppg data before and after filtering. This function does not change the data.'''

    x = data[:,0] #['TIMESTAMP_MS']
    y = data[:,4] #['PLETH']
    
    fig, axs = plt.subplots(2,2)
    fig.suptitle('PPG data plot')

    ax1 = axs[0,0]
    ax2 = axs[1,0]
    ax3 = axs[0,1]
    ax4 = axs[1,1]

    ax1.plot(x, y)
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('PPG')
    ax3.set_title('Raw PLETH data')

    # Get spectrum 
    N = len(x)
    T = 1/100    # https://www.mdpi.com/2673-4591/2/1/80 says sampling rate is 100Hz

    yf = fft(y)
    xf = fftfreq(N, T)[:N//2]

    ax2.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Amplitude')

    # Band pass filtered with 0.4 Hz and 16.8 Hz based on https://www.mdpi.com/2673-4591/2/1/80#B7-engproc-02-00080

    bp_filt = butter_bandpass(lowcut=0.1,highcut=16.8,fs=1/T,order=4)
    yfilt = sosfilt(bp_filt, y)

    ax3.plot(x, yfilt)
    ax3.set_xlabel('Time [ms]')
    ax3.set_ylabel('PPG')
    ax3.set_title('Filtered PLETH data')

    # Get spectrum 
    N = len(x)
    T = 1/100 

    yfiltf = fft(yfilt)
    xf = fftfreq(N, T)[:N//2]

    ax4.plot(xf, 2.0/N * np.abs(yfiltf[0:N//2]))
    ax4.set_xlabel('Frequency [Hz]')
    ax4.set_ylabel('Amplitude')

    fig.show()

    input()

def plot_ppg_feats(data, loc_max, loc_min, ppg_features):
    '''This function plots ppg data with marked max and min, as well as P2P time interval and P2P lengths.'''
    fig, axs = plt.subplots(3,1)
    fig.suptitle('PPG data plot')

    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]

    ax1.plot(data[:,0], data[:,4])
    ax1.plot(data[loc_max,0], data[loc_max,4], 'r*')
    ax1.plot(data[loc_min,0], data[loc_min,4], 'b*')
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('no-DC PPG')
    ax1.set_title('Pleth data with marked maxima and minima.')
    
    ax2.hist(ppg_features["p2p_int"])
    ax2.set_xlabel('Time [ms]')
    ax2.set_ylabel('Number')
    ax2.set_title('Peak-to-peak lengths in ms')
    
    ax3.hist(ppg_features["las_ind"])
    ax3.set_xlabel('Time [ms]')
    ax3.set_ylabel('Number')
    ax3.set_title('Las index or p2p time interval')
    
    fig.show()
    input()