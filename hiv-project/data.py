import csv
import pandas as pd
import pathlib
import os
import re

def get_file_names(path):
    csv_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    
    return csv_files

def get_datatype(path):

    datatype = pathlib.PurePath(path).name
    if datatype == "Shimmer":
        return "shimmer"
    if datatype == "PPG-Smartcare":
        return "ppg"
    
    print(f"Not supported data type {datatype}")
    return "no-type"
#column_names = ['ECG_LL-RA_24BIT', 'ECG_LL-RA_24BIT', 'System_Timestamp_Plot_Zeroed',  ...
    #'ECG_LL-LA_24BIT', 'ECG_LL-LA_24BIT', 'Timestamp', 'Timestamp', 'ECG_Vx-RL_24BIT',  ... 
    #'ECG_Vx-RL_24BIT', 'ECG_LA-RA_24BIT', 'ECG_LA-RA_24BIT', 'ECG_EMG_Status1', 'ECG_EMG_Status1', ...
    #'ECG_EMG_Status2', 'ECG_EMG_Status2']
    
def get_label(file):
    "Return 0 for a patient wih low CVD risk and 1 for high risk."
    
    patient_id = int(re.split('-|_|/',os.path.split(file)[0])[5])

    visit = 1
    if ("L2" in file) or ("_2.csv" in file) or ("-2.csv" in file):
        visit = 2 

    return int(patient_id > 30), visit 

def read_data(file, datatype):
    "Reads shimmer and ppg data, returns the data and column names as numpy arrays."

    if datatype == 'shimmer':
        df = pd.read_csv(file, delimiter='\t', low_memory=False, header=1)

        # Filter the ECG data
        filter = ( (df != 'no_units')).all()
        sub_df = df.loc[: , filter]
        filter = ( (df != 'nan')).all()
        sub_df = sub_df.loc[: , filter]
        
        sub_df = sub_df[2:].copy()
        column_names = sub_df.columns
        
        sub_df = sub_df.to_numpy(dtype=float)[:,:-1]
        column_names = column_names.to_numpy(dtype=str)[:-1]

    elif datatype == 'ppg':
        df = pd.read_csv(file, delimiter=',', low_memory=False, header=0)
        
        # Example print to check data print(df.loc[5:10,:])
        # All cols are ['TIMESTAMP_MS', 'COUNTER', 'DEVICE_ID', 'PULSE_BPM', 'SPO2_PCT', 'SPO2_STATUS', 'PLETH',
        #                'BATTERY_PCT', 'RED_ADC', 'IR_ADC','PERFUSION_INDEX']

        # Column names I don't need for now: ['COUNTER', 'DEVICE_ID', 'RED_ADC', 'IR_ADC', 'BATTERY_PCT']
        dropped_cols = ['COUNTER', 'DEVICE_ID', 'RED_ADC', 'IR_ADC', 'BATTERY_PCT']
        df = df.drop(dropped_cols, axis=1)
        
        # Filter the ppg data, basic data cleaning
        filter = ( (df != 'no_units')).all()
        sub_df = df.loc[: , filter]
        filter = ( (sub_df != 'nan')).all()
        sub_df = sub_df.loc[: , filter]

        # Extract data and column names
        column_names = sub_df.columns.to_numpy(dtype=str)
        sub_df = sub_df.to_numpy(dtype=float)

    else:
        print('Not recognised datatype.')

    return sub_df, column_names