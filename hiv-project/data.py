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

    if patient_id <= 30 : return 0

    return 1

def read_data(file):

    df = pd.read_csv(file, delimiter='\t', low_memory=False, header=1)
    
    filter = ( (df != 'no_units')).all()
    sub_df = df.loc[: , filter]
    filter = ( (df != 'nan')).all()
    sub_df = sub_df.loc[: , filter]
    
    sub_df = sub_df[2:].copy()
    column_names = sub_df.columns

    sub_df = sub_df.to_numpy(dtype=float)[:,:-1]
    column_names = column_names.to_numpy(dtype=str)[:-1]

    return sub_df, column_names