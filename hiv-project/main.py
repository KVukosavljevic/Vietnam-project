# Load NeuroKit and other useful packages
import pathlib
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from data import get_file_names, get_datatype, read_data, get_label
from features import get_features

from pathlib import Path 
import typer
import os

# Path to data
sh_path=Path("./RAW_DATA/Shimmer/")
ppg_path=Path("./RAW_DATA/PPG-Smartcare/")


def main(
    path : Path = typer.Option(
        ..., '--path', help='Path to data'
    ),
):
    path = path
    file_names = get_file_names(path)
    print(path.absolute())
    print(file_names)
    
    datatype = get_datatype(path) # For later use

    # Data prep
    tr_normal = []
    tr_abnormal = []

    ts_normal = []
    ts_abnormal = []

    for file in file_names:
        # Get the data
        data, columns = read_data(file)
    
        # Get label
        label, visit = get_label(file)

        print(f"Data read for patient {label} and visit {visit} for file {file}.")

        # Obtain features
        try:
            features = get_features(data, columns)

            if label == 0 and visit == 1:
                if np.array(features).shape[2] == 25:
                    tr_normal.extend(features)
            elif label == 1 and visit == 1:
                if np.array(features).shape[2] == 25:
                    tr_abnormal.extend(features)
            elif label == 0 and visit == 2:
                if np.array(features).shape[2] == 25:
                    ts_normal.extend(features)
            elif label == 1 and visit == 2:
                if np.array(features).shape[2] == 25:
                    ts_abnormal.extend(features)
            else:
                print(f"Not categorised for label {label} and visit {visit}.")
        except:
            print(f"Not able to obtain features for file: {file}")

    tr_normal = np.array(tr_normal)
    tr_abnormal = np.array(tr_abnormal)

    ts_normal = np.array(ts_normal)
    ts_abnormal = np.array(ts_abnormal)
    
    # Data shapes
    print(f"Training data shapes: normal {tr_normal.shape} and abnormal {tr_abnormal.shape}.")
    print(f"Test data shapes: normal {ts_normal.shape} and abnormal {ts_abnormal.shape}.")

    # Saving the data
    np.save('tr_normal', tr_normal)
    np.save('tr_abnormal', tr_abnormal)

    np.save('ts_normal', ts_normal)
    np.save('ts_abnormal', ts_abnormal)

    print(f"Data successfully saved.")

if __name__ == "__main__":
    typer.run(main)

