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
    
    datatype = get_datatype(path) # For later use

    # Data prep
    normal = []
    abnormal = []
    for file in file_names:
        # Get the data
        data, columns = read_data(file)
    
        # Get label
        label = get_label(file)

        print(label)

        # Obtain features
        try:
            features = get_features(data, columns)

            if label == 0 :
                if np.array(features).shape[2] == 25:
                    normal.extend(features)
            else:
                if np.array(features).shape[2] == 25:
                    abnormal.extend(features)
        except:
            print(file)


    normal = np.array(normal)
    abnormal = np.array(abnormal)
    print(normal)

    # Saving the data
    np.save('normal', normal)
    np.save('abnormal', abnormal)


if __name__ == "__main__":
    typer.run(main)

