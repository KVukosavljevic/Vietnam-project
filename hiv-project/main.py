# Load NeuroKit and other useful packages
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

from pathlib import Path 
import typer

# Path to data
sh_path=Path("./RAW_DATA/Shimmer/")
ppg_path=Path("./RAW_DATA/PPG-Smartcare/")


def main():
    print(sh_path)
    print(ppg_path)

if __name__ == "__main__":
    typer.run(main)

