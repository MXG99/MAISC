import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt

from preprocess_and_plots import plot, plotByMonth, preprocess

filename = "Environment_Temperature_change_E_All_Data_NOFLAG.csv"

if __name__ == '__main__':
    dataset = pd.read_csv(filename, encoding='Windows-1252')
    print(dataset.head())
    print(dataset.shape)

    romaniaData = dataset[dataset["Area"] == "Romania"]
    print(romaniaData.head())
    print(romaniaData.shape)

    romanianDataPreprocessed = preprocess(romaniaData)
    plotByMonth(romanianDataPreprocessed, "July")
    