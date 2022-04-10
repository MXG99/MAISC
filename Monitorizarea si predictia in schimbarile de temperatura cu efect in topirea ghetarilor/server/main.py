import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt

from preprocess_and_plots import *
from model import *

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

filename = "../datasets/Environment_Temperature_change_E_All_Data_NOFLAG.csv"

if __name__ == '__main__':
    dataset = pd.read_csv(filename, encoding='Windows-1252')
    # print(dataset.head())
    # print(dataset.shape)

    romaniaData = dataset[dataset["Area"] == "Romania"]
    # print(romaniaData.head())
    # print(romaniaData.shape)

    romanianDataPreprocessed = preprocess(romaniaData)
    print(romanianDataPreprocessed.head())

    lineOfRegression()
    # plot_years(romanianDataPreprocessed)
    # plot_overtime(romanianDataPreprocessed)
    # plotAllMonths(romanianDataPreprocessed)

    # plot_years(romanianDataPreprocessed)
    # plotByMonth(romanianDataPreprocessed, 'December')

    # processedData = preprocess(dataset)
    # print(processedData.head())

    # processedData.to_csv('ProcessedData.csv')
    # romanianDataPreprocessed.to_csv('../datasets/RomaniaData.csv')

    # X = romanianDataPreprocessed["Months", "Year"]
    # Y = romanianDataPreprocessed["Temperature"]
    # split_data(X, Y)

    romania_dataset = pd.read_csv("../datasets/RomaniaData.csv")
    bucuresti_dataset = pd.read_csv("../datasets/Bucuresti.csv")

    header = ["t_chg", "date", "tavg", "tmin", "tmax"]
    with open('../datasets/RomaniaDataV3.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row1, row2 in zip(romania_dataset.itertuples(), bucuresti_dataset.itertuples()):
            date = datetime.strptime(row1.date, '%d/%m/%Y').date()
            list = [round(row1.t_chg, 3), date, int(row2.tavg + round(row1.t_chg, 3)), row2.tmin, row2.tmax]
            new_row = [str(i) for i in list]
            writer.writerow(new_row)
