import numpy as np
import pandas as pd
import csv, os
from matplotlib import pyplot as plt

from preprocess_and_plots import *
from model import *

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

filename = "../datasets/Environment_Temperature_change_E_All_Data_NOFLAG.csv"

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
          'November', 'December']

if __name__ == '__main__':
    #dataset = pd.read_csv(filename, encoding='Windows-1252')
    # print(dataset.head())
    # print(dataset.shape)

    #romaniaData = dataset[dataset["Area"] == "Romania"]
    # print(romaniaData.head())
    # print(romaniaData.shape)

    # romaniaDataset = pd.read_csv("../datasets/RomaniaDataV2.csv")
    # X = pd.DataFrame()
    # X = romaniaDataset["date"].str.split("-", 1, expand=True)[0]
    # Y = romaniaDataset["tavg"]
    # X_train, y_train, X_test, y_test, X_valid, y_valid = split_data(X, Y)
    #
    # sets = [X_train, y_train, X_test, y_test, X_valid, y_valid]
    #
    # linearReg(sets)

    for month in range(1, 13):
        df = read_data_per_month(os.path.join(os.getcwd(), "Monitorizarea si predictia in schimbarile de temperatura cu efect in topirea ghetarilor", "datasets", "month"), month)
        sns.regplot(x="year", y="tavg", data=df)
        plt.title("Regression line for month " + months[month-1])
        plt.show()
        X, Y = split_label_outcome(df, "year", "tavg")
        X_train, y_train, X_test, y_test = split_data(X, Y)
        # X_train, y_train, X_test, y_test, X_valid, y_valid = split_data(X, Y)
        sets = [X_train, y_train, X_test, y_test]

        linearReg(sets)
    # split_dataset("../datasets/RomaniaDataV2.csv")

