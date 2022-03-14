import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt
import chardet

filename = "Environment_Temperature_change_E_All_Data_NOFLAG.csv"

if __name__ == '__main__':
    import chardet

    # with open(filename, 'rb') as rawdata:
    #     result = chardet.detect(rawdata.read(100000))
    # print(result)
    # #
    dataset = pd.read_csv(filename, encoding='Windows-1252')
    print(dataset.head())
