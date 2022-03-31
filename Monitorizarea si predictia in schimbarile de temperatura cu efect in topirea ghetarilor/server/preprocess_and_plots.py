from typing import Optional
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np


def preprocess(df: DataFrame) -> DataFrame:
    df = df.drop(columns=['Area Code', 'Months Code', 'Element Code', 'Unit'])
    # df=df.loc[df.Months.isin(['January', 'February', 'March', 'April', 'May', 'June', 'July','August', 'September', 'October', 'November', 'December'])]
    Tr_df = df.melt(id_vars=['Area', 'Months', 'Element'], var_name='Year', value_name='Temperature')
    Tr_df['Year'] = Tr_df['Year'].str[1:].astype('str')
    return Tr_df[Tr_df["Element"] == "Temperature change"]


def plot(df: DataFrame, place: str = 'Romania'):
    plt.figure(figsize=(15, 10))
    plt.scatter(df['Year'].loc[df.Element == 'Temperature change'],
                df['Temperature'].loc[df.Element == 'Temperature change'])
    plt.plot(df.loc[df.Element == 'Temperature change'].groupby(['Year']).mean(), 'r', label='Average')
    plt.axhline(y=0.0, color='b', linestyle='-')
    plt.xlabel('Year')
    plt.xticks(np.linspace(0, 58, 20), rotation=45)
    plt.ylabel('Temperature change')
    plt.legend()
    plt.title(f'temp change in {place}')
    plt.show()


def plotByMonth(df: DataFrame, month: Optional[str] = None):
    if month is None:
        plot(df)
        return
    temp_df = df[df["Months"] == month]
    plt.figure(figsize=(15, 10))
    plt.scatter(temp_df['Year'].loc[df.Element == 'Temperature change'],
                temp_df['Temperature'].loc[temp_df.Element == 'Temperature change'])
    plt.plot(temp_df.loc[temp_df.Element == 'Temperature change'].groupby(['Year']).mean(), 'r', label='Average')
    plt.axhline(y=0.0, color='b', linestyle='-')
    plt.xlabel('Year')
    plt.xticks(np.linspace(0, 58, 20), rotation=45)
    plt.ylabel('Temperature change')
    plt.legend()
    plt.title(f'temp change in Romania')
    plt.show()
