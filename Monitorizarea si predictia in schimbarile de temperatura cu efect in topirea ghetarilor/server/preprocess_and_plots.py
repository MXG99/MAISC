from datetime import date, datetime

from pandas import DataFrame, Series
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
          'November', 'December']
monthsd = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'September':9, 'October':10,
          'November':11, 'December':12}


def preprocess(df: DataFrame) -> DataFrame:
    df=df.drop(columns=['Area Code','Months Code','Element Code','Unit'])
    df = df[df["Months"] != 'Dec–Jan–Feb']
    df = df[df["Months"] != 'Mar–Apr–May']
    df = df[df["Months"] != 'Jun–Jul–Aug']
    df = df[df["Months"] != 'Sep–Oct–Nov']
    df = df[df["Months"] != 'Meteorological year']
    #df=df.loc[df.Months.isin(['January', 'February', 'March', 'April', 'May', 'June', 'July','August', 'September', 'October', 'November', 'December'])]
    Tr_df=df.melt(id_vars=['Area','Months','Element'],var_name='Year', value_name='Temperature')
    Tr_df['Year']=Tr_df['Year'].str[1:].astype('str')
    Tr_df = Tr_df.replace(monthsd)
    for i in range(Tr_df.shape[0]):
        Tr_df.loc[i, 'Date'] = datetime(int(Tr_df.loc[i, 'Year']), Tr_df.loc[i, 'Months'], 1)
        Tr_df.loc[i, 'DateInt'] = int(Tr_df.loc[i, 'Year'])*100 + int(Tr_df.loc[i, 'Months'])
    return Tr_df[Tr_df["Element"]=="Temperature change"]


def plot(df: DataFrame):
    plt.figure(figsize=(15,10))
    plt.scatter(df['Year'].loc[df.Element=='Temperature change'],df['Temperature'].loc[df.Element=='Temperature change'])
    plt.plot(df.loc[df.Element=='Temperature change'].groupby(['Year']).mean(),'r',label='Average')
    plt.axhline(y=0.0, color='b', linestyle='-')
    plt.xlabel('Year')
    plt.xticks(np.linspace(0,58,20),rotation=45)
    plt.ylabel('Temperature change')
    plt.legend()
    plt.title('temp change in Romania')
    plt.show()


def plotByMonth(df: DataFrame, month):
    plt.figure(figsize=(25, 20))
    plt.axhline(y=0.0, color='b', linestyle='-')
    temp_df = df[df["Months"] == month]
    plt.scatter(temp_df['Year'].loc[df.Element == 'Temperature change'], temp_df['Temperature'].loc[temp_df.Element == 'Temperature change'])
    plt.plot(temp_df.loc[temp_df.Element == 'Temperature change'].groupby(['Year']).mean(), 'r', label='Average')
    plt.tight_layout()
    plt.xlabel('Year')
    plt.xticks(np.linspace(0,58,20),rotation=45)
    plt.ylabel('Temperature change')
    plt.legend()
    plt.title('temp change in Romania')
    plt.show()


def lineOfRegression():
    data = pd.read_csv('../datasets/RomaniaData.csv')
    y = np.array(data.iloc[:, :-1].values)
    y = np.array([a[0] for a in y])
    arr = np.array(data.iloc[:, -1].values)
    x = np.empty(0)
    for elem in arr:
        day, month, year = map(int, elem.split("/"))
        dates = int(year * 100 + month)
        x = np.append(x, dates)
    n = np.size(arr)

    y_mean = np.mean(y)
    x_mean = np.mean(x)

    Sxy = np.sum(x * y) - n * x_mean * y_mean
    Sxx = np.sum(x * x) - n * x_mean * x_mean

    b1 = Sxy / Sxx
    b0 = y_mean - b1 * x_mean
    #print('slope b1 is', b1)
    #print('intercept b0 is', b0)

    y_pred = b1 * x + b0

    plt.scatter(x, y, color='red')
    plt.plot(x, y_pred, color='green')
    plt.xlabel('Year')
    plt.ylabel('Temperature change')
    plt.title("Line of Regression")
    plt.show()


def plotAllMonths(df: DataFrame):
    colors = ['red', 'plum', 'royalblue', 'slateblue', 'magenta', 'maroon',
              'dodgerblue', 'orange', 'forestgreen', 'lime', 'peru', 'teal']
    # plt.figure(figsize=(25, 20))
    plt.axhline(y=0.0, color='b', linestyle='-')
    for i in range(12):
        temp_df = df[df["Months"] == i+1]
        plt.scatter(temp_df['Year'].loc[df.Element == 'Temperature change'],
                    temp_df['Temperature'].loc[temp_df.Element == 'Temperature change'])
    plt.plot(df.loc[df.Element == 'Temperature change'].groupby(['Year']).mean()['Temperature'],
            label='Average')
    plt.xlabel('Year')
    plt.xticks(np.linspace(0, 58, 20), rotation=45)
    plt.ylabel('Temperature change')
    plt.legend(months)
    plt.title('temp change in Romania')
    plt.tight_layout()
    plt.show()


def plot_overtime(df: DataFrame):
    yr = [str(d).split("-")[0] for d in df['Date']]
    mth = [str(d).split("-")[1] for d in df['Date']]
    dates = [int(yr[i])*100+int(mth[i]) for i in range(len(yr))]
    plt.figure(figsize=(55, 15))
    plt.scatter(dates, np.array(df["Temperature"]))
    plt.tight_layout()
    plt.show()


def plot_years(df: DataFrame):
    plt.set_cmap('Set3')
    plt.figure(figsize=(40, 15))
    for mth in months:
        df_mth = df[df["Months"] == mth]
        plt.plot(df_mth["Year"], df_mth["Temperature"], label=mth)
    plt.plot(df["Year"].unique(), np.zeros(len(df["Year"].unique())))
    plt.legend(loc='best', fontsize=20)
    plt.title("Temperature change over 60 years in Romania", fontsize=30)
    plt.xlabel('Years', fontsize=20)
    plt.ylabel('Temperature', fontsize=20)
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(np.arange(min(df["Temperature"]), max(df["Temperature"])+1, 1.0), fontsize=20)
    plt.tight_layout()
    plt.show()

