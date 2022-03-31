from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def preprocess(df: DataFrame) -> DataFrame:
    df=df.drop(columns=['Area Code','Months Code','Element Code','Unit'])
    #df=df.loc[df.Months.isin(['January', 'February', 'March', 'April', 'May', 'June', 'July','August', 'September', 'October', 'November', 'December'])]
    Tr_df=df.melt(id_vars=['Area','Months','Element'],var_name='Year', value_name='Temperature')
    Tr_df['Year']=Tr_df['Year'].str[1:].astype('str')
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

def plotByMonth(df: DataFrame, month: str):
    temp_df = df[df["Months"] == month]
    plt.figure(figsize=(15,10))
    plt.scatter(temp_df['Year'].loc[df.Element=='Temperature change'],temp_df['Temperature'].loc[temp_df.Element=='Temperature change'])
    plt.plot(temp_df.loc[temp_df.Element=='Temperature change'].groupby(['Year']).mean(),'r',label='Average')
    plt.axhline(y=0.0, color='b', linestyle='-')
    plt.xlabel('Year')
    plt.xticks(np.linspace(0,58,20),rotation=45)
    plt.ylabel('Temperature change')
    plt.legend()
    plt.title('temp change in Romania')
    plt.show()

def lineOfRegression():
    data = pd.read_csv('RomaniaData.csv', header=None)
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