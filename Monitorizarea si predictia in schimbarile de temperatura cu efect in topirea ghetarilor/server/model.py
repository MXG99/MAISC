import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def read_data_per_month(filename, month):
    dataset = pd.read_csv(filename + "_" + str(month))
    return dataset


def split_label_outcome(dataset: pd.DataFrame, labelColumnName, outcomeColumnName):
    X = dataset[labelColumnName]
    Y = dataset[outcomeColumnName]
    return X, Y


def split_data(X: pd.DataFrame, Y: pd.DataFrame):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=4 / 5, random_state=42)
    # X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=1 / 3, random_state=42)

    return [X_train, y_train, X_test, y_test]


# sets: X_train, y_train, X_test, y_test, X_valid, Y_valid
def linearReg(sets: []):
    X_train, y_train, X_test, y_test = [a for a in sets]
    reg = LinearRegression()
    X_train_array = np.array(X_train).reshape(-1, 1)
    reg.fit(X_train_array, y_train)
    print(reg.intercept_)
    print(reg.coef_)

    X_test_array = np.array(X_test).reshape(-1, 1)
    prediction = reg.predict(X_test_array)

    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    r2 = r2_score(y_test, prediction)
    print("Mean squared error: " + str(rmse))
    print("Prediction correlation: " + str(r2))

    plt.scatter(y_test, prediction)
    p = 0
    for i in X_test.index.values.tolist():
        plt.annotate(X_test.loc[i], (y_test.loc[i], prediction[p]))
        p += 1
    plt.title("Testing the model")
    plt.ylabel("Predicted value")
    plt.xlabel("Real value")
    plt.show()

    # plt.hist(y_test - prediction)
    # plt.show()

    # lstm

def polynomialReg(sets: []):
    X_train, y_train, X_test, y_test = [a for a in sets]
    poly = PolynomialFeatures(degree = 2)
    X_train_array = poly.fit_transform(X_train).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X_train_array, y_train)
    prediction = model.predict(X_train_array)
    #print(reg.intercept_)
    #print(reg.coef_)

    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    r2 = r2_score(y_test, prediction)
    print("Mean squared error: " + str(rmse))
    print("Prediction correlation: " + str(r2))

    plt.scatter(y_test, prediction)
    p = 0
    for i in X_test.index.values.tolist():
        plt.annotate(X_test.loc[i], (y_test.loc[i], prediction[p]))
        p += 1
    plt.title("Testing the model")
    plt.ylabel("Predicted value")
    plt.xlabel("Real value")
    plt.show()