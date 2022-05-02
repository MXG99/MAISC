from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from utils import timeseries_to_supervised, scale, fit_lstm, forecast_lstm, invert_scale, inverse_difference, difference
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import mean_shift
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


def read_data_per_month(filename, month):
    dataset = pd.read_csv(filename + "_" + str(month))
    return dataset


def split_label_outcome(dataset: pd.DataFrame, labelColumnName, outcomeColumnName):
    X = dataset[labelColumnName]
    Y = dataset[outcomeColumnName]
    return X, Y


def split_data(X: pd.DataFrame, Y: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, train_size=4 / 5, random_state=42)
    # X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=1 / 3, random_state=42)

    return [X_train, y_train, X_test, y_test]


# sets: X_train, y_train, X_test, y_test, X_valid, Y_valid
def linearReg(sets: []):
    print("Linear Regression")
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
    print("Polynomial Regression")
    X_train, y_train, X_test, y_test = [a for a in sets]
    poly = PolynomialFeatures(degree = 2)
    X_train_array = poly.fit_transform(np.array(X_train).reshape(-1, 1))
    X_test_array = poly.fit_transform(np.array(X_test).reshape(-1, 1))

    model = LinearRegression()
    model.fit(X_train_array, y_train)
    prediction = model.predict(X_test_array)
    #print(reg.intercept_)
    #print(reg.coef_)

    print(mean_squared_error(y_test, prediction))
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    r2 = r2_score(y_test, prediction)
    print("Mean squared error: " + str(rmse))
    print("Prediction correlation: " + str(r2))

    zipped_lists = zip(X_test, prediction)
    sorted_pairs = sorted(zipped_lists)

    tuples = zip(*sorted_pairs)
    X_test_sorted, prediction_sorted = [list(tuple) for tuple in tuples]

    plt.plot(X_test_sorted, prediction_sorted, "r-", linewidth=2, label="Predictions")
    plt.plot(X_train, y_train, "b.", label='Training points')
    plt.plot(X_test, y_test, "g.", label='Testing points')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    plt.scatter(y_test, prediction)
    p = 0
    for i in X_test.index.values.tolist():
        plt.annotate(X_test.loc[i], (y_test.loc[i], prediction[p]))
        p += 1
    plt.title("Testing the model")
    plt.ylabel("Predicted value")
    plt.xlabel("Real value")

# sets: X_train, y_train, X_test, y_test, X_valid, Y_valid
def lstm(month, month_name):
    # Load dataset
    csv_name = "../datasets/month_" + str(month)
    tavgs = pd.read_csv(csv_name, header=0, parse_dates=[0], index_col=0)

    # Diff values
    values = tavgs["tavg"].values
    diff_values = difference(values, 1)

    # Supervised values
    supervised = timeseries_to_supervised(values, 1)
    supervised_values = supervised.values

    # Split the data
    train, test = train_test_split(supervised_values, train_size=4 / 5, shuffle=False)
    print(len(train))
    print(len(test))
    scaler, train_scaled, test_scaled = scale(train, test)

    # Starting year of test values
    year_test = 2019 - len(test_scaled) + 1

    lstm_model = fit_lstm(train_scaled, 1, 100, 4)

    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    lstm_model.predict(train_reshaped, batch_size=1)
    predictions = list()
    years = []
    for i in range(len(test_scaled)):
        years.append(year_test + i)
        # make one-step forecast
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forecast_lstm(lstm_model, 1, X)
        # invert scaling
        yhat = invert_scale(scaler, X, yhat)
        # invert differencing
        yhat = inverse_difference(values, yhat, len(test_scaled) + 1 - i)
        # store forecast
        predictions.append(yhat)
        expected = values[len(train) + i]
        print('Year = %d, Predicted=%f, Expected=%f' % ((year_test + i), yhat, expected))

    rmse = sqrt(mean_squared_error(values[-len(test_scaled):], predictions))
    print('Test RMSE: %.3f' % rmse)
    plt.title("Predicted vs Real values with LSTM Forecasting for month %s" % (month_name))
    plt.ylabel("Average temperature")
    plt.plot(values[-len(test_scaled):])
    plt.plot(predictions)
    plt.grid()
    plt.show()


def ann(sets: []):
    X_train, y_train, X_test, y_test = [a for a in sets]
    reg = MLPRegressor(hidden_layer_sizes=(32,16,16, 8), max_iter=500, learning_rate_init=0.0001, )
    X_train_array = np.array(X_train).reshape(-1, 1)
    reg.fit(X_train_array, y_train)

    X_test_array = np.array(X_test).reshape(-1, 1)
    prediction = reg.predict(X_test_array)

    rmse_train = np.sqrt(mean_squared_error(y_test, prediction))

    print(rmse_train)

    plt.scatter(y_test, prediction)
    plt.title("Testing the model")
    plt.xlabel("Predicted value")
    plt.ylabel("Real value")
    plt.show()

def svm(sets: []):
    X_train, y_train, X_test, y_test = [a for a in sets]
    reg = SVR(kernel="rbf", C=100, gamma="auto")
    X_train_array = np.array(X_train).reshape(-1, 1)
    reg.fit(X_train_array, y_train)

    X_test_array = np.array(X_test).reshape(-1, 1)
    prediction = reg.predict(X_test_array)

    rmse_train = np.sqrt(mean_squared_error(y_test, prediction))

    print(rmse_train)

    plt.scatter(y_test, prediction)
    plt.title("Testing the model")
    plt.xlabel("Predicted value")
    plt.ylabel("Real value")
    plt.show()

