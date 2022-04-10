import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def split_data(X: pd.DataFrame, Y: pd.DataFrame):

    X_train, X_rem, y_train, y_rem = train_test_split(X, Y, train_size=4 / 5, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=1 / 3, random_state=42)

    return [X_train, y_train, X_test, y_test, X_valid, y_valid]


# sets: X_train, y_train, X_test, y_test, X_valid, Y_valid
def linearReg(sets: []):
    X_train, y_train, X_test, y_test, X_valid, Y_valid = [a for a in sets]
    print(X_train)
    print(y_train)
    # reg = LinearRegression().fit(X, y)