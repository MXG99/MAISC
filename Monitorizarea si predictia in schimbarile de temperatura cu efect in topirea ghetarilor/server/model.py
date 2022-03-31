import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(X: pd.Dataframe, Y: pd.Dataframe):

    X_train, X_rem, Y_train, Y_rem = train_test_split(X, Y, train_size=4 / 5, random_state=42)
    X_valid, X_test, Y_valid, Y_test = train_test_split(X_rem, Y_rem, test_size=1 / 3, random_state=42)

    