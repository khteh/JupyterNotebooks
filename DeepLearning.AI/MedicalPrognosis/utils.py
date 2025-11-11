import numpy as np
import pandas as pd
import lifelines
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def cindex(y_true, scores):
    return lifelines.utils.concordance_index(y_true, scores)

def generate_data(n=200):
    df = pd.DataFrame(
        columns=['Age', 'Systolic_BP', 'Diastolic_BP', 'Cholesterol']
    )
    df.loc[:, 'Age'] = np.exp(np.log(60) + (1 / 7) * np.random.normal(size=n))
    df.loc[:, ['Systolic_BP', 'Diastolic_BP', 'Cholesterol']] = np.exp(
        np.random.multivariate_normal(
            mean=[np.log(100), np.log(90), np.log(100)],
            cov=(1 / 45) * np.array([
                [0.5, 0.2, 0.2],
                [0.2, 0.5, 0.2],
                [0.2, 0.2, 0.5]]),
            size=n))
    return df

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def f(x):
    p = 0.4 * (np.log(x[0]) - np.log(60)) + 0.33 * (
            np.log(x[1]) - np.log(100)) + 0.3 * (
                np.log(x[3]) - np.log(100)) - 2.0 * (
                np.log(x[0]) - np.log(60)) * (
                np.log(x[3]) - np.log(100)) + 0.05 * np.random.logistic()
    if p > 0.0:
        return 1.0
    else:
        return 0.0

def prob_drop(age):
    return 1 - (np.exp(0.25 * age - 5) / (1 + np.exp(0.25 * age - 5)))


def nhanesi(data_x, data_y, display=False):
    """Same as shap, but we use local data."""
    X = pd.read_csv(data_x)
    y = pd.read_csv(data_y)["y"]
    if display:
        X_display = X.copy()
        X_display["Sex"] = ["Male" if v == 1 else "Female" for v in X["Sex"]]
        return X_display, np.array(y)
    return X, np.array(y)

def load_data(n=200):
    np.random.seed(0)
    df = generate_data(n)
    for i in range(len(df)):
        df.loc[i, 'y'] = f(df.loc[i, :])
        X = df.drop('y', axis=1)
        y = df.y
    return X, y

def load_data(threshold, data_x, data_y):
    X, y = nhanesi(data_x, data_y)
    df = X.drop([X.columns[0]], axis=1)
    df.loc[:, 'time'] = y
    df.loc[:, 'death'] = np.ones(len(X))
    df.loc[df.time < 0, 'death'] = 0
    df.loc[:, 'time'] = np.abs(df.time)
    df = df.dropna(axis='rows')
    mask = (df.time > threshold) | (df.death == 1)
    df = df[mask]
    X = df.drop(['time', 'death'], axis='columns')
    y = df.time < threshold

    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    feature_y = 'Systolic BP'
    frac = 0.7

    drop_rows = X_dev.sample(frac=frac, replace=False,
                             weights=[prob_drop(X_dev.loc[i, 'Age']) for i in
                                      X_dev.index], random_state=10)
    drop_rows.loc[:, feature_y] = None
    drop_y = y_dev[drop_rows.index]
    X_dev.loc[drop_rows.index, feature_y] = None

    return X_dev, X_test, y_dev, y_test
