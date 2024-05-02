import os
import pandas as pd
import sklearn


def normalize_dataframe(df):
    return (df - df.mean()) / df.std()


def dataset_to_X_y(filename, nrows=None, normalize=False):
    file_extention = os.path.splitext(filename)[-1].lower()
    assert file_extention in ['.txt', '.csv'], 'unknown dataset resolution'
    data = pd.read_csv(f'datasets/{filename}', nrows=nrows)
    X = data.loc[:, data.columns[:-1]]
    y = data.loc[:, data.columns[-1]]
    if normalize:
        X = normalize_dataframe(X)
    return X.to_numpy(), y.to_numpy()


def load_libxvm(filename, n_features, normalize=False):
    data = sklearn.datasets.load_svmlight_file(f'datasets/{filename}')
    X, y = data[0], data[1]
    X = X.toarray()
    if normalize:
        X = normalize_dataframe(X)
    return X, y
