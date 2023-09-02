import os
import pandas as pd


def dataset_to_X_y(filename, nrows=None):
    file_extention = os.path.splitext(filename)[-1].lower()
    assert file_extention in ['.txt', '.csv'], 'unknown dataset resolution'
    data = pd.read_csv(f'datasets/{filename}', nrows=nrows)
    X = data.loc[:, data.columns[:-1]].to_numpy()
    y = data.loc[:, data.columns[-1]].to_numpy()
    return X, y
