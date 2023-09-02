import os
import pandas as pd


def dataset_to_X_y(filename):
    file_extention = os.path.splitext(filename)[-1].lower()
    assert file_extention in ['.txt', '.csv'], 'unknown dataset resolution'
    data = pd.read_csv(f'datasets/{filename}')
    X = data.loc[:, data.columns != 'Outcome'].to_numpy()
    y = data.loc[:, 'Outcome'].to_numpy()
    return X, y
