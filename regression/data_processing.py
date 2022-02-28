import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_data() -> pd.DataFrame:
    """Returns the data as a pandas DataFrame.
    """
    df = pd.read_excel('data.xlsx')
    return df


def get_train_test(test_size: float = 0.2, \
                    scale: bool=False, \
                    name: str='min_max') -> tuple:
    """Returns the train and test data.

    Parameters:
    -----------
    test_size: size of the test data.
    scale: whether to scale the data.
    name: name of the preprocessing method.
    """
    df = get_data()
    y = df.loc[:, 'Y1': 'Y2'].values
    X = df.loc[:, 'X1': 'X8'].values
    X_train, X_test, y_train, y_test = train_test_split(X, \
                 y, test_size=test_size, random_state=42)
    if scale:
        X_train, X_test = preprocess_data(X_train, X_test,\
                         name)

    return X_train, X_test, y_train, y_test


def preprocess_data(X_train: np.ndarray, \
                    X_test: np.ndarray, \
                    name: str) -> np.ndarray:
    """Returns a preprocessed data.

    Paramerters:
    -----------
    X: input data.
    name: name of the preprocessing method.
           either 'standard' or 'min_max'.

    Returns:
    --------
    X_preprocessed: preprocessed data.
    """
    
    scalers = {'min_max': MinMaxScaler(), \
               'standard': StandardScaler()}
    scaler = scalers.get(name)
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
    

if __name__ == '__main__':
    ...
