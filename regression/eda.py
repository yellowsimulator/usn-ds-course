import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from data_processing import get_data, get_train_test

df = get_data()


def plot_the_data():
    """Plots the entile dataset.
    """
    df.plot()
    plt.show()


def plot_correlation_matrix():
    """Plots the correlation matrix."""
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()


def plot_input_vs_target(X: str, Y: str):
    """Plots an input X versus an output Y.
    Parameters
    ----------
    X : input
    Y : output
    """
    df.plot(kind='line', x=X, y=Y, alpha=0.5)
    plt.show()


if __name__ == '__main__':
    # plot_input_vs_target('X4', 'Y1')
    # plot_correlation_matrix()
    ...



