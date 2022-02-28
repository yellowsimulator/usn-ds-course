import json
import numpy as np
from webbrowser import get
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from data_processing import get_train_test

model_name = "power_plant_model"
log_dir = 'logs/fit' # for tensorboard
model_path = f"model_repository/{model_name}"

# get the data
X_train, X_test, y_train, y_test = get_train_test(scale=True)


def get_model() -> keras.Model:
    """Returns the model.

    Returns:
        model: a keras model model.
    """
    model_name = "power_plant_model"
    model_path = f"model_repository/{model_name}"
    model = keras.models.load_model(model_path)
    return model


def get_model_history() -> dict:
    """Returns the model history.

    Returns:
        model_history: a dictionary 
        containing the model history.
    """
    with open("model_history.json", "r") as f:
        model_history = json.load(f)
    return model_history


def plot_true_vs_predicted(y_true: np.ndarray, \
           y_pred: np.ndarray, title: str=''):
    """Plots the true vs predicted values.
    Parameters
    ----------
    y_true : true values
    y_pred : predicted values
    """
    plt.scatter(y_true, y_pred)
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    plt.plot([-100, 100], [-100, 100])
    plt.show()


def plot_metrics(metric_name: str, title: str, ylim=5):
    """Plots different metrics

    Parameters
    ----------
    metric_name: the name of the metric to plot
    title: the title of the plot
    ylim (int, optional):Defaults to 5.
    """
    model_history = get_model_history()
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(model_history[metric_name], color='blue', label=metric_name)
    plt.plot(model_history['val_' + metric_name], color='green', label='val_' + metric_name)
    plt.show()


if __name__ == '__main__':
    model_history = get_model_history()
    print(model_history.keys())
    # model = keras.models.load_model(model_path)
    # y_pred = model.predict(X_test)

    # plot_true_vs_predicted(y_test[..., 0], y_pred[0], title='Y1')
    # plot_true_vs_predicted(y_test[..., 1], y_pred[1], title='Y2')
    # plot_metrics('y1_output_root_mean_squared_error', 'Y1')
    # plot_metrics('y2_output_root_mean_squared_error', 'Y2')