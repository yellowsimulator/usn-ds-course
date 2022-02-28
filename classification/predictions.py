import keras
import numpy as np
import matplotlib.pyplot as plt

from data import get_train_test_data
from params import model_path, class_name_fashion_mnist



train_X, train_y, test_X, test_y = get_train_test_data()
model = keras.models.load_model(model_path)
y_pred = model.predict(test_X)


def get_predictions(row: int, column: int):
    """Return row*column predictions

    Parameters
    ----------
    row: the number of rows
    column: the number of columns
    """
    items = row*column
    plt.figure(figsize=(10, 10))
    for i in range(0, items):
        plt.subplot(row, column, i+1)
        plt.imshow(test_X[i])
        label = np.argmax(y_pred[i])
        item = class_name_fashion_mnist[label]
        plt.title(f"Pred: {item}")
        plt.xticks([])
    plt.show()



if __name__ == '__main__':
    get_predictions(5, 5)