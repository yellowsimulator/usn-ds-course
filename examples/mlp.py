import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import Input, Model


def mpl_sequential_api_v1() -> Sequential:
    """Implements a sequential api.
    """
    model = Sequential([
        Dense(units=64, activation='relu', input_shape=(784,)),
        ])
    return model


def mlp_functional_api(input_shape: tuple) -> Model:
    """Implements a functional api.
    """
    input_image = Input(shape=input_shape)



if __name__ == '__main__':
    
    model = mlp_functional_api()
    print(model.summary())

    tf.keras.utils.plot_model(model, to_file='model.png')
