import keras
from keras.layers import Dense, Input, \
    MaxPooling2D, Conv2D, Flatten,\
    Dropout, BatchNormalization

from keras.layers import Rescaling

def mlp_model() -> keras.models.Sequential:
    """A basic multilayer perceptron.

    Returns:
        model: a tensorflow model. 
    """
    model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=10, activation="softmax")])
    return model



def cnn_model() -> keras.models.Sequential:
    """Returns a convolution neural net model.
    """
    model_name = 'convolution_network_model'
    model = keras.models.Sequential([
        Rescaling(1./127.5, offset=-1, name='rescaling_0_1'),
        Conv2D(input_shape=(28, 28, 1),
               kernel_size=(3, 3),
               filters=128,
               strides=2,
               padding='same', 
               activation='relu', name='conv2d_layer_1'),
       BatchNormalization(name='batch_norm_1'),
       MaxPooling2D(name='max_pool_1'),
       Conv2D(input_shape=(28, 28, 1),
               kernel_size=(3, 3),
               filters=64,
               strides=2,
               padding='same', 
               activation='relu', name='conv2d_layer_2'),
        BatchNormalization(name='batch_norm_2'),
        MaxPooling2D(name='max_pool_2'),
        # Conv2D(input_shape=(28, 28, 1),
        #        kernel_size=(3, 3),
        #        filters=32,
        #        strides=2,
        #        padding='same', 
        #        activation='relu', name='conv2d_layer_3'),
        # BatchNormalization(name='batch_norm_3'),
        Flatten(),
        Dense(units=10, activation='softmax', name='output_layer')

    ], name=model_name) 
    return model



if __name__ == '__main__':
    
    task_name = "cnn"
    all_models = {"cnn": cnn_model(), "mlp": mlp_model()}
    model = all_models.get(task_name)
    input_shape = (None, 28, 28, 1)
    model.build(input_shape)
    print(model.summary())
