# unet model with a convolutional neural network
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model



def model(input_shape):
    """Returns a functional model.

    Args:
        input_shape (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Define model layers.
    input_layer = Input(shape=(input_shape,))
    layer1 = Dense(units='128', activation='relu')(input_layer)
    layer2 = Dense(units='128', activation='relu')(layer1)
    # Y1 output will be fed directly from the second dense
    y1_output_layer = Dense(units='1', name='y1_output')(layer2)
    layer3 = Dense(units='64', activation='relu')(layer2)
    # Y2 output will come via the third dense
    y2_output_layer = Dense(units='1', name='y2_output')(layer3)
    # Define the model with the input layer and a list of output layers
    model = Model(inputs=input_layer, outputs=[y1_output_layer,\
                                                y2_output_layer])
    return model