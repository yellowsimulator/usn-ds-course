from datetime import datetime
from keras.losses import SparseCategoricalCrossentropy

# tensorboard log directory
log_dir = "logs/fit/" + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
model_path = "model_repo/model1"
# hyperparameters
hyperparams = {
        'input_shape': (28, 28),
        'units_layer_1': 512,
        'units_layer_2': 512,
        'units_last_layer': 10,
        'epochs': 20,
        'optimizer': 'adam',
        'metric': 'accuracy',
        'loss': SparseCategoricalCrossentropy(from_logits=True)
    }

class_name_fashion_mnist = [
             'T-shirt/top', 'Trouser', 
             'Pullover', 'Dress', 'Coat',
             'Sandal', 'Shirt', 'Sneaker', 
             'Bag', 'Ankle boot']

def label_name(label_id):
    label_map = {
        k:class_name_fashion_mnist[k]
        for k in range(len(class_name_fashion_mnist))
    }
    return label_map[label_id]