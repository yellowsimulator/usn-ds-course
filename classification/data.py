from keras.datasets import fashion_mnist



def get_train_test_data():
    """Returns train test data
    """
    (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
    return train_X, train_y, test_X, test_y