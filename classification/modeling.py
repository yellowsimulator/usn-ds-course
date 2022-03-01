import numpy as np
import tempfile
import os
import shutil
import tensorflow as tf
import mlflow
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.models import save_model
from tf_explain.callbacks.grad_cam import GradCAMCallback
# Local libraries
from params import log_dir, hyperparams, model_path
from params import  class_name_fashion_mnist
from call_backs import ConfusionMatrixCallBack
from data import get_train_test_data
from models import mlp_model, cnn_model



def train_model(model: tf.keras.models,
                task_name: str,
                class_names: list,
                hyperparams: dict):
    """Trains a sequential model.

    Args:
        model: our sequential model.
        task_name: the name of the task.
        class_names: class label 
        hyperparams: dictionary of hyperparameters
        data_name: dataset name. Defaults to 'mnist'.
    """
    #mlflow.set_experiment(f"{task_name}")
    #mlflow.start_run(run_name=f"run_{task_name}")
    MODEL_DIR = 'models_repo'
    version = 1
    export_path = os.path.join(MODEL_DIR, str(version))

    train_X, train_y, test_X, test_y = get_train_test_data()
    train_X = train_X.reshape(train_X.shape[0], 28, 28, 1)
    test_X = test_X.reshape(test_X.shape[0], 28, 28, 1)
    mlflow.tensorflow.autolog()
    # compile the model
    model.compile(
        loss=hyperparams['loss'], 
        optimizer=hyperparams['optimizer'],
        metrics=[hyperparams['metric']]
    )
    # call backs setup
    tensorboar_callback = TensorBoard(log_dir=log_dir)
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10)
    confusion_matrix_callback = ConfusionMatrixCallBack(
        validation_data=(train_X, train_y),
        output_dir=f'logs/fit/confusion_matrix',
        class_names=class_names)
    grad_cam_callback = GradCAMCallback(
        validation_data=(test_X, test_y),
        layer_name='batch_norm_2',
        class_index=0,
        output_dir='logs/fit/grad_cam')
    # fit model
    model.fit(x=train_X,
              y=train_y, 
              epochs=hyperparams['epochs'],
              validation_data=(test_X, test_y),
              callbacks=[tensorboar_callback, 
                         confusion_matrix_callback,
                         early_stop_callback])
    # save model
    model.save(model_path)
    #mlflow.end_run

    
    

def run_training(model: tf.keras.models,
                 task_name: str, 
                 hyperparams: dict):
    """Runs training."""
    class_names = class_name_fashion_mnist
    train_model(model, task_name, class_names, hyperparams,)



if __name__ == '__main__':
    #model = base_model()
    #model = cnn_model()
    task_name = "cnn"
    all_models = {"cnn": cnn_model(), "mlp": mlp_model()}
    model = all_models.get(task_name)
    run_training(model, task_name, hyperparams)
