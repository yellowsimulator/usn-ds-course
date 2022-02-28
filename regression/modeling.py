
import os
import json
import shutil
import mlflow
import sys
import tensorflow as tf
from keras.callbacks import TensorBoard
log_dir = 'logs'
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
from models import model
from data_processing import get_train_test


experiment = 1
#mlflow.set_experiment(f"power_{experiment}")
#mlflow.start_run(run_name=f"run_{experiment}")

model_name = "power_plant_model"
log_dir = 'logs/fit' # for tensorboard
model_path = f"model_repository/{model_name}"


# get the model and the data
X_train, X_test, y_train, y_test = get_train_test(scale=True)
model = model(input_shape=X_train.shape[1])

# compile the model, by setting up the optimisation problem
# set up training configuration
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss={'y1_output': 'mse', 'y2_output': 'mse'},
                metrics={'y1_output': tf.keras.metrics.RootMeanSquaredError(),
                         'y2_output': tf.keras.metrics.RootMeanSquaredError()})
             


# fit the model. Forward and back propagation.
#mlflow.tensorflow.autolog()
tensorboar_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
history = model.fit(x=X_train,
          y=y_train,
          epochs=50,
          batch_size=30,
          validation_data=(X_test, y_test),
          callbacks=[tensorboar_callback])



#mlflow.end_run()
# save the model and model history
model.save(model_path)
model_history = history.history
with open("model_history.json", "w") as f:
    json.dump(model_history, f)

