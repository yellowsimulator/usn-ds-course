import numpy as np
import tensorflow as tf
from datetime import datetime
from pathlib import Path
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix
from utility_functions import plt_plot_to_tf_image
from utility_functions import get_confusion_matrix_plot
from params import hyperparams

class ConfusionMatrixCallBack(Callback):
    def __init__(self, validation_data, output_dir, class_names):
        super(ConfusionMatrixCallBack, self)
        self.validation_data = validation_data
        self.output_dir = self.output_dir = Path(output_dir) / \
            datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        self.X = validation_data[0]
        self.y = validation_data[1]
        self.class_names = class_names
        self.writer = tf.summary.create_file_writer(str(self.output_dir))

    
    def on_train_end(self, logs=None):
        """Logs a matplotlib figure of a confusion metrix.
        """
        test_pred_proba = self.model.predict(self.X)
        test_pred_label = np.argmax(test_pred_proba, axis=1)
        conf_matrix = confusion_matrix(self.y, test_pred_label)
        cm_figure = get_confusion_matrix_plot(conf_matrix, \
                                          self.class_names)
        cm_image = plt_plot_to_tf_image(cm_figure)
        with self.writer.as_default():
            tf.summary.image("Confusion matrix",cm_image, \
                                step=hyperparams.get("epochs"))





