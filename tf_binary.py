import os
import numpy as np
import tensorflow as tf
from numpy import newaxis
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import LSTM
from tftest import get_train_test_data, Getvalues

# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(session)

# gpu = tf.config.experimental.list_physical_devices('GPU')[0]
# tf.config.experimental.set_memory_growth(gpu, True)

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


class NBatchLogger(Callback):
    """
    A Logger that log average performance per `display` steps.
    """
    def __init__(self, display):
        self.step = 0
        self.display = display
        self.metric_cache = {}

    def on_batch_end(self, batch, logs={}):
        self.step += 1
        for k in self.params['metrics']:
            if k in logs:
                self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]
        if self.step % self.display == 0:
            metrics_log = ''
            for (k, v) in self.metric_cache.items():
                val = v / self.display
                if abs(val) > 1e-3:
                    metrics_log += ' - %s: %.4f' % (k, val)
                else:
                    metrics_log += ' - %s: %.4e' % (k, val)
            print('step: {}/{} ... {}'.format(self.step,
                                          self.params['steps'],
                                          metrics_log))
            self.metric_cache.clear()


corpus_training_file = 'D:/data/train/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml'
corpus_training_predator_id_file = 'D:/data/train/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt'
dummy_training = 'D:/data/train/traindummy.xml'

corpus_test_file = 'D:/data/test/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml'
corpus_test_predator_id_file = 'D:/data/test/pan12-sexual-predator-identification-groundtruth-problem1.txt'

X_train, y_train = get_train_test_data(corpus_training_file, corpus_training_predator_id_file)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)


# X_train = X_train[:, newaxis]
# X_train = tf.ragged.constant(X_train)
# X_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# print(X_train.output_shapes)


X_test, y_test = get_train_test_data(corpus_test_file, corpus_test_predator_id_file)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# X_test = X_test[:, newaxis]
# X_test = tf.ragged.constant(X_test)
# X_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Save data
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

# Load data
# X_train = np.load('X_train.npy')
# y_train = np.load('y_train.npy')
# X_test = np.load('X_test.npy')
# y_test = np.load('y_test.npy')


# define LSTM

model = Sequential()
model.add(LSTM(1, return_sequences=True))
# model.add(LSTM(5, input_shape=(1, 1), return_sequences=True))
model.compile(loss="mean_absolute_error", optimizer="adam", metrics=['accuracy'])
print("Starting to train Model please wait..")
out_batch = NBatchLogger(display=300)

model.fit(X_train, y_train, epochs=20, batch_size=500, verbose=2)
test_loss, test_acc = model.evaluate(X_test, y_test)

# Sensitivity, Specificity, PPV, NPV, F1_score = Getvalues(conf_mat)
# print("Acuuracy : " + str(round(test_acc, 2)) + " Sensitivity : " + str(Sensitivity) +
#       " Specificity: " + str(Specificity) + " PPV: " + str(PPV) + " NPV: " + str(NPV) + " F1_score: " + str(F1_score))

print('Test accuracy:', test_acc)
