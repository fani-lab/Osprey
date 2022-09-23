import os
import numpy as np
import tensorflow as tf
from numpy import newaxis
from tensorflow.python.keras import Sequential, Input
# # from tensorflow.pyt
# hon.keras.callbacks import Callback
from tensorflow.python.keras.layers import LSTM, Dense, CuDNNLSTM
from tensorflow.python.keras.models import Model

from tftest import get_train_test_data, Getvalues
from rnn_binary import get_train_test_data as gtt



corpus_training_file = 'D:/data/train/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml'
corpus_training_predator_id_file = 'D:/data/train/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt'
dummy_training = 'D:/data/train/traindummy.xml'

corpus_test_file = 'D:/data/test/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml'
corpus_test_predator_id_file = 'D:/data/test/pan12-sexual-predator-identification-groundtruth-problem1.txt'

# X_train, y_train = get_train_test_data(dummy_training, corpus_training_predator_id_file)
# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# np.save('X_train.npy', X_train)
# np.save('y_train.npy', y_train)

X_test, y_test = get_train_test_data(dummy_training, corpus_test_predator_id_file)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

X_train = X_test
y_train = y_test

# Save data
# np.save('X_train.npy', X_train)
# np.save('y_train.npy', y_train)
# np.save('X_test.npy', X_test)
# np.save('y_test.npy', y_test)

# Load data
# X_train = np.load('X_train.npy')
# y_train = np.load('y_train.npy')
# X_test = np.load('X_test.npy')
# y_test = np.load('y_test.npy')


# define LSTM

# model = Sequential()
# model.add(CuDNNLSTM(1, return_sequences=True,))
# model.add(Dense(1, activation='tanh'))

# visible = Input(shape=(300,1))
# hidden1 = LSTM(1, return_sequences=True, return_state= True)(visible)
# output = Dense(1, activation='tanh')(hidden1)
# model = Model(inputs=visible, outputs=output)

# inputs = Input(shape=(300,1))
# lstm = tf.keras.layers.LSTM(4, return_sequences=True, return_state=True)
# output = lstm(inputs)
# # print(output.shape)
#
# lstm = tf.keras.layers.LSTM(4, return_sequences=True, return_state=True)
# whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
# # print(whole_seq_output.shape)
# model = Model(inputs=inputs, outputs=output)



model = Sequential()
model.add(LSTM(10, return_sequences=True))
model.add(Dense(1, activation='sigmoid'))

# model.add(LSTM(5, input_shape=(1, 1), return_sequences=True))







# model = tf.keras.Sequential([
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])
# x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))
# model = tf.keras.layers.Dense(64, activation='relu')(x)
# model = tf.keras.layers.Dense(1)(model)


# MODEL = 'RNN'
# NUM_UNITS = 5
#
# inp = layers.Input(shape=(n_timesteps,))
# out = layers.Embedding(input_dim=vocab_size, output_dim= EMBEDDING_DIM, input_length= n_timesteps)(inp)
# if MODEL == 'GRU':
#     out, state = layers.CuDNNGRU(NUM_UNITS, return_state=True)(out)
# if MODEL == 'RNN':
#     out, state = layers.SimpleRNN(NUM_UNITS, return_state=True)(out)
# if MODEL == 'LSTM':
#     out, state = layers.CuDNNLSTM(NUM_UNITS, return_state=True)(out)
# out = layers.Dense(1, activation='softmax')(out)
# model = Model(inputs=inp, outputs=[out, state])
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])







model.compile(loss="mean_absolute_error", optimizer="adam", metrics=['accuracy'])
print("Starting to train Model please wait..")

model.fit(X_train, y_train, epochs=5, batch_size=500, verbose=1)
test_loss, test_acc = model.evaluate(X_test, y_test)
model.save('model.hdf5')
# Sensitivity, Specificity, PPV, NPV, F1_score = Getvalues(conf_mat)
# print("Acuuracy : " + str(round(test_acc, 2)) + " Sensitivity : " + str(Sensitivity) +
#       " Specificity: " + str(Specificity) + " PPV: " + str(PPV) + " NPV: " + str(NPV) + " F1_score: " + str(F1_score))

print('Test accuracy:', test_acc)
y_pred = model.predict(X_test,verbose =1)
y_pred_bool = np.argmax(y_pred, axis=1)
