from tensorflow import keras
import numpy as np
from sklearn.metrics import classification_report

model = keras.models.load_model('model.hdf5')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
y_pred = model.predict(X_test, verbose=1)
np.save('y_pred.npy',y_pred)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))
