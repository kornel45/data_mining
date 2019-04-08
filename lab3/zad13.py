import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from keras import models
from keras.backend import tensorflow_backend
from keras.layers import Dropout
from keras.utils import to_categorical
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

train = fetch_20newsgroups(subset='train', shuffle=True)
test = fetch_20newsgroups(subset='test', shuffle=True)

config = tf.ConfigProto(device_count={"CPU": 8})
tensorflow_backend.set_session(tf.Session(config=config))

vec_pipe = Pipeline([
    ('cv', CountVectorizer(ngram_range=(1, 2), max_df=0.7, max_features=500000)),
    ('tfidf', TfidfTransformer(use_idf=True))
])

n = 8000

x_train = vec_pipe.fit_transform(train.data)
x_test = vec_pipe.transform(test.data)

y_train = to_categorical(train.target)
y_test = to_categorical(test.target)

network = models.Sequential()
network.add(layers.Dense(256, activation='relu', input_shape=(x_train.shape[1],)))
network.add(Dropout(0.6))
network.add(layers.Dense(64, activation='relu'))
network.add(Dropout(0.5))
network.add(layers.Dense(20, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print(network.summary())
history = network.fit(x_train, y_train, epochs=7, batch_size=256, validation_data=(x_test, y_test))

predicted = network.predict(x_test).argmax(axis=-1)
print('Accuracy on validation set: {}'.format(np.mean(predicted == test.target)))


loss_values = history.history['loss']
val_loss_values = history.history['val_loss']
epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc_values = history.history['acc']
val_acc_values = history.history['val_acc']
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
