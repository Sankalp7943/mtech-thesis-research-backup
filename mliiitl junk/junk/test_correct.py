from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import mliiitl_main
 
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
 
def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results
 
# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)
# Our vectorized labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#rmsprop


original_model = models.Sequential()
original_model.add(layers.Dense(512, activation='relu', input_shape=(10000,)))
original_model.add(layers.Dense(512, activation='relu'))
original_model.add(layers.Dense(1, activation='sigmoid'))

original_model.compile(optimizer='rmsprop',
                       loss='binary_crossentropy',
                       metrics=['acc'])
original_hist = original_model.fit(x_train, y_train,
                                   epochs=5,
                                   batch_size=128,
                                   validation_data=(x_test, y_test))
epochs = range(1, 21)
original_val_loss = original_hist.history['val_loss']
plt.figure(figsize=(15,10))
plt.ylim(0, 1)

plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()