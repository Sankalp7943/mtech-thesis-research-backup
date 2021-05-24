#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

original_model = models.Sequential()
original_model.add(layers.Dense(512, activation='relu', input_shape=(10000,)))
original_model.add(layers.Dense(512, activation='relu'))
original_model.add(layers.Dense(1, activation='sigmoid'))

object_mliiitl = mliiitl_main.mliiitl(x_train, y_train,x_test,y_test,original_model,'binary_crossentropy',5,512)
mliiitl_main.mliiitl.test_performance(object_mliiitl,True, True)