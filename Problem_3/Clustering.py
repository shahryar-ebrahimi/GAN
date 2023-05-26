
# =======================================================================================================================
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import UpSampling2D
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Reshape, Flatten, Lambda, Conv2DTranspose
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from tensorflow.contrib.factorization.python.ops import clustering_ops

# =======================================================================================================================
(TRAINx, TRAINy), (TESTx, TESTy) = mnist.load_data()

TRAINx = TRAINx.reshape(TRAINx.shape[0], 28, 28, 1) #transform 2D 28x28 matrix to 3D (28x28x1) matrix
TESTx = TESTx.reshape(TESTx.shape[0], 28, 28, 1)

TRAINx = TRAINx.astype('float32')
TESTx = TESTx.astype('float32')

TRAINx /= 255 #inputs have to be between [0, 1]
TESTx /= 255


# ====================================================Building Model======================================================
model = Sequential()
 
#1st convolution layer
model.add(Conv2D(16, (3, 3) #16 is number of filters and (3, 3) is the size of the filter.
, padding='same', input_shape=(28,28,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
 
#2nd convolution layer
model.add(Conv2D(2,(3, 3), padding='same')) # apply 2 filters sized of (3x3)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
 
#here compressed version
 
#3rd convolution layer
model.add(Conv2D(2,(3, 3), padding='same')) # apply 2 filters sized of (3x3)
model.add(Activation('relu'))
model.add(UpSampling2D((2, 2)))
 
#4th convolution layer
model.add(Conv2D(16,(3, 3), padding='same'))
model.add(Activation('relu'))
model.add(UpSampling2D((2, 2)))
 
model.add(Conv2D(1,(3, 3), padding='same'))
model.add(Activation('sigmoid'))

model.summary()

# =======================================================================================================================
# Fitting and optimizing...

EPOCHs = 3
model.compile(optimizer='adadelta', loss='binary_crossentropy')
model.fit(TRAINx, TRAINx, epochs=EPOCHs, validation_data=(TESTx, TESTx))
# =======================================================================================================================

compressed_layer = 5
get_3rd_layer_output = K.function([model.layers[0].input], [model.layers[compressed_layer].output])
compressed = get_3rd_layer_output([TESTx])[0]
 
#flatten compressed representation to 1 dimensional array
compressed = compressed.reshape(10000,7*7*2)

# =========================================================Trainer===================================================
def Trainer():
  data = tf.constant(compressed, tf.float32)
  return (data, None)
 
unsupervised_model = tf.contrib.learn.KMeansClustering(
10 #num of clusters
, distance_metric = clustering_ops.SQUARED_EUCLIDEAN_DISTANCE
, initial_clusters=tf.contrib.learn.KMeansClustering.RANDOM_INIT
)
 
unsupervised_model.fit(input_fn=Trainer, steps=1000)

# =======================================================================================================================
# Plotting the clusters ...
clusters = unsupervised_model.predict(input_fn=Trainer)
 
index = 0
for i in clusters:
  current_cluster = i['cluster_idx']
  features = TESTx[index]
  if index < 200 and current_cluster == 3:
    plt.imshow(TESTx[index].reshape(28, 28))
    plt.gray()
    plt.show()
    plt.axis('off')
  index = index + 1

