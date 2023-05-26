
# =======================================================================================================================

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Reshape, Flatten, Lambda, Conv2DTranspose
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist

# ======================================================================================================================
#____________________________________LOADING____________________________________

(TRAINx, TRAINy), (TESTx, TESTy) = mnist.load_data()
TRAINx = TRAINx.astype('float32') / 255.
TRAINx = TRAINx.reshape(-1,28,28,1)

TESTx = TESTx.astype('float32') / 255.
TESTx = TESTx.reshape(-1,28,28,1)
print(TRAINx.shape, TESTx.shape)


# __________________________________ENCODER_____________________________________
# Create encoder network

latent_dim = 2 # dimension of latent variable

inputs = Input(shape = (28,28,1))
conv1 = Conv2D(16, (3,3), activation = 'relu', padding = "SAME")(inputs)
conv1_1 = Conv2D(16, (3,3), activation = 'relu', padding = "SAME")(conv1)
pool1 = MaxPooling2D(pool_size = (2,2), strides = 2)(conv1_1)
conv2 = Conv2D(32, (3,3), activation = 'relu', padding = "SAME")(pool1)
pool2 = MaxPooling2D(pool_size = (2,2), strides = 2)(conv2)

flat = Flatten()(pool2)
input_to_z = Dense(32, activation = 'relu')(flat)
mu = Dense(latent_dim, name='mu')(input_to_z)
sigma = Dense(latent_dim, name='log_var')(input_to_z)

encoder = Model(inputs, mu)
encoder.summary()

#-------------------------------------------
# sampling data to pass the decoder as input
def sampling(args):
    mu, sigma = args
    epsilon = K.random_normal(shape=(K.shape(mu)[0], latent_dim),
                              mean=0.0, stddev=1.0)
    return mu + K.exp(0.5*sigma) * epsilon

z = Lambda(sampling)([mu, sigma])


# _________________________________DECODER______________________________________
 
#creating the decoder network... 

decoder_inputs = Input(K.int_shape(z)[1:])
dense_layer_d = Dense(7*7*32, activation = 'relu')(decoder_inputs)
output_from_z_d = Reshape((7,7,32))(dense_layer_d)
DeConv1 = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(output_from_z_d)
DeConv2 = Conv2DTranspose(16, 3, padding='same', activation='relu', strides=(2, 2))(DeConv1)
DeConv3 = Conv2DTranspose(1, 3, padding='same', activation='relu')(DeConv2)

decoder = Model(decoder_inputs, DeConv3)
decoder.summary()
z_decoded = decoder(z)


# __________________________________
#calculate reconstruction loss and KL divergence

class Diverging(keras.layers.Layer):

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)

        rec_loss = keras.metrics.binary_crossentropy(x, z_decoded)  # reconstruction loss
        kl_loss = 5e-4 * K.sum( K.square(mu) + K.exp(sigma) - sigma - 1, axis=-1)  # KL loss
        return K.mean(rec_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

outputs = Diverging()([inputs, z_decoded])


vae = Model(inputs, outputs)

# _____________________________________TRAINING_________________________________________

BATCH = 128
n_epoch = 3
vae.compile(optimizer='adam', loss=None)
results = vae.fit(TRAINx, epochs=n_epoch, batch_size=BATCH, shuffle=True, validation_data=(TESTx, None))


# ==============================================================
n = 10  # figure with nxn digits

z_mean = encoder.predict(TESTx, batch_size=BATCH)
plt.figure(figsize=(10, 10))
plt.scatter(z_mean[:, 0], z_mean[:, 1], c=TESTy)
plt.colorbar()
plt.xlabel("Feature_1")
plt.ylabel("Feature_2")
plt.title('New Feature Space')
plt.show()


#_____________________________________________RESULTS___________________________________________

dig_num = 28  # Size of the generated image
figure = np.zeros((dig_num * n, dig_num * n))

grid_x = np.linspace(-1, 1, n)
grid_y = np.linspace(-1, 1, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]]) * 1.
        x_decoded = decoder.predict(z_sample)
        
        digit = x_decoded[0].reshape(dig_num, dig_num)
        figure[i * dig_num: (i + 1) * dig_num,
               j * dig_num: (j + 1) * dig_num] = digit

# ----------------------------------------------------------------------------------------			   
plt.figure(figsize=(8, 8))
plt.imshow(figure)
plt.axis('off')
plt.title('Generated Numbers by VAE')
plt.show()
# ----------------------------------------------------------------------------------------
plt.figure(figsize=(8, 8))
plt.plot(results.epoch, np.array(results.history['val_loss']), label='Test Loss')
plt.plot(results.epoch, np.array(results.history['loss']), label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss of Variational Auto-encoder Model')
plt.show()