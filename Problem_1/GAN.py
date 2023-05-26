
# ======================================================================================================================

from __future__ import print_function, division
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys
import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow
import math
# ======================================================================================================================

class GAN():
  def __init__(self):
    self.Img_rows = 28
    self.Img_cols = 28
    self.channels = 1
    self.Img_shape = (self.Img_rows, self.Img_cols, self.channels)
    self.latent_dim = 100
		
		# Optimizer : Adam
    OPT = Adam(0.0002, 0.5)
    # Building and compiling the discriminator
    self.discriminator = self.Discriminator()
    self.discriminator.compile(loss='binary_crossentropy', optimizer=OPT, metrics=['accuracy'])

    # Building the generator
    self.generator = self.Generator()

    # Apply noise as input and generating images
    z = Input(shape=(self.latent_dim,))
    IMAGE = self.generator(z)

    # we only train the generator for the combined model
    self.discriminator.trainable = False
    # The discriminator determines validity of generated images
    validity = self.discriminator(IMAGE)

        
    # Training the generator to fool the discriminator
    self.combined = Model(z, validity)
    self.combined.compile(loss='binary_crossentropy', optimizer=OPT)
	#___________________________________Generator Function__________________________________
  # Building the generator serving multi-layer perceptron  
  def Generator(self):
    MODEL = Sequential()
    MODEL.add(Dense(256, input_dim=self.latent_dim))
    MODEL.add(LeakyReLU(alpha=0.2))
    MODEL.add(BatchNormalization(momentum=0.8))
    MODEL.add(Dense(512))
    MODEL.add(LeakyReLU(alpha=0.2))
    MODEL.add(BatchNormalization(momentum=0.8))
    MODEL.add(Dense(1024))
    MODEL.add(LeakyReLU(alpha=0.2))
    MODEL.add(BatchNormalization(momentum=0.8))
    MODEL.add(Dense(np.prod(self.Img_shape), activation='tanh'))
    MODEL.add(Reshape(self.Img_shape))
		
    MODEL.summary()

    noise = Input(shape=(self.latent_dim,))
    IMAGE = MODEL(noise)

    return Model(noise, IMAGE)
		
	#_________________________________Discriminator Function__________________________________
  # Building the Discriminator serving multi-layer perceptron  
  def Discriminator(self):
    MODEL = Sequential()

    MODEL.add(Flatten(input_shape=self.Img_shape))
    MODEL.add(Dense(512))
    MODEL.add(LeakyReLU(alpha=0.2))
    MODEL.add(Dense(256))
    MODEL.add(LeakyReLU(alpha=0.2))
    MODEL.add(Dense(1, activation='sigmoid'))
    MODEL.summary()

    IMAGE = Input(shape=self.Img_shape)
    validity = MODEL(IMAGE)

    return Model(IMAGE, validity)
		
	#_____________________________________Train Function_______________________________________
    
  def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (TRAINx, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        TRAINx = TRAINx / 127.5 - 1.
        TRAINx = np.expand_dims(TRAINx, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ________________________Discriminator Training__________________________
           

            # Select a random batch of images
            idx = np.random.randint(0, TRAINx.shape[0], batch_size)
            Imgs = TRAINx[idx]

            noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            ImgGen = self.generator.predict(noise)

            # Train the discriminator
            Real_loss_Dis = self.discriminator.train_on_batch(Imgs, valid)
            Fake_loss_Dis = self.discriminator.train_on_batch(ImgGen, fake)
            loss_Dis = 0.5 * np.add(Real_loss_Dis, Fake_loss_Dis)

            
            # ___________________________Generator Training__________________________
            

            noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            loss_Gen = self.combined.train_on_batch(noise, valid)


            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                
                print('Epoch#{}'.format(epoch))
                self.save_sample(ImgGen, batch_size)
	
	#___________________________________________________________________________________
  
  
  def save_sample(self,imagearray, batch_size):
    width = imagearray.shape[2]
    height = imagearray.shape[1]
    if (imagearray.shape[3] == 1):
        mode = 'L'
        imagearray = imagearray[:, :, :, 0]
    else:
        mode = 'RGB'
    
    num_elements = int(math.sqrt(batch_size))
    imagegrid = Image.new(mode, (width * num_elements, height * num_elements))
    for j in range(num_elements * num_elements):
        randimg = imagearray[j] * 127.5 + 127.5
        img = Image.fromarray(randimg.astype('uint8'), mode=mode)
        imagegrid.paste(im=img, box=((j % num_elements) *width, height * (j // num_elements)))
        
    plt.imshow(imagegrid, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show();
	
		

# ====================================================_MAIN_=======================================================		
EPOCHS = 30000
batch_size = 120
Sample_Interval = 1000

if __name__ == '__main__':
  
  GAN().train(epochs=EPOCHS, batch_size=batch_size, sample_interval=Sample_Interval)