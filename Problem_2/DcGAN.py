
# =======================================================================================================================

import numpy as np
import math
import random
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation,Dropout
from keras.layers.core import Dense, Reshape, Flatten
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal
from keras.datasets import cifar10
from keras.models import Sequential
from keras.optimizers import Adam
from PIL import Image


# ======================================================================================================================
#------------------------------------------------
batch_size=128
EPOCHS=60
# -----------------------------------------------

class DCGAN():
  def __init__(self, img_dims):
    
    self.img_row = img_dims[0]
    self.img_col = img_dims[1]
    self.channels = img_dims[2]
    self.latent_dim = 100
		
    if(self.channels == 3):
      self.downsize_factor = 3
    else:
      self.downsize_factor = 2
	
	# __________________________________________Deconvolution________________________________________
  def DeConv(self, MODEL, out_channels):
    MODEL.add(Conv2DTranspose(out_channels, [5, 5], strides=(
        2, 2), padding='same', kernel_initializer=RandomNormal(stddev=0.02)))
    MODEL.add(BatchNormalization())
    MODEL.add(LeakyReLU(alpha=0.2))
    return MODEL		
	# __________________________________________CONVOLUTION__________________________________________
  
  def Convolution(self, MODEL, out_channels):
    MODEL.add(Conv2D(out_channels, (5, 5),
                     kernel_initializer=RandomNormal(stddev=0.02)))
    MODEL.add(BatchNormalization())
    MODEL.add(LeakyReLU(alpha=0.2))
    return MODEL
	
	# __________________________________________GENERATOR___________________________________________
    
  def Generator(self):
    scale = 2**self.downsize_factor
    MODEL = Sequential()
    MODEL.add(Dense(self.img_row // scale * self.img_col // scale * 1024,input_dim=self.latent_dim, kernel_initializer=RandomNormal(stddev=0.02)))
    MODEL.add(BatchNormalization())
    MODEL.add(LeakyReLU(alpha=0.2))
    MODEL.add(Reshape([self.img_col // scale, self.img_row // scale, 1024]))
    MODEL = self.DeConv(MODEL, 64)
    MODEL = self.DeConv(MODEL, 32)
    MODEL.add(Conv2DTranspose(self.channels, [5, 5], strides=( 2, 2), activation='tanh', padding='same', kernel_initializer=RandomNormal(stddev=0.02)))
    print('Generator Model: \n', MODEL.summary())
    return MODEL

	# __________________________________________DISCRIMINATOR_______________________________________
   
  def Discriminator(self):
    MODEL = Sequential()
    MODEL.add(Conv2D(64, (5, 5),   input_shape=(self.img_col, self.img_row, self.channels), kernel_initializer=RandomNormal(stddev=0.02)))
    MODEL.add(LeakyReLU(alpha=0.2))
    MODEL.add(AveragePooling2D(pool_size=(2, 2)))
    MODEL = self.Convolution(MODEL, 128)
    MODEL.add(AveragePooling2D(pool_size=(2, 2)))
    MODEL = self.Convolution(MODEL, 128)
    MODEL.add(Flatten())
    MODEL.add(Dense(1))
    MODEL.add(Activation('sigmoid'))
    print('Discriminator Model: \n', MODEL.summary())
    return MODEL  
      
# ____________________________________SAVING_SAMPLES_THROUGH_TIME____________________________________

def Sampling(Sample, batch_size):
  col = Sample.shape[2]
  row = Sample.shape[1]
  mode = 'RGB'  
  NUMEL = int(math.sqrt(batch_size))
  IMG = Image.new(mode, (col * NUMEL, row * NUMEL))
  for i in range(NUMEL**2):
    randimg = Sample[i] * 127.5 + 127.5
    img = Image.fromarray(randimg.astype('uint8'), mode='RGB')
    IMG.paste(im=img, box=((i % NUMEL) * col, (i // NUMEL) * row))
        
  plt.imshow(IMG)
  plt.axis('off')
  plt.show();

 # __________________________________________PLOT_RESULTS______________________________________________  

def RESULTS(epoch):
  plt.figure()
  plt.plot(loss_Dis, label='Loss of Discriminator')
  plt.plot(loss_Gen, label='Loss of Generator')
  plt.xlabel('Epoch'), plt.ylabel('Loss')
  plt.legend(), plt.title('LOSS of Network')
  plt.show()
    
  plt.figure()
  plt.plot(acc_Dis, label='Accuracy of Discriminator')
  plt.plot(acc_Gen, label='Accuracy of Generator')
  plt.xlabel('Epoch'), plt.ylabel('Accuracy')
  plt.title('Accuracy of Network'), plt.ylim((0,1))
  plt.legend()
  plt.show()

# _____________________________________________TRAINING_____________________________________________        

loss_Dis, loss_Gen = [], []
acc_Dis, acc_Gen = [], []
def Train():
  
  (TRAINx, _), (_, _) = cifar10.load_data()
  TRAINx = TRAINx.astype(np.float32)
  TRAINx = (TRAINx - 127.5) / 127.5
  GAN = DCGAN(img_dims=TRAINx.shape[1:])
	
  generator = GAN.Generator()
  OPT_Gen = Adam(lr=0.0002, beta_1=0.5)
    
  discriminator = GAN.Discriminator()
  OPT_Dis = Adam(lr=0.0002, beta_1=0.5)
  discriminator.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer=OPT_Dis)
  gan = Sequential()
  gan.add(generator)
  discriminator.trainable = False
  gan.add(discriminator)
  gan.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=OPT_Gen)
  discriminator.trainable = True
	
  for epoch in range(EPOCHS):
    print("Epoch#{}".format(epoch))
    for i in range((2 * (TRAINx.shape[0] // batch_size))):
      #print(i)
      Noise = np.random.uniform(-1, 1, (batch_size, RANDOM_SIZE))
      Img_Gen = generator.predict_on_batch(Noise)
      if(i % 2 == 0):
        images = TRAINx[(i // 2 * batch_size) :(((i // 2) + 1) * batch_size), :, :, :]
              
        REAL = [1] * (batch_size)
                
        FAKE=[0] * (batch_size)
               

      else:
        images = Img_Gen
        REAL = [0] * (batch_size)
               
        FAKE=[1] * (batch_size)


      Dis = discriminator.train_on_batch(images, REAL)
      Noise = np.random.uniform(-1, 1, (batch_size, 100))
      labels = [1] * (batch_size)
      discriminator.trainable = False
      Gen = gan.train_on_batch(Noise,labels)
      discriminator.trainable = True          
    loss_Dis.append(Dis[0]), loss_Gen.append(Gen[0])
    acc_Dis.append(Dis[1]), acc_Gen.append(Gen[1])
        
    if (epoch%5==0):      
      Sampling(Img_Gen, batch_size)        
  RESULTS(epoch)

  
if __name__ == '__main__':
    Train()
    plt.show()     