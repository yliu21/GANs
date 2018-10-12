
# coding: utf-8

# In[1]:


# GAN on fashion-minist
# Code adapted from https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/GAN/dcgan_mnist.py
import numpy as np
import os
import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')
from keras.models import *
from keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Activation, Reshape
from keras.layers import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import time
from keras.datasets import fashion_mnist


# In[2]:


# For time monitoring
class ElapsedTimer:
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )


# In[3]:


class DCGAN:
    def __init__(self, img_rows=28, img_cols=28, channel=1):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # (W−F+2P)/S+1
    def discriminator(self, train_status=True):
        if self.D:
            return self.D # If there is already a discriminator, it will not be overwritten.
        self.D = Sequential()
        depth = 64
        dropout = 0.4
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same', trainable=train_status))
        self.D.add(LeakyReLU(0.3))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same', trainable=train_status ))
        self.D.add(LeakyReLU(0.3))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same', trainable=train_status))
        self.D.add(LeakyReLU(0.3))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same', trainable=train_status))
        self.D.add(LeakyReLU(0.3))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D
    
    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 256
        dim = 7
        # In: 100
        # Out: dim x dim x depth
        # In: 100
        self.G.add(Dense(dim*dim*depth, input_dim=100))
        self.G.add(BatchNormalization(momentum=0.9))
        # self.G.add(Activation('relu'))
        self.G.add(LeakyReLU(0.3))
        self.G.add(Reshape((dim, dim, depth)))# Out: 7 x 7 x 256
        self.G.add(Dropout(dropout))
        # In: dim x dim x depth
        self.G.add(UpSampling2D()) # Out: 14 x 14 x 256
        self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same')) # Out: 14 x 14 x 128 
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(LeakyReLU(0.3))
        self.G.add(UpSampling2D()) # Out: 28 x 28 x 128
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same')) # Out: 28 x 28 x 64 
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(LeakyReLU(0.3))
        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same')) # Out: 28 x 28 x 32
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(LeakyReLU(0.3))
        self.G.add(Conv2DTranspose(1, 5, padding='same'))# Out: 28 x 28 x 1
        self.G.add(Activation('tanh')) # Out: 28 x 28 x 1 grayscale image [-1.0,1.0] per pix
        self.G.summary()
        return self.G
    
    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0003, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator(train_status=False))
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.AM


# In[6]:


class MNIST_DCGAN:
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1
        
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train/255
        self.x_train = x_train
        self.x_train = self.x_train.reshape(-1, self.img_rows, self.img_cols, 1).astype(np.float32)
        self.DCGAN = DCGAN() #DCGAN class
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()
        
    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None
        d_loss_history = []
        a_loss_history = []
        d_acc_history = []
        a_acc_history = []
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=(16, 100)) # to test the GAN for plotting purpose
        for i in range(train_steps):
            images_train = self.x_train[np.random.randint(0,
                self.x_train.shape[0], size=batch_size), :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0 # fake images get label 0
            d_loss = self.discriminator.train_on_batch(x, y)
            d_loss_history.append(d_loss[0])
            d_acc_history.append(d_loss[1])

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)
            a_loss_history.append(a_loss[0])
            a_acc_history.append(a_loss[1])
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],noise=noise_input, step=(i+1))
                    self.plot_history(step=(i+1), d_loss_history=d_loss_history, a_loss_history=a_loss_history, d_acc_history=d_acc_history, a_acc_history=a_acc_history)
                    
                    
    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
            filename = 'mnist.png'
            if fake:# if want to save the fake images generated by the generator
                if noise is None:
                    noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
                else:
                    filename = "mnist_%d.png" % step
                images = self.generator.predict(noise)
            else:# if do not want to save the fake images, randomly choose images (n=smaples) from the training set to plot
                i = np.random.randint(0, self.x_train.shape[0], samples)
                images = self.x_train[i, :, :, :] 

            plt.figure(figsize=(10,10))
            for i in range(images.shape[0]):
                plt.subplot(4, 4, i+1)
                image = images[i, :, :, :]
                image = np.reshape(image, (self.img_rows, self.img_cols))
                plt.imshow(image, cmap='gray')
                plt.axis('off')
            plt.tight_layout() # tight_layout automatically adjusts subplot params so that the subplot(s) fits in to the figure area. 
            if save2file:
                plt.savefig(filename)
                plt.close('all')
            else:
                plt.show()
    
    
    def plot_history(self, step=0, d_loss_history=None, a_loss_history=None, d_acc_history=None, a_acc_history=None, loss_only=False):
            if d_loss_history and a_loss_history and d_acc_history and a_acc_history:
                #Plot loss history
                plt.plot(d_loss_history)
                plt.plot(a_loss_history)
                plt.title('Model Loss')
                plt.ylabel('Loss')
                plt.xlabel('Step')
                plt.legend(['Discriminator Model', 'Adversial Model'], loc='upper left')
                filename = "loss_%d.png" % step
                plt.savefig(filename)
                if loss_only == True:
                    plt.plot(d_acc_history)
                    plt.plot(a_acc_history)
                    plt.title('Model Accuracy')
                    plt.ylabel('Accuracy')
                    plt.xlabel('Step')
                    plt.legend(['Discriminator Model', 'Adversial Model'], loc='upper left')
                    filename = "acc_%d.png" % step
                    plt.savefig(filename)
            else:
                print('Do not have complete history!')
                
                


# In[ ]:


if __name__ == '__main__':
    mnist_dcgan = MNIST_DCGAN()
    timer = ElapsedTimer()
    mnist_dcgan.train(train_steps=5000, batch_size=256, save_interval=500)
    timer.elapsed_time()
#     mnist_dcgan.plot_images(fake=True,save2file=True) # plot fake images
#     mnist_dcgan.plot_images(fake=False, save2file=True) # plot orginal images 



