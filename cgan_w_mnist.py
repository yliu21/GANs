# Conditional Wasserstein Gan
# Code adpated based on https://myurasov.github.io/2017/09/24/wasserstein-gan-keras.html

import os
import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')
from keras.models import *
from keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Activation, Reshape, Embedding, multiply
from keras.layers import Conv2D, UpSampling2D, Conv2DTranspose, MaxPool2D
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import time
from keras.datasets import mnist
import keras.backend as K
from keras.initializers import RandomNormal
plt.switch_backend('Agg')

RND = 777
np.random.seed(RND)

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

class WGAN:
    def __init__(self, img_rows=28, img_cols=28, channel=1):
       
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model
    
    def d_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
    # The real loss function should be z * pred_prob + (1 - z) * -log(1 - pred_prob)
    # where z=1 if real images and z=0 if fake images
    # But since we want to construct different mini-batches for real and fake, 
    # i.e. each mini-batch needs to contain only all real images or all generated images (https://github.com/soumith/ganhacks)
    # we could use the above loss function to calculate loss with z=-1 for real images and 1 for fake images
    # More details: https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/WassersteinGAN/src/model
    # https://github.com/Zardinality/WGAN-tensorflow/issues/6

    # Discriminator takes image as input and has two ouputs:
    # measure of it’s “fakeness” (maximized for generated images) with linear activation
    # predicted image class with softmax activation
    def discriminator(self):
        if self.D:
            return self.D # If there is already a discriminator, it will not be overwritten.        
        depth = 32
        dropout = 0.3
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        # weights are initialized from normal distribution with stddev of 0.02 so initial clipping doesn’t cut off all the weights
        weight_init = RandomNormal(mean=0., stddev=0.02)
        input_shape = (self.img_rows, self.img_cols, self.channel)
        input_image = Input(shape=input_shape)
        x = Conv2D(depth*1, 3, strides=1, input_shape=input_shape, padding='same', kernel_initializer=weight_init)(input_image)
        x = LeakyReLU()(x)
        x = MaxPool2D()(x)
        # x = BatchNormalization(momentum=0.9)(x)
        x = Dropout(dropout)(x)

        x = Conv2D(depth*2, 3, strides=1, padding='same', kernel_initializer=weight_init)(x)
        x = MaxPool2D(pool_size=1)(x)
        x = LeakyReLU()(x)
        # x = BatchNormalization(momentum=0.9)(x)
        x = Dropout(dropout)(x)

        x = Conv2D(depth*4, 3, strides=1, padding='same', kernel_initializer=weight_init)(x)
        x = MaxPool2D()(x)
        x = LeakyReLU()(x)
        # x = BatchNormalization(momentum=0.9)(x)
        x = Dropout(dropout)(x)

        x = Conv2D(depth*8, 3, strides=1, padding='same', kernel_initializer=weight_init)(x)
        x = MaxPool2D(pool_size=1)(x)
        x = LeakyReLU()(x)
        # x = BatchNormalization(momentum=0.9)(x)
        x = Dropout(dropout)(x)

        features = Flatten()(x)
        output_real_fake = Dense(1, activation='linear', name='output_real_fake')(features)# No sigmoid activation for last layer!
        output_class = Dense(10, activation='softmax', name='output_class')(features)
        
        self.D = Model(inputs=[input_image], outputs=[output_real_fake, output_class], name='D') 
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G
        
        z_size=100
        num_classes=10
        dropout = 0.4
        depth = 128
        dim = 7
        weight_init = RandomNormal(mean=0., stddev=0.02)

        input_class=Input(shape=(1,), dtype='int32', name='input_class')
        # encode class to the same size as Z to use hadamard multiplication later on
        e = Embedding(num_classes, z_size, embeddings_initializer='glorot_uniform')(input_class)
        embedded_class = Flatten(name='embedded_class')(e)# a vector that has the same size as the noise vector
        input_z = Input(shape=(z_size,),name='input_z')
        # calculate the hadamard product - elementwise multiplication
        h = multiply([embedded_class, input_z], name='h')

        # cnn
        x = Dense(1024)(h)
        x = LeakyReLU()(x)
        x = Dense(dim*dim*depth)(x)
        # x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU()(x)
        x = Reshape((dim, dim, depth))(x) # Out: 7 x 7 x 128
        # x = Dropout(dropout)(x)
        # In: dim x dim x depth
        x = UpSampling2D()(x) # Out: 14 x 14 x 128
        x = Conv2D(256, 5, padding='same', kernel_initializer=weight_init)(x) # Out: 14 x 14 x 256
        # x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU()(x)
        x = UpSampling2D()(x) # Out: 28 x 28 x 128
        x = Conv2D(128, 5, padding='same', kernel_initializer=weight_init)(x) # Out: 28 x 28 x 32 
        # x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU()(x)
        x = Conv2D(1, 2, padding='same', activation='tanh', kernel_initializer=weight_init)(x) # Out: 28 x 28 x 1
        # x = BatchNormalization(momentum=0.9)(x)
        # Out: 28 x 28 x 1 grayscale image [-1.0,1.0] per pix
        self.G = Model(inputs=[input_z, input_class], outputs=x, name='G')
        self.G.summary()
        return self.G
    
    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.00005)
        self.DM = self.discriminator()
        self.DM.compile(loss=[self.d_loss, 'sparse_categorical_crossentropy'], optimizer=optimizer, metrics=None)
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        z_size=100
        optimizer = RMSprop(lr=0.00005)
        input_z = Input(shape=(z_size, ))
        input_class = Input(shape=(1, ),dtype='int32')
        D=self.discriminator()
        G=self.generator()
        output_real_fake, output_class = D(G(inputs=[input_z, input_class]))
        self.AM = Model(inputs=[input_z, input_class], outputs=[output_real_fake, output_class])
        self.AM.get_layer('D').trainable = False # freeze D in generator training faze
        self.AM.compile(loss=[self.d_loss, 'sparse_categorical_crossentropy'], optimizer=optimizer, metrics=None)
        return self.AM


class MNIST_WGAN:
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1

        
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # use all available 70k samples
        x_train = np.concatenate((x_train, x_test))
        y_train = np.concatenate((y_train, y_test))

        x_train = (x_train.astype(np.float32)-127.5)/127.5 # scale to the range of tanh activation function [-1,1]
        x_train = np.expand_dims(x_train, axis=3)
        self.x_train = x_train
        self.y_train = y_train 
        self.WGAN = WGAN() #DCGAN class
        self.discriminator =  self.WGAN.discriminator_model()
        self.adversarial = self.WGAN.adversarial_model()
        self.generator = self.WGAN.generator()
        
    def train(self, train_steps=2000, batch_size=100, save_interval=0, D_ITERS= 5):# Train discriminator D_ITERS times before updating the generator
        noise_input = None
        d_loss_history = []
        a_loss_history = []
        if save_interval>0:
            noise_input = np.random.normal(0., 1., size=(16, 100)) # to test the GAN for plotting purpose
        for i in range(train_steps):
            if (i % 1000) < 25 or i % 500 == 0: # 25 times in 1000, every 500th
                d_iters = 100
            else:
                d_iters = D_ITERS
            # Train the discriminator for D_ITERS rounds 
            for j in range(d_iters):

                self.discriminator.trainable = True
                for l in self.discriminator.layers: l.trainable = True
                # weight clipping
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -0.01, 0.01) for w in weights]
                    l.set_weights(weights)
                index = np.random.choice(len(self.x_train), batch_size, replace=False)
                images_train = self.x_train[index, :, :, :]
                true_class = self.y_train[index,]
                noise = np.random.normal(0., 1., size=[batch_size, 100])
                generated_classes = np.random.randint(0, 10, batch_size)
                images_fake = self.generator.predict([noise, generated_classes.reshape(-1,1)])
                y_true = -np.ones(batch_size)
                y_fake = np.ones(batch_size)
                # update discriminator on a batch of real images
                d_loss_images_train = self.discriminator.train_on_batch(images_train, [y_true, true_class])
                # print(self.discriminator.metrics_names)
                # update discriminator on a batch of fake images
                d_loss_images_fake = self.discriminator.train_on_batch(images_fake, [y_fake, generated_classes])
                # Calculate Wasserstein Distance
                d_loss = -d_loss_images_train[1]-d_loss_images_fake[1]
                d_loss_history.append(d_loss)
                
        

            # Train the generator
            self.discriminator.trainable = False
            for l in self.discriminator.layers: l.trainable = False
            y = -np.ones(batch_size)
            noise = np.random.normal(0., 1., size=[batch_size, 100])
            generated_classes = np.random.randint(0, 10, batch_size)
            a_loss = self.adversarial.train_on_batch([noise, generated_classes.reshape((-1, 1))] , [y,generated_classes])
            # a_loss_history.append(a_loss)
            log_mesg = "%d: [D loss: %f]" % (i, d_loss)
            # log_mesg = "%s  [A loss: %f]" % (log_mesg, a_loss)
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],noise=noise_input, step=(i+1))
                    self.plot_history(step=(i+1), d_loss_history=d_loss_history)
                    
                    
    def plot_images(self, save2file=False, samples=16, noise=None, step=0):
        if noise is None:
            noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
        generated_classes = np.random.randint(0, 10, samples)
        filename = "mnist_wgan_%d.png" % step
        images = self.generator.predict([noise, generated_classes])
        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, (self.img_rows, self.img_cols))
            plt.imshow(image, cmap='gray')
            plt.title('Class %d' % generated_classes[i])
            plt.axis('off')
        plt.tight_layout() # tight_layout automatically adjusts subplot params so that the subplot(s) fits in to the figure area. 
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()
    
    
    def plot_history(self, step=0, d_loss_history=None):
            if d_loss_history:
                #Plot loss history
                d_loss_less=[d_loss_history[i] for i in range(0,len(d_loss_history),10)]
                plt.plot(d_loss_less)
                # plt.plot(a_loss_history)
                plt.title('Model Loss')
                plt.ylabel('Estimated Wasserstein Distance')
                plt.xlabel('Steps/10')
                # plt.legend(['Discriminator Model', 'Adversial Model'], loc='upper left')
                filename = "loss_wgan_%d.png" % step
                plt.savefig(filename)
            else:
                print('Do not have complete history!')


if __name__ == '__main__':
    mnist_wgan = MNIST_WGAN()
    timer = ElapsedTimer()
    mnist_wgan.train(train_steps=25000, batch_size=100, save_interval=100)
    timer.elapsed_time()
#     mnist_dcgan.plot_images(fake=True,save2file=True) # plot fake images
#     mnist_dcgan.plot_images(fake=False, save2file=True) # plot orginal images 
