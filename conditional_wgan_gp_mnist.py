from __future__ import division, print_function

import sys
from functools import partial

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          Flatten, Input, Reshape, ZeroPadding2D, concatenate)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.merge import _Merge
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model, to_categorical


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""

    def _merge_function(self, inputs):# inputs are a list: [real data, fake data]
        alpha = K.random_uniform(shape=K.shape(inputs[0]))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class WGANGP():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows*self.img_cols) # This code did not use CNN
        self.latent_dim = 100
        self.label_dim = 10
        self.layers = [256, 128]
        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = Adam(lr=0.0001, beta_1=0.5, beta_2=0.9)
        optimizer = RMSprop(lr=0.0001)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        # -------------------------------
        # Construct Computational Graph
        #       for the Critic
        # -------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=(self.img_shape,))
        label = Input(shape=(self.label_dim,)) #Label input is a one-hot vector
        # Noise input
        z_disc = Input(shape=(100,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator([z_disc, label])

        # Discriminator determines validity of the real and fake images
        fake = self.critic([fake_img, label])
        valid = self.critic([real_img, label])

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic([interpolated_img, label])

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc, label],
                                  outputs=[valid, fake, validity_interpolated])

        self.critic_model.compile(loss=[self.wasserstein_loss,
                                        self.wasserstein_loss,
                                        partial_gp_loss],
                                  optimizer=optimizer,
                                  loss_weights=[1, 1, 10])#use loss_weights instead define lambda directly
                                  #the paper use lambda=10 for gp loss
        # -------------------------------
        # Construct Computational Graph
        #         for Generator
        # -------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(100,))
        # Generate images based of noise
        img = self.generator([z_gen, label])
        # Discriminator determines validity
        valid = self.critic([img, label])
        # Defines generator model
        self.generator_model = Model([z_gen, label], valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.label_dim,))
        x = concatenate([noise, label])
        for l in self.layers[::-1][1:]:# extended slices: [::n] means extracting objects for every n objects.
        #minus sign here means starting from the reverse order
            x = Dense(l)(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = Activation('relu')(x)

        out = Dense(self.img_shape, activation='tanh')(x)
        return Model([noise, label], out)

    def build_critic(self):
        nodes = int(np.median(self.layers))
        
        input_critic = Input(shape=(self.img_shape+self.label_dim,))
        x=Dense(nodes)(input_critic)
        x=LeakyReLU(0.2)(x)
        if len(self.layers) > 1:
            for l in self.layers[1:]:
                x=Dense(nodes)(x)
                x=LeakyReLU(0.2)(x)
        x=Dense(1, activation='linear')(x)
        model=Model(input=input_critic, output=x)        
        model.summary()

        data = Input(shape=(self.img_shape, ))
        label = Input(shape=(self.label_dim, ))
        model_input = concatenate([data, label])
        validity = model(model_input)
        discriminator = Model([data, label], validity)
        return discriminator

    def train(self, epochs, batch_size, sample_interval=100):

        # Load the dataset
        (X_train, Y_train), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))

        Y_train = to_categorical(Y_train)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1)) # use -1 for true sample since keras can only minimize a loss function
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))  # Dummy gt for gradient penalty
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                labels = Y_train[idx]

                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise, labels],
                                                          [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            labels = Y_train[idx]
            g_loss = self.generator_model.train_on_batch([noise, labels], valid)# For fake samples set the label as -1 for the generator

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0 or epoch == epochs:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        labels = to_categorical(np.arange(0, 10).reshape(-1, 1))

        # Use the below labels if displaying the sampels for a single label 
        # noise = np.random.normal(0,1,(1,100))
        # labels = np.array([0., 0., 1.0, 0., 0., 0., 0., 0., 0., 0.]).reshape(-1,10)
        gen_imgs = self.generator.predict([noise, labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 1
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt].reshape(28, 28))
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    wgan = WGANGP()
    wgan.train(epochs=20000, batch_size=32, sample_interval=1000)