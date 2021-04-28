'''
  Variational Autoencoder (VAE) with the Keras Functional API.
'''

import os
from PIL import Image
import tensorflow as tf
import keras
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from keras.layers import BatchNormalization
from keras.models import Model
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution
import matplotlib.pyplot as plt

disable_eager_execution()


class VAE_Model():
    def __init__(self, input_train, save_path, latent_dim, num_channels):
        raw_input = input_train
        self.img_width = raw_input.shape[1]
        self.img_height = raw_input.shape[2]
        self.num_channels = num_channels
        self.save_path = save_path
        self.latent_dim = latent_dim

        raw_input = raw_input.reshape(raw_input.shape[0], self.img_width, self.img_height, self.num_channels)
        self.input_shape = (self.img_width, self.img_height, self.num_channels)
        # Parse numbers as floats
        raw_input = raw_input.astype('float32')
        # Normalize data
        self.input_train = raw_input / 255

        self.encoder, self.input, self.mu, self.sigma, self.conv_shape = self.build_encoder()
        self.decoder = self.build_decoder()
        self.vae = None

    @staticmethod
    def sample_z(args):
        mu, sigma = args
        batch = K.shape(mu)[0]
        dim = K.int_shape(mu)[1]
        eps = K.random_normal(shape=(batch, dim))
        return mu + K.exp(sigma / 2) * eps

    @staticmethod
    def dataset_split(images, portion=0.05):
        validation_split = int(images.shape[0] * portion)
        train_images = images[validation_split:]
        test_images = images[:validation_split]
        return train_images, test_images

    def build_encoder(self):
        # Definition
        i = Input(shape=self.input_shape, name='encoder_input')
        cx = Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(i)
        cx = BatchNormalization()(cx)
        cx = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(cx)
        cx = BatchNormalization()(cx)
        x = Flatten()(cx)
        x = Dense(20, activation='relu')(x)
        x = BatchNormalization()(x)
        mu = Dense(self.latent_dim, name='latent_mu')(x)
        sigma = Dense(self.latent_dim, name='latent_sigma')(x)

        # Get Conv2D shape for Conv2DTranspose operation in decoder
        conv_shape = K.int_shape(cx)

        # Use reparameterization trick to ....??
        z = Lambda(self.sample_z, output_shape=(self.latent_dim,), name='z')([mu, sigma])

        # Instantiate encoder
        encoder = Model(i, [mu, sigma, z], name='encoder')
        encoder.summary()

        return encoder, i, mu, sigma, conv_shape

    def build_decoder(self):
        # Definition
        d_i = Input(shape=(self.latent_dim,), name='decoder_input')
        x = Dense(self.conv_shape[1] * self.conv_shape[2] * self.conv_shape[3], activation='relu')(d_i)
        x = BatchNormalization()(x)
        x = Reshape((self.conv_shape[1], self.conv_shape[2], self.conv_shape[3]))(x)
        cx = Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        cx = BatchNormalization()(cx)
        cx = Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(cx)
        cx = BatchNormalization()(cx)
        o = Conv2DTranspose(filters=self.num_channels, kernel_size=3, activation='sigmoid', padding='same',
                            name='decoder_output')(cx)

        # Instantiate decoder
        decoder = Model(d_i, o, name='decoder')
        decoder.summary()
        return decoder

    def build_vae(self):
        vae_outputs = self.decoder(self.encoder(self.input)[2])
        vae = Model(self.input, vae_outputs, name='vae')
        return vae

    # Define loss
    def kl_reconstruction_loss(self, true, pred):
        # Reconstruction loss
        reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * self.img_width * self.img_height
        # KL divergence loss
        kl_loss = 1 + self.sigma - K.square(self.mu) - K.exp(self.sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        # Total loss = 50% rec + 50% KL divergence loss
        return K.mean(reconstruction_loss + kl_loss)

    def train(self, epochs, batch_size, validation_split):
        # Compile VAE
        if self.vae is None:
            self.vae = self.build_vae()
        self.vae.compile(optimizer='adam', loss=self.kl_reconstruction_loss)
        self.vae.summary()
        # Train autoencoder
        self.vae.fit(self.input_train, self.input_train, epochs=epochs, batch_size=batch_size,
                     validation_split=validation_split)

        self.vae.save(self.save_path, save_format='tf')

    def load_model(self, load_path):
        get_custom_objects().update({"kl_reconstruction_loss": self.kl_reconstruction_loss})
        new_vae = keras.models.load_model(load_path)
        new_vae.summary()
        # Train autoencoder
        new_vae.fit(self.input_train, self.input_train, epochs=100, batch_size=128, validation_split=0.2)

    def print_result(self, file_path):
        input_test, x_test = self.dataset_split(self.input_train)
        encoded_imgs = self.encoder.predict(x_test)
        decoded_imgs = self.decoder.predict(encoded_imgs)
        n = 10  # How many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # Display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i].reshape(256, 256, 3))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i].reshape(256, 256, 3))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
        plt.savefig(file_path)


def load_images(dir_path):
    images = []
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
        image = np.asarray(Image.open(filepath))
        images.append(image)
    return np.stack(images, axis=0)


# Load MNIST dataset
images_dataset = load_images("./images/")

# Data & model configuration
model_path = "./models/model5"
image_path = "./result5.png"
model_batch_size = 128
no_epochs = 100
val_split = 0.2
verbosity = 1
latent_dimension = 2
number_channels = 3

vae_model = VAE_Model(images_dataset, model_path, latent_dimension, number_channels)
vae_model.train(epochs=no_epochs, batch_size=model_batch_size, validation_split=val_split)
vae_model.load_model(model_path)
vae_model.print_result(image_path)
