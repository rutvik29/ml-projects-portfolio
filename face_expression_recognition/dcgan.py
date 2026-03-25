"""DCGAN for facial expression generation.

Trained for 100 epochs on 15,000 images (64x64).
Evaluated with Inception Score (IS) and FID.
"""
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = 64
LATENT_DIM = 128
BATCH_SIZE = 64


def make_generator():
    model = tf.keras.Sequential([
        layers.Input(shape=(LATENT_DIM,)),
        layers.Dense(4 * 4 * 512, use_bias=False),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Reshape((4, 4, 512)),
        layers.Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh'),
    ], name="generator")
    return model


def make_discriminator():
    model = tf.keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(64, 4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, 4, strides=2, padding='same'),
        layers.BatchNormalization(), layers.LeakyReLU(0.2),
        layers.Conv2D(256, 4, strides=2, padding='same'),
        layers.BatchNormalization(), layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid'),
    ], name="discriminator")
    return model


class DCGAN:
    def __init__(self):
        self.generator = make_generator()
        self.discriminator = make_discriminator()
        self.g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()

    @tf.function
    def train_step(self, real_images):
        noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            fake = self.generator(noise, training=True)
            real_pred = self.discriminator(real_images, training=True)
            fake_pred = self.discriminator(fake, training=True)
            d_loss = self.loss_fn(tf.ones_like(real_pred), real_pred) + self.loss_fn(tf.zeros_like(fake_pred), fake_pred)
            g_loss = self.loss_fn(tf.ones_like(fake_pred), fake_pred)
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        return {"d_loss": d_loss, "g_loss": g_loss}

    def train(self, dataset, epochs=100):
        for epoch in range(epochs):
            for batch in dataset:
                metrics = self.train_step(batch)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: D_loss={metrics['d_loss']:.4f}, G_loss={metrics['g_loss']:.4f}")
                self.save_samples(epoch)

    def save_samples(self, epoch, num_samples=16):
        noise = tf.random.normal([num_samples, LATENT_DIM])
        samples = self.generator(noise, training=False)
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            img = (samples[i].numpy() + 1) / 2
            ax.imshow(img); ax.axis('off')
        plt.savefig(f"./outputs/epoch_{epoch:04d}.png", bbox_inches='tight')
        plt.close()
