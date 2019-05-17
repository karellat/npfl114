#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from mnist import MNIST

# The neural network model
class Network:
    def __init__(self, args):
        self._z_dim = args.z_dim

        # TODO: Define `self.generator` as a Model, which
        # - takes vectors of [args.z_dim] shape on input
        input1 = tf.keras.layers.Input([args.z_dim])
        # - applies batch normalized dense layer with 1024 units and ReLU (do not forget about `use_bias=False` in suitable places)
        dense1 = tf.keras.layers.Dense(1024,use_bias=False)(input1)
        dense1 = tf.keras.layers.BatchNormalization()(dense1)
        dense1 = tf.keras.layers.Activation("relu")(dense1)

        # - applies batch normalized dense layer with mnist.h // 4 * mnist.w // 4 * 64 units and ReLU
        dense2 = tf.keras.layers.Dense(MNIST.H // 4 * MNIST.W // 4 *64,
                use_bias=False)(dense1)
        dense2 = tf.keras.layers.BatchNormalization()(dense2)
        dense2 = tf.keras.layers.Activation("relu")(dense2)
        # - reshapes the current hidder output to [MNIST.H // 4, MNIST.W // 4, 64]
        reshaper = tf.keras.layers.Reshape([MNIST.H // 4, MNIST.W // 4,
            64])(dense2)
        # - applies batch normalized transposed convolution with 32 filters, kernel size 5,
        #   stride 2, same padding, and ReLU activation
        conv = tf.keras.layers.Conv2DTranspose(32, 5, strides=2, padding='same',
                use_bias=False)(reshaper)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Activation("relu")(conv)
        # - applies transposed convolution with 1 filters, kernel size 5,
        #   stride 2, same padding, and sigmoid activation
        conv2 = tf.keras.layers.Conv2DTranspose(1, 5, strides=2, padding='same',
                activation='sigmoid')(conv)

        self.generator = tf.keras.Model(inputs=[input1], outputs=[conv2])

        # TODO: Define `self.discriminator` as a Model, which
        # - takes input images with shape [MNIST.H, MNIST.W, MNIST.C]
        input2 = tf.keras.layers.Input([MNIST.H, MNIST.W, MNIST.C])
        # - computes batch normalized convolution with 32 filters, kernel size 5,
        #   same padding, and ReLU activation (do not forget `use_bias=False` where appropriate)
        conv = tf.keras.layers.Conv2D(32, 5, padding='same',use_bias=False)(input2)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Activation("relu")(conv)
        # - max-pools with kernel size 2 and stride 2
        pooling = tf.keras.layers.MaxPool2D(2,2)(conv)
        # - computes batch normalized convolution with 64 filters, kernel size 5,
        #   same padding, and ReLU activation
        conv = tf.keras.layers.Conv2D(64, 5, padding='same',
                use_bias=False)(pooling)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Activation("relu")(conv)
        # - max-pools with kernel size 2 and stride 2
        pooling = tf.keras.layers.MaxPool2D(2,2)(conv)
        # - flattens the current representation
        flatten = tf.keras.layers.Flatten()(pooling)
        # - applies batch normalized dense layer with 1024 uints and ReLU activation
        dense1 = tf.keras.layers.Dense(1024,use_bias=False)(flatten)
        dense1 = tf.keras.layers.BatchNormalization()(dense1)
        dense1 = tf.keras.layers.Activation("relu")(dense1)
        # - applies output dense layer with one output and a suitable activation function
        output2 = tf.keras.layers.Dense(1, activation='sigmoid')(dense1)

        self.discriminator = tf.keras.Model(inputs=[input2],
                outputs=[output2])

        self._generator_optimizer, self._discriminator_optimizer = tf.optimizers.Adam(), tf.optimizers.Adam()
        self._loss_fn = tf.losses.BinaryCrossentropy()
        self._discriminator_accuracy = tf.metrics.Mean()
        self._writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def _sample_z(self, batch_size):
        """Sample random latent variable."""
        return tf.random.uniform([batch_size, self._z_dim], -1, 1, seed=42)

    @tf.function
    def train_batch(self, images):
        # TODO: Generator training. Using a Gradient tape:
        with tf.GradientTape() as tape:
        # - generate random images using a `generator`; do not forget about `training=True` where appropriate
            gen_images = self.generator(self._sample_z(images.shape[0]), training=True)
        # - run discriminator on the generated images, also using `training=True` (even if
        #   not updating discriminator parameters, we want to perform possible BatchNorm in it)
            disc_gen_images = self.discriminator(gen_images, training=True)
        # - compute loss using `_loss_fn`, with target labels `tf.ones_like(discriminator_output)`
            generator_loss = self._loss_fn(tf.ones_like(disc_gen_images),
               disc_gen_images)
        # Then, compute the gradients with respect to generator trainable variables and update
        gradients = tape.gradient(generator_loss, self.generator.variables)
        # generator trainable weights using self._generator_optimizer.
        self._generator_optimizer.apply_gradients(zip(gradients, self.generator.variables))

        # TODO: Discriminator training. Using a Gradient tape:
        with tf.GradientTape() as tape:
        # - discriminate `images`, storing results in `discriminated_real`
        # TODO: TRUE???
            discriminated_real = self.discriminator(images, training=True)
        # - discriminate images generated in generator training, storing results in `discriminated_fake`
            discriminated_fake = self.discriminator(gen_images, training=True)
        # - compute loss by summing
        #   - `_loss_fn` on discriminated_real with suitable target labels
            loss_real = self._loss_fn(tf.ones_like(discriminated_real),
                    discriminated_real)
        #   - `_loss_fn` on discriminated_fake with suitable targets (`tf.{ones,zeros}_like` come handy).
            loss_fake = self._loss_fn(tf.zeros_like(discriminated_fake),
                    discriminated_fake)
            discriminator_loss = loss_real + loss_fake
        # Then, compute the gradients with respect to discriminator trainable variables and update
        gradients = tape.gradient(discriminator_loss, self.discriminator.variables)
        # discriminator trainable weights using self._discriminator_optimizer.
        self._discriminator_optimizer.apply_gradients(zip(gradients, self.discriminator.variables))

        self._discriminator_accuracy(tf.greater(discriminated_real, 0.5))
        self._discriminator_accuracy(tf.less(discriminated_fake, 0.5))
        tf.summary.experimental.set_step(self._discriminator_optimizer.iterations)
        with self._writer.as_default():
            tf.summary.scalar("gan/generator_loss", generator_loss)
            tf.summary.scalar("gan/discriminator_loss", discriminator_loss)
            tf.summary.scalar("gan/discriminator_accuracy", self._discriminator_accuracy.result())

        return generator_loss + discriminator_loss

    def generate(self):
        GRID = 20

        # Generate GRIDxGRID images
        random_images = self.generator(self._sample_z(GRID * GRID))

        starts, ends = self._sample_z(GRID), self._sample_z(GRID)
        interpolated_z = tf.concat(
            [starts[i] + (ends[i] - starts[i]) * tf.expand_dims(tf.linspace(0., 1., GRID), -1) for i in range(GRID)], axis=0)
        interpolated_images = self.generator(interpolated_z)

        # Stack the random images, then an empty row, and finally interpolated imates
        image = tf.concat(
            [tf.concat(list(images), axis=1) for images in tf.split(random_images, GRID)] +
            [tf.zeros([MNIST.H, MNIST.W * GRID, MNIST.C])] +
            [tf.concat(list(images), axis=1) for images in tf.split(interpolated_images, GRID)], axis=0)
        with self._writer.as_default():
            tf.summary.image("gan/images", tf.expand_dims(image, 0))

    def train_epoch(self, dataset, args):
        self._discriminator_accuracy.reset_states()
        loss = 0
        for batch in dataset.batches(args.batch_size):
            loss += self.train_batch(batch["images"])
        self.generate()
        return loss


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--dataset", default="mnist", type=str, help="MNIST-like dataset to use.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--z_dim", default=100, type=int, help="Dimension of Z.")
    args = parser.parse_args()

    # Fix random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = lambda: tf.initializers.glorot_uniform(seed=42)
        tf.keras.utils.get_custom_objects()["orthogonal"] = lambda: tf.initializers.orthogonal(seed=42)
        tf.keras.utils.get_custom_objects()["uniform"] = lambda: tf.initializers.RandomUniform(seed=42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    mnist = MNIST(args.dataset)

    # Create the network and train
    network = Network(args)
    for epoch in range(args.epochs):
        loss = network.train_epoch(mnist.train, args)

    with open("gan.out", "w") as out_file:
        print("{:.2f}".format(loss), file=out_file)
