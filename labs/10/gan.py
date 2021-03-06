#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from mnist import MNIST

# The neural network model
class Network:
    def __init__(self, args):
        self._z_dim = args.z_dim
        self._batch_size = args.batch_size

        # TODO: Define `self.generator` as a Model, which
        # - takes vectors of [args.z_dim] shape on input
        input1 = tf.keras.layers.Input([args.z_dim])
        # - applies len(args.generator_layers) dense layers with ReLU activation,
        #   i-th layer with args.generator_layers[i] units
        dense1 = input1
        for l in args.generator_layers:
            dense1 = tf.keras.layers.Dense(l, activation='relu')(dense1)

        # - applies output dense layer with MNIST.H * MNIST.W * MNIST.C units
        #   and sigmoid activation
        output1 = tf.keras.layers.Dense(MNIST.H * MNIST.W * MNIST.C, activation='sigmoid')(dense1)
        # - reshapes the output (tf.keras.layers.Reshape) to [MNIST.H, MNIST.W, MNISt.C]
        output1 = tf.keras.layers.Reshape([MNIST.H, MNIST.W, MNIST.C])(output1)


        self.generator = tf.keras.Model(inputs=[input1], outputs=[output1])
        # TODO: Define `self.discriminator` as a Model, which
        # - takes input images with shape [MNIST.H, MNIST.W, MNIST.C]
        input2 = tf.keras.layers.Input([MNIST.H, MNIST.W, MNIST.C])
        # - flattens them
        flatten2 = tf.keras.layers.Flatten()(input2)
        # - applies len(args.discriminator_layers) dense layers with ReLU activation,
        #   i-th layer with args.discriminator_layers[i] units
        dense2 = flatten2
        for l in args.generator_layers:
            dense2 = tf.keras.layers.Dense(l, activation='relu')(dense2)
        # - applies output dense layer with one output and a suitable activation function

        output2 = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)
        # TODO: Define `self.discriminator` as a Model, which
        self.discriminator = tf.keras.Model(inputs=[input2], outputs=[output2])


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
        b = 0
        for batch in dataset.batches(args.batch_size):
            b += 1
            print('Batch {}'.format(b))
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
    parser.add_argument("--discriminator_layers", default="128", type=str, help="Discriminator layers.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
    parser.add_argument("--generator_layers", default="128", type=str, help="Generator layers.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--z_dim", default=100, type=int, help="Dimension of Z.")
    args = parser.parse_args()
    args.discriminator_layers = [int(discriminator_layer) for discriminator_layer in args.discriminator_layers.split(",")]
    args.generator_layers = [int(generator_layer) for generator_layer in args.generator_layers.split(",")]

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
        print('Epoch {}'.format(epoch+1))
        loss = network.train_epoch(mnist.train, args)

    with open("gan.out", "w") as out_file:
        print("{:.2f}".format(loss), file=out_file)

