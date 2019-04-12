#!/usr/bin/env python2
import numpy as np
import tensorflow as tf

from mnist import MNIST

# The neural network model
class Network:
    def __init__(self, args):
        # Load data
        mnist = MNIST()
        # Create the model
        inputs1 = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])
        inputs2 = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])
        conv_10_1 = tf.keras.layers.Conv2D(
                filters=10,
                kernel_size=(3,3),
                strides=2,
                padding="valid")
        conv_20_1 = tf.keras.layers.Conv2D(
                filters=20,
                kernel_size=(3,3),
                strides=2,
                padding="valid")
        flatten_1 = tf.keras.layers.Flatten()
        dense_1 = tf.keras.layers.Dense(200, activation="relu")
        shared_dense = tf.keras.layers.Dense(10, activation="softmax")

        hidden_1 = shared_dense(dense_1(flatten_1(conv_20_1(conv_10_1(inputs1)))))
        hidden_2 = shared_dense(dense_1(flatten_1(conv_20_1(conv_10_1(inputs2)))))

        concat   = tf.keras.layers.Concatenate()([hidden_1, hidden_2])
        compare = tf.keras.layers.Dense(200, activation="relu")(concat)
        output_3 = tf.keras.layers.Dense(1, activation="sigmoid")(compare)
        self.model = tf.keras.Model(inputs=[inputs1,inputs2], outputs=[hidden_1, hidden_2, output_3])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=[tf.keras.losses.SparseCategoricalCrossentropy(),
                tf.keras.losses.SparseCategoricalCrossentropy(),
                tf.keras.losses.BinaryCrossentropy()],
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],)
        # It then passes each input image through the same network (with shared weights), performing
        # - convolution with 10 filters, 3x3 kernel size, strides 2, "valid" padding, ReLU activation
        # - convolution with 20 filters, 3x3 kernel size, strides 2, "valid" padding, ReLU activation
        # - flattening layer
        # - fully connected layer with 200 neurons and ReLU activation
        # obtaining a 200-dimensional feature representation of each image.
        #
        # Then, it produces three outputs:
        # - classify the computed representation of the first image using a densely connected layer
        #   into 10 classes;
        # - classify the computed representation of the second image using the
        #   same connected layer (with shared weights) into 10 classes;
        # - concatenate the two image representations, process them using another fully connected
        #   layer with 200 neurons and ReLU, and finally compute one output with tf.nn.sigmoid
        #   activation (the goal is to predict if the first digit is larger than the second)
        #
        # Train the outputs using SparseCategoricalCrossentropy for the first two inputs
        # and BinaryCrossentropy for the third one, utilizing Adam with default arguments.

    @staticmethod
    def _prepare_batches(batches_generator):
        batches = []
        for batch in batches_generator:
            batches.append(batch)
            if len(batches) >= 2:
                model_inputs = [batches[0:2][0]['images'],
                        batches[0:2][1]['images']]
                model_targets = [
                        batches[0:2][0]['labels'],
                        batches[0:2][1]['labels'],
                        batches[0:2][0]['labels'] > batches[0:2][1]['labels']
                ]

                # TODO: yield the suitable modified inputs and targets using batches[0:2]
                yield (model_inputs, model_targets)
                batches.clear()

    def train(self, mnist, args):
        for epoch in range(args.epochs):
            # TODO: Train for one epoch using `model.train_on_batch` for each batch.
            for inputs, targets in self._prepare_batches(mnist.train.batches(args.batch_size)):
                self.model.train_on_batch(inputs, targets, reset_metrics=False)

            # Print development evaluation
            print("Dev {}: directly predicting: {:.4f}, comparing digits: {:.4f}".format(epoch + 1, *self.evaluate(mnist.dev, args)))

    def evaluate(self, dataset, args):
        # TODO: Evaluate the given dataset, returning two accuracies, the first being
        # the direct prediction of the model, and the second computed by comparing predicted
        # labels of the images.
        direct_accuracies = []
        indirect_accuracies = []
        for inputs, targets in self._prepare_batches(dataset.batches(args.batch_size)):
            test = self.model.predict(inputs)
            pred_1 = np.argmax(test[0],axis=-1)
            pred_2 = np.argmax(test[1],axis=-1)
            dire   = test[2] >= 0.5
            idire  = pred_1 > pred_2
            assert len(dire) == len(targets[2])
            assert len(idire) == len(targets[2])
            direct_accuracies.append(np.sum(targets[2] == dire)/len(dire))
            indirect_accuracies.append(np.sum(targets[2] == idire)/len(idire))
        return np.mean(direct_accuracies), np.mean(indirect_accuracies)

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
    parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = lambda: tf.keras.initializers.glorot_uniform(seed=42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    mnist = MNIST()

    # Create the network and train
    network = Network(args)
    network.train(mnist, args)
    with open("mnist_multiple.out", "w") as out_file:
        direct, indirect = network.evaluate(mnist.test, args)
        print("{:.2f} {:.2f}".format(100 * direct, 100 * indirect), file=out_file)
