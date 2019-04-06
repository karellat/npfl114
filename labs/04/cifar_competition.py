#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from cifar10 import CIFAR10

def get_layer(arg, inputs):
    C_args = arg.split('-')
    if arg.startswith('C-'):
        return tf.keras.layers.Conv2D(
                int(C_args[1]),
                int(C_args[2]),
                int(C_args[3]),
                padding=C_args[4],
                activation="relu")(inputs)
    elif arg.startswith('CB-'):
        new_layer = tf.keras.layers.Conv2D(
                int(C_args[1]),
                int(C_args[2]),
                int(C_args[3]),
                padding=C_args[4],
                use_bias=False)(inputs)
        new_layer = tf.keras.layers.BatchNormalization()(new_layer)
        return tf.keras.layers.Activation("relu")(new_layer)
    elif arg.startswith('M-'):
       return tf.keras.layers.MaxPool2D(
           int(C_args[1]),
           int(C_args[2]))(inputs)
    elif arg.startswith('R-'):
        assert len(arg[3:-1].split(';')) != 0
        new_layer = inputs
        print(arg[3:-1])
        for a in arg[3:-1].split(';'):
            new_layer = get_layer(a, new_layer)
        return tf.keras.layers.Add()([new_layer, inputs])
    elif arg.startswith('D-'):
        return tf.keras.layers.Dense(
           int(C_args[1]),
            activation="relu")(inputs)
    elif arg.startswith('F'):
        return tf.keras.layers.Flatten()(inputs)
    else:
        raise Exception('Unknown cnn argument {}'.format(arg))

# The neural network model
class Network(tf.keras.Model):
    def __init__(self, args):
        # TODO: Define a suitable model, by calling `super().__init__`
        # with appropriate inputs and outputs.
        #
        # Alternatively, if you prefer to use a `tf.keras.Sequential`,
        # replace the `Network` parent, call `super().__init__` at the beginning
        # of this constructor and add layers using `self.add`.

        # TODO: After creating the model, call `self.compile` with appropriate arguments.
        inputs = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])
        hidden = inputs
        for layer in args.cnn.split(","):
            hidden = get_layer(layer, hidden)
        outputs = tf.keras.layers.Dense(CIFAR10.LABELS,
                activation=tf.nn.softmax)(hidden)
        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None
        super().__init__(inputs=inputs, outputs=outputs)

        self.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )


    def train(self, cifar, args,call_backs):
        self.fit(
            cifar.train.data["images"], cifar.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(cifar.dev.data["images"], cifar.dev.data["labels"]),
            callbacks=call_backs,

        )


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--cnn", default="F", type=str,help="Layer after layer architecture of the network")
    args = parser.parse_args()

    # Fix random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    cifar = CIFAR10()

    # Create the network and train
    network = Network(args)
    network.train(cifar, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as out_file:
        for probs in network.predict(cifar.test.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=out_file)
