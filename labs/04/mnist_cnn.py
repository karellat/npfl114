#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

from mnist import MNIST

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
        # TODO: Quick fix change
        if  args.cnn is None:
            res_cnn_args = ""
        else:
            res_cnn_args = ""
            in_brack = False
            for c in args.cnn:
                if c == "," and in_brack:
                    res_cnn_args += ";"
                elif c == "[":
                    in_brack = True
                    res_cnn_args += c
                elif c == "]":
                    in_brack = False
                    res_cnn_args += c
                else:
                    res_cnn_args += c
        print(res_cnn_args) 
        inputs = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])
        hidden = inputs
        for a in res_cnn_args.split(","):
            hidden = get_layer(a, hidden)
        # TODO: Add CNN layers specified by `args.cnn`, which contains
        # comma-separated list of the following layers:
        # - `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
        #   activation and specified number of filters, kernel size, stride and padding.
        # - `CB-filters-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
        #   In detail, start with a convolutional layer without bias and activation,
        #   then add batch normalization layer, and finally ReLU activation.
        # - `M-kernel_size-stride`: Add max pooling with specified size and stride.
        # - `R-[layers]`: Add a residual connection. The `layers` contain a specification
        #   of at least one convolutional layer (but not a recursive residual connection `R`).
        #   The input to the specified layers is then added to their output.
        # - `F`: Flatten inputs. Must appear exactly once in the architecture.
        # - `D-hidden_layer_size`: Add a dense layer with ReLU activation and specified size.
        # Produce the results in variable `hidden`.

        # Add the final output layer
        outputs = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)(hidden)

        super().__init__(inputs=inputs, outputs=outputs)

        self.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    def train(self, mnist, args):
        self.fit(
            mnist.train.data["images"], mnist.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
            callbacks=[self.tb_callback],
        )

    def test(self, mnist, args):
        test_logs = self.evaluate(mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size)
        self.tb_callback.on_epoch_end(1, dict(("val_test_" + metric, value) for metric, value in zip(self.metrics_names, test_logs)))
        return test_logs[self.metrics_names.index("accuracy")]


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--cnn", default=None, type=str, help="CNN architecture.")
    parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
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

    # Compute test set accuracy and print it
    accuracy = network.test(mnist, args)
    with open("mnist_cnn.out", "w") as out_file:
        print("{:.2f}".format(100 * accuracy), file=out_file)
