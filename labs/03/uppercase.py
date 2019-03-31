#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=0, type=int, help="If nonzero, limit alphabet to this many most frequent chars.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
parser.add_argument("--hidden_layer", default=200, type=int, help="Size of the hidden layer.")
parser.add_argument("--batch_size", default=1024, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default="500", type=str, help="Hidden layer configuration.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=10, type=int, help="Window size to use.")
args = parser.parse_args()
args.hidden_layers = [int(hidden_layer) for hidden_layer in args.hidden_layers.split(",") if hidden_layer]

args.logdir = os.path.join("logs", "{}-{}-{}".format(
    os.path.basename(__file__),
    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
))

# Fix random seeds
np.random.seed(42)
tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(args.threads)
tf.config.threading.set_intra_op_parallelism_threads(args.threads)

# Load data
uppercase_data = UppercaseData(args.window, args.alphabet_size)

train_data = uppercase_data.train.data["windows"]
train_labels = uppercase_data.train.data["labels"]
batch_size = args.batch_size

dev_data = uppercase_data.dev.data["windows"]
dev_labels = uppercase_data.dev.data["labels"]


# print(train_data.shape)
# print(train_labels.shape)
# print(dev_data.shape)
# print(dev_labels.shape) 
# print(batch_size)

inputs = tf.keras.layers.Input(shape=(2 * args.window + 1), dtype=tf.int32)
encoded = tf.one_hot(inputs, len(uppercase_data.train.alphabet))
flattened = tf.keras.layers.Flatten()(encoded)
outputs = tf.keras.layers.Dense(1, activation=tf.nn.softmax)(flattened)
model = tf.keras.Model(inputs=inputs, outputs=outputs) 

print(inputs.shape)
model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
)

model.fit(
        train_data, train_labels,
        batch_size=batch_size,
        epochs=args.epochs,
        validation_data=(dev_data, dev_labels),
)


# TODO: Implement a suitable model, optionally including regularization, select
# good hyperparameters and train the model.
#
# The inputs are _windows_ of fixed size (`args.window` characters on left,
# the character in question, and `args.window` characters on right), where
# each character is representedy by a `tf.int32` index. To suitably represent
# the characters, you can:
# - Convert the character indices into _one-hot encoding_. There is no
#   explicit Keras layer, so you can
#   - use a Lambda layer which can encompass any function:
#       Sequential([
#         tf.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32),
#         tf.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
#   - or use Functional API and a code looking like
#       inputs = tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32)
#       encoded = tf.one_hot(inputs, len(uppercase_data.train.alphabet))
#   You can then flatten the one-hot encoded windows and follow with a dense layer.
# - Alternatively, you can use `tf.keras.layers.Embedding`, which is an efficient
#   implementation of one-hot encoding followed by a Dense layer, and flatten afterwards.

with open("uppercase_test.txt", "w", encoding="utf-8") as out_file:
    # TODO: Generate correctly capitalized test set.
    # Use `uppercase_data.test.text` as input, capitalize suitable characters,
    # and write the result to `uppercase_test.txt` file.
    pass
