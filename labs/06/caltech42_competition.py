#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub # Note: you need to install tensorflow_hub

from caltech42 import Caltech42

# The neural network model
class Network:

    @staticmethod
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
      elif arg.startswith('Dr'):
          return tf.keras.layers.Dropout(rate=0.5)(inputs)
      else:
        raise Exception('Unknown cnn argument {}'.format(arg))

    def __init__(self, args):
        # TODO: You should define `self.model`. You should use the following layer:
        #   mobilenet = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280])
        # The layer:
        # - if given `trainable=True/False` to KerasLayer constructor, the layer weights
        #   either are marked or not marked as updatable by an optimizer;
        # - however, batch normalization regime is set independently, by `training=True/False`
        #   passed during layer execution.
        #
        # Therefore, to not train the layer at all, you should use
        # On the other hand, to fully train it, you should use
        #   mobilenet = tfhub.KerasLayer(
        # "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2",
        # output_shape=[1280],
        # trainable=True)
        #   features = mobilenet(inputs)
        # where the `training` argument to `mobilenet` is passed automatically in that case.
        # Note that a model with KerasLayer can currently be saved only using
        #   tf.keras.experimental.export_saved_model(model, path, serving_only=True/False)
        # where `serving_only` controls whether only prediction, or also training/evaluation
        # graphs are saved. To again load the model, use
        #   model = tf.keras.experimental.load_from_saved_model(path, {"KerasLayer": tfhub.KerasLayer})
        ## TODO: Check the input layer
        inputs = tf.keras.layers.Input(shape=[224, 224, Caltech42.C])
        mobilenet = tfhub.KerasLayer(
                 "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2",
                 output_shape=[1280],
                 trainable=False)
        features = mobilenet(inputs, training=False)
        hidden = features

        for l in filter(None, args.nn.split(",")):
            hidden = self.get_layer(l, hidden)

        flatten = tf.keras.layers.Flatten()(hidden)
        outputs = tf.keras.layers.Dense(42, activation="softmax")(hidden)
        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss=tf.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.metrics.CategoricalAccuracy()]
                )

    def train(self, caltech42, args):
        #for i in range(args.epochs):
        #    for batch in caltech42.train.batches(args.batch_size):
        #           train = model.train_on_batch(
        #                   batch["image
        self.model.fit(
              x=caltech42.train.data["images"],
              y=caltech42.train.data["labels"],
              batch_size=args.batch_size,
              epochs=args.epochs,
              validation_data=(caltech42.dev.data["images"],caltech42.dev.data["labels"])
          )

    def predict(self, caltech42, args):
        self.model.predict(caltech42.test.data["images"])

    def predict_dev(self, caltech42, args):
        self.model.predict(caltech42.dev.data["images"])


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--nn", default="",type=str, help="Shared convolution layers")
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

    # TODO: preprocess
    caltech42 = Caltech42()    # Create the network and train
    # image_processing=lambda x: tf.image.decode_image(x, channels=3))
    caltech42 = Caltech42(image_processing=lambda x:
            tf.image.resize(tf.image.decode_image(x,
                channels=3),[224,224]).numpy())


    network = Network(args)
    network.train(caltech42, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "caltech42_competition_test.txt"), "w", encoding="utf-8") as out_file:
        for probs in network.predict(caltech42.test, args):
            print(np.argmax(probs), file=out_file)
