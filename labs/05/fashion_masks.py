#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
print(tf.__version__)

from fashion_masks_data import FashionMasks

# TODO: Define a suitable model in the Network class.
# A suitable starting model contains some number of shared
# convolutional layers, followed by two heads, one predicting
# the label and the other one the masks.
class Network:
    def __init__(self, args):
      # Inputs 
      fashion_masks = FashionMasks()
      
      input_layer = tf.keras.layers.Input(shape=[fashion_masks.H, fashion_masks.W, fashion_masks.C]) 
      
      shared_cnn = input_layer
      # Shared layers  --cnn
      for l in filter(None, args.cnn.split(",")):
        shared_cnn = self.get_layer(l, shared_cnn)
      
      mask_last = shared_cnn
      # Fully connected for mask --masks
      for l in filter(None, args.masks.split(",")): 
         mask_last = self.get_layer(l, mask_last)
      
      label_last = shared_cnn
      # Fully connected for --labels 
      for l in filter(None, args.labels.split(",")): 
        label_last = self.get_layer(l, mask_last)
        
        
      # Labels prediction 
      flatten_pred =  tf.keras.layers.Flatten()(label_last)
      label_pred = tf.keras.layers.Dense(fashion_masks.LABELS, activation=tf.nn.softmax)(flatten_pred)
      # Masks predictions
      flatten_mask = tf.keras.layers.Flatten()(mask_last)
      mask_pred = tf.keras.layers.Dense(fashion_masks.H * fashion_masks.W, activation=tf.nn.sigmoid)(flatten_mask)
      mask_pred = tf.keras.layers.Reshape([fashion_masks.H, fashion_masks.W, 1])(mask_pred)
      #label_pred
      
      learning_rate_schedular = tf.keras.optimizers.schedules.ExponentialDecay(
        0.001,
        decay_steps=(args.batch_size*args.epochs),
        decay_rate=0.99,
        staircase=True)
      
      self.model = tf.keras.Model(inputs=input_layer, outputs=[label_pred,mask_pred])
      self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate_schedular),
            loss=[tf.losses.SparseCategoricalCrossentropy(name="label_crossentropy"), tf.losses.BinaryCrossentropy(name="masks_binarycrossentropy")]
      )
      
    def predict_dev(self, fashion_masks,args):
      return self.model.predict(fashion_mask.dev.data["images"])
    
    def predict(self, fashion_masks,args):
      return self.model.predict(fashion_mask.test.data["images"])
      
    def train(self, fashion_masks, args, callbacks):
      self.model.fit(
      x=fashion_masks.train.data["images"],
      y=[fashion_masks.train.data["labels"], fashion_masks.train.data["masks"]],
      batch_size=args.batch_size,
      validation_data=(fashion_masks.dev.data["images"],[fashion_masks.dev.data["labels"], fashion_masks.dev.data["masks"]]),
      epochs=args.epochs)
        
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


import argparse
import datetime
import os
import re

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--cnn", default="C-64-3-1-same,C-64-3-1-same,M-3-2,C-128-3-1-same,C-128-3-1-same,M-3-2,C-256-3-1-same,C-256-3-1-same,M-3-2",type=str, help="Shared convolution layers")
parser.add_argument("--labels",default="F,D-256,Dr", type=str, help="Predicting labels layers")
parser.add_argument("--masks", default="", type=str, help="Predicting masks layers")
args = parser.parse_args([])

# Fix random seeds
np.random.seed(42)
tf.random.set_seed(42)
#tf.config.threading.set_inter_op_parallelism_threads(args.threads)
#tf.config.threading.set_intra_op_parallelism_threads(args.threads)

# Create logdir name
args.logdir = os.path.join("logs", "{}-{}-{}".format(
    os.path.basename("fashion_mnist"),
    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
))

tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=1000, profile_batch=1)
tb_callback.on_train_end = lambda *_: None

# Load data
fashion_masks = FashionMasks()

# Create the network and train
network = Network(args)
network.train(fashion_masks, args, [])

# Predict test data in args.logdir
with open(os.path.join("fashion_masks_test.txt"), "w", encoding="utf-8") as out_file:
    # TODO: Predict labels and masks on fashion_masks.test.data["images"],
    # into test_labels and test_masks (test_masks is assumed to be
    # a Numpy array with values 0/1).
    test_labels, test_masks = network.predict(fashion_masks, args)
    test_labels = np.argmax(test_labels,axis=-1)
    test_masks = (test_masks > 0.5) * 1
    for label, mask in zip(test_labels, test_masks):
      print(label, *mask.astype(np.uint8).flatten(), file=out_file)
      
# Predict test data in args.logdir
with open(os.path.join("fashion_masks_dev.txt"), "w", encoding="utf-8") as out_file:
    # TODO: Predict labels and masks on fashion_masks.test.data["images"],
    # into test_labels and test_masks (test_masks is assumed to be
    # a Numpy array with values 0/1).
    test_labels, test_masks = network.predict_dev(fashion_masks, args)
    test_labels = np.argmax(test_labels,axis=-1)
    test_masks = (test_masks > 0.5) * 1
    for label, mask in zip(test_labels, test_masks):
      print(label, *mask.astype(np.uint8).flatten(), file=out_file)
  
!python fashion_masks_eval.py fashion_masks_dev.txt dev
