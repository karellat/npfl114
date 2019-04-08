# 5. cviceni
## Mnist_cnn
* technical hw 
* R- remember residual input and for ] - add residual connection
## Cifar_competion

## Newtasks
### TensorFlow hub 
* exchange the pretrained models 
* new pip package
* cannot derive the **output_shape**, but same as Keras Layers
* tf.image.decode_image - from jpgs, png to the data format
* example: 
	* mobilenet - classification on ImageNet
		* faster, for phones
### caltech42_competition
* small dataset, but sizes 224 or more
* use pretrained network (MobileNet2)
* 94 % threshold
* subset Caltech42
* dataset stored in zip 
* dateset image processing function -> You can specify place to be run
* image preprocessing -> ds image processing after 

		tfTensor.numpy(C) -> return np array

* pytorch variant for mobile exists too
* trainable switch if the original network should be normalized
* batch regime is independent on normalization 
* saving model - tf.keras.experimental.export_saved_model
	* contains protobuff of symbolic graph
	* weights 
	* serving_only - True just real weights, all support variables will be saved 
	* currently there are json

### sequence_classification
* recurrent neural network 
* task is to compare two sequence
* different types of the memorizing cell, dimension of these cells
* sequence_dim - size of onehot
* **tf.SimpleRNN** - is cell but can be used for sequence too
	* return sequence
		* False -> get state
		* True -> output 
* **tf.SimpleRNNCell** - just single
* **tf.LSTMCell** - just single
* **tf.LSTM** - same as SimpleRNN but with LSTM
* GRU - relativily to previous once
* Use method specific due to computation issues

