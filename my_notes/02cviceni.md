# 02 cvičení 
* deadline extension possible

## Car Pole
* Recodex - Two Outputs layers
    * jeden lze rozsirit na 1 - p a p 

## Tensorflow 2.0a.0 - alpha
* weights of neurons in Keras is offen called kernel
* conda has to be forced to install 2.0 
* loss - SparseCategoricalCrossentropy  
    * only single label with 1.0 probability 
    * expecting distributions of labels
    * can accept logit(unormalize, output layer without softmax) 
### fit training
* manual version of training (less Keras more us)
    * train_on_batch 
    * test_on_batch 
* example manual 
    * tensorflow specific
    * sequential model
        * optimazer 
        * loss_fn 
        * accuracy 
    * iterate over the batches 
        * call the model (training=True; cause of the DropOut)
        * compute loss (give the gold, probabilities)
        * compute accuracy 
        * with tf.GradientTape() 
            * records all the operation during model 
            * can differentiate the mode; 
        * apply_gradients(all the gradients of the neurons)
    * 3 times slower
* example manual_tf 
    * with flag  **@tf.function** 
        * from the python source and create compution graph 
    * original speed 

### Sequential model 
* opposite API 
    * variables
    * inputs = somelayer()
    * flattened = somelayer()(inputs)
    * hidden1 = somelayer()(flattened)
    * hidden2 = somelayer()(hidden1)
    * concatenated = somelayer()([hidden_1, hidden_2])
    * outputs = someOutputLayer()(concatenated)
    * model = tf.keras.Model(inputs, outputs)
    * from this point is the same as Sequential model 

### mnist_regularization 
* implement dropout, L2, label smoothing 
* using only :5000 overfitting is much more likely 
* DROPOUT 
    * dropout layer
        * tf.keras.layers.Dropout
            * rate 
    * put after the Flatten layer and Dense layer
    * not after the output layer
* L2 - just set args (as you can see in the snippet)
* Smoothing 
    * trun off SparseCategorical -> to CategorialCrossentropy(label_smoothing=True)
    * use onehot for the labels
    * numpy OneHot or Keras method

##  mnist_ensemble 
* model arg = many models
* test accuracy M0 then test M[0,1]
* test accuracy M1 then test M[0,1]
* ....
* ..

* to do averaging - predict on all models then average the results
* use tf.keras.metrics.SparseCategoricalAccuracy
* manual: 
    inputs = tf.keras.layers.Inputs(sh...)
    o1 = m1(inputs)
    o2 = m2(inputs)
    o = average(o1,alpha)
    em = t.keras.Model() 

## uppercase
* 4 points for reaching the accuracy 
* 2 points relevant to the position in the best solutions 

## lambda Layer
* converts input given lambda function 