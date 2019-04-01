# 04 Cviceni

## Weighting average of test data 
* Sometimes it is usefull to weight classes examples
* usefull for unbalanced classes 

## Ensemble 
* you can use models as layers

## Practic implementation - MNIST DEMO
* normalization same on the test data(Relu does not mind too much) 
* repository contains example of mnist and running it in web-browser
* aware of the crop

## New Tasks

### More advance stuff without keras

#### mnist_mupltiple
* create your own model without keras
* inputs - 2 
* outputs - 3 
* multiple losses, array of metrics  - possible to do in keras 
* try direct and in-direct prediction 
* compute their accuracy 
	* multiple input/outputs 
		* The losses, you have to implement it manually 
* **_prepare_batches** prepare pairs of the images and 3 labels
* do not use fit, use only model.train_on_batches 
* **evaluate** method
#### Competition Fashion MNIST dataset - fashion mnist 
* same format as the original Mnists
* our dataset has noises
* you have to predict the class and mask 
* small dataset 
* needs big computation power
* one .txt and .py 
* IoU - wrong class nothing, mean of IoU over all the pictures
* eval metrics could show errors
* for 5 points 75% (more then the hour) 
* Anotation: 
	* first on the line class space than all rows of mask in a single line 
* how to teach only one of the layers(not simple by keras) 
* two separable models could be better
* simple solution 
### Student cluster
* during april 
* metacentrum
* Colab (2Gb, 8 hours) 
