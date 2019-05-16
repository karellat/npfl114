# 8. prednaska
## 3d recognition
* alternativni postupu, udelat nekolik rezu ve 2D a pak delat 2D recognition(State of the art) 
## tagger_competition
### original tagger_cle, LSTM 256, WE 256, CLEA 128
* overfittuje 
	* LayerNorm - 3x pomalejsi 
	* dropout 0.5, LSTM 512, WE 512, CLE 256 [less overfitting] 
	* embedding dropout 0.5, 2 LSTM layers
	* WE+CLE concat, residual before dropout
	* + dropout before residual 
	* + label smoothing 0.1 [overfitting stopped]
	* lazy Adam, 30 epoch
	* beta_2  0.98, 40 epocs
	* varitional dropout
	* word dropout 0.15
	* beta 0.99 
	* finetunning with  1e-4 lr
	....
* tf.addons
	* lazy Adama
	* udpipe-future

## Deep Generative Models
* set X of realization of a random variable x
* P(x) 
* X ~ P(X)
* zavisla na latetni(skryte) promenne

## AutoEncoder 
* X -> z -> X 
* X -> Z = Encoder
* Z -> X = decoder
* unsupervised feature extraction
* input compression for z < x
* when x + e is used as input, autoencoders can perform denoising

##  Variational AutoEncoders
* We assume P(z) is fixed and independent on x. 
* We approximate P(x|z) using P_0(x|z). However in order to train an autoencdoder, we need to know the posterior P_0(z|x), which is usually intractable. 
* We therefore approximate P_0(z|x) by a trainable Q_alpha(z|x)

KL(p||q) = E_p log (q(x)/p(x))

GAN 
Generator- from z to x
Discriminator - from x to (fake or real)

