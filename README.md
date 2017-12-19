# File definition:

#### Config.py Configuration file that store the hyper parameters. Shared through several scripts.

#### eval.py Useless file for reader. Used for validating the training process.

#### get_fixed_doc.py Function to get the document embedding matrix. Called in logit.py to access the document embedding matrix. Call after complete train_doc_fixed.py

#### get_word_embedding.py Function to get the word embedding matrix. Called in train_doc_fixed.py. Call after complete train_doc.py

#### logit.py Single layer neural network. Feed the document embedding matrix to the network to train the classifier. It evaluates the performance of our model. Call it after complete train_doc_fixed.py

#### make_word_embedding.py Helper function to store work embedding matrix in a neater way.

#### rate.py Unsupervised classifer. Use k-means to classify our documents. Evaluate the performance of our model.

#### train_doc_fixed.py Train the document embedding matrix with the word embedding matrix fixed. Call after train_doc.py

#### train_doc.py Train the document embedding matrix and word embedding matrix simultaneously. The first train file to run. Stop it when the word embedding matrix has been trained enough.

# Reference
TensorFlow word embedding tutorial:
https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
