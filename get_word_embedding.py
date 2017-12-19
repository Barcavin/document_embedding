import numpy as np
import tensorflow as tf
import Config

def get_word_embedding(directory='word_embedding/'):
    
    new_graph = tf.Graph()

    with new_graph.as_default():
        word_embedding = tf.Variable(tf.random_uniform([Config.vocabulary_size, Config.embedding_size], -1.0, 1.0))
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    with tf.Session(graph = new_graph) as sess:
        saver.restore(sess,directory)
        result = word_embedding.eval()
    return result


