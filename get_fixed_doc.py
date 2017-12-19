def get_fixed_doc():
    import numpy as np
    import tensorflow as tf
    import collections
    import math
    import time
    import os

    import Config

    holding = Config.holding

    from get_word_embedding import get_word_embedding
    word_embedding_matrix = get_word_embedding()
    document_size = holding
    new_graph = tf.Graph()
    with new_graph.as_default():
        train_inputs = tf.placeholder(tf.int32,shape=[Config.batch_size])
        train_document = tf.placeholder(tf.int32,shape=[Config.batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[Config.batch_size, 1])

        embeddings = tf.constant(word_embedding_matrix,name='embedding')
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        document_embeddings = tf.Variable(tf.random_uniform([document_size, Config.document_embedding_size], -1.0, 1.0),name='document_embedding')
        document_embed = tf.nn.embedding_lookup(document_embeddings, train_document)

        nce_weights = tf.Variable(
            tf.truncated_normal([Config.vocabulary_size, Config.embedding_size + Config.document_embedding_size ],
                                stddev=1.0 / math.sqrt(Config.embedding_size + Config.document_embedding_size)),name = "softmax_weight")
        nce_biases = tf.Variable(tf.zeros([Config.vocabulary_size]),name = "softmax_biases")
        total_embed = tf.concat([embed,document_embed],1)

        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=train_labels,
            inputs=total_embed,
            num_sampled=Config.num_sampled,
            num_classes=Config.vocabulary_size))
        optimizer = tf.train.AdamOptimizer().minimize(loss)
        saver = tf.train.Saver(max_to_keep = 3)
        init = tf.global_variables_initializer()


    model_path  = tf.train.latest_checkpoint(Config.model_path)


    with tf.Session(graph=new_graph) as session:
        saver.restore(session,model_path)
        print("Restored")

        return document_embeddings.eval()
      
