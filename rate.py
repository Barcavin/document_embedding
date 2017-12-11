import tensorflow as tf
import numpy as np
import Config
import math
import time
import os


def read_data_preprocessed(dataset_name=Config.dataset_name,train=True):
    """
    return lines:
                    lines[i][0] is the label
                    lines[i][-1] is the document_index
    """
    if train:
        train_path = 'train'
    else:
        train_path = 'test'
    path = 'data/'+train_path+"_"+dataset_name+".txt"
    with open(path,'r') as h:
        content = h.readlines()
    lines = [x.split()+[i] for i,x in enumerate(content)]
    return lines

def lines2struct(l=read_data_preprocessed()):
    """
        return a data structure which looks like : [[index,label,[words]],...,]
    """
    result = list()
    for each in l:
        index = each[-1]
        label = each[0]
        dataflow = each[1:-1]
        result.append([index,label,dataflow])
    return result


model_path  = tf.train.latest_checkpoint(Config.model_path)
document_index = int(model_path[10:])
result = lines2struct()
document_size = len(result)
document_label = [int(label[1]) for label in result][:document_index]
del result,lines2struct,read_data_preprocessed
# =============================================================
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
# Build the graph
# ===================================================================================================================================================================
graph = tf.Graph()
with graph.as_default():

    train_inputs = tf.placeholder(tf.int32,shape=[Config.batch_size])
    train_document = tf.placeholder(tf.int32,shape=[Config.batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[Config.batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # with tf.device('/device:GPU:0'):
    with tf.device('/cpu:0'): # Use CPU instead if the GPU memory is not large enough to store all the parameters
        embeddings = tf.Variable(tf.random_uniform([Config.vocabulary_size, Config.embedding_size], -1.0, 1.0),name='embedding')
        embed = tf.nn.embedding_lookup(embeddings, train_inputs) # 128x512 batch_size x embedding_size

        document_embeddings = tf.Variable(tf.random_uniform([document_size, Config.document_embedding_size], -1.0, 1.0),name='document_embedding')
        document_embed = tf.nn.embedding_lookup(document_embeddings, train_document)

        nce_weights = tf.Variable(
            tf.truncated_normal([Config.vocabulary_size, Config.embedding_size + Config.document_embedding_size ],
                                stddev=1.0 / math.sqrt(Config.embedding_size + Config.document_embedding_size)),name = "softmax_weight")
        nce_biases = tf.Variable(tf.zeros([Config.vocabulary_size]),name = "softmax_biases")

        total_embed = tf.concat([embed,document_embed],1)
    # ====================================================================
    # loss = tf.reduce_mean(
    #     tf.nn.nce_loss(weights=nce_weights,
    #                    biases=nce_biases,
    #                    labels=train_labels,
    #                    inputs=embed,
    #                    num_sampled=Config.num_sampled,
    #                    num_classes=Config.vocabulary_size))
    # ====================================================
    # logits = tf.matmul(embed, tf.transpose(nce_weights))
    # logits = tf.nn.bias_add(logits, nce_biases)
    # labels_one_hot = tf.one_hot(train_labels, Config.vocabulary_size)
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    #   labels=labels_one_hot,
    #   logits=logits))
    # =====================================================================
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
        weights=nce_weights,
        biases=nce_biases,
        labels=train_labels,
        inputs=total_embed,
        num_sampled=Config.num_sampled,
        num_classes=Config.vocabulary_size))
    # optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = tf.Variable(embeddings / norm)
    doc_norm = tf.sqrt(tf.reduce_sum(tf.square(document_embeddings), 1, keep_dims=True))
    doc_normalized_embeddings = tf.Variable(document_embeddings / doc_norm)

    saver = tf.train.Saver()
    # saver = tf.train.Saver(max_to_keep = 3)
    #
    temp_normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        temp_normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, temp_normalized_embeddings, transpose_b=True)

    init = tf.global_variables_initializer()

# ===========================================================================================================

with tf.Session(graph=graph) as session:
    saver.restore(session,model_path)
    doc = doc_normalized_embeddings.eval()
doc = doc[:document_index,:]
document_label = np.array(document_label)
document_label -=1
from sklearn.cluster import KMeans
est = KMeans(n_clusters=2)
est.fit(doc)
print(np.mean(est.labels_ == document_label))
