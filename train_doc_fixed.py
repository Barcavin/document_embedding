import numpy as np
import tensorflow as tf
import collections
import math
import time
import os

import Config

holding = Config.holding # Number of documents to train
epochs = 50
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



def build_dataset(struct=lines2struct(),n_words=Config.vocabulary_size):
    """
        Transform the word to integer token. Build the dictionary between word and integer token.
        All words that are not top N-frequent will be marked as 'UNK'
        dictionary : {word:index}
        reverse_dictionary : {index:word}
    """
    count = [['UNK', -1]]
    words = [word for each in struct for word in each[2]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for each in struct:
        one_data = list()
        for word in each[2]:
            index = dictionary.get(word,0)
            if index == 0:
                unk_count += 1
            one_data.append(index)
        data.append([each[0],each[1],one_data])
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


data, count, dictionary,reverse_dictionary = build_dataset()
del read_data_preprocessed,lines2struct,build_dataset

document_index = 0 # Index indicating which document is being used
data_index = 0 # Index in corresponding document
epoch = 0

def generate_batch(batch_size=Config.batch_size,skip_window=Config.skip_window): # CBOW batch
    global document_index
    global data_index
    global epoch
    batch = np.ndarray(shape=(batch_size),dtype = np.int32)
    document_label = np.ndarray(shape=(batch_size),dtype = np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype = np.int32)
    span = 2 * skip_window + 1
    # load = collections.deque(maxlen=span) # Queue container
    strides = np.ceil(batch_size/2/skip_window)
    i = 0
    temp = list() # [[doc,[batch],target]
    while strides > 0:
        indicate = len(data[document_index][2]) - data_index - span + 1
        if indicate <= 0:
            document_index += 1
            data_index = 0
            if document_index > holding:
                document_index = 0
                epoch += 1
            continue
        whole = data[document_index][2][data_index:data_index+span]
        target = whole[skip_window]
        del whole[skip_window]
        temp.append([document_index,whole,target])
        data_index += 1
        strides -= 1
    for each in temp:
        for e in each[1]:
            batch[i] = e
            document_label[i] = each[0]
            labels[i,0] = each[2]
            i += 1
            if i==batch_size:
                break
    return batch,document_label,labels

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

restore = False
if os.path.exists(Config.model_path):
    restore = True
    model_path  = tf.train.latest_checkpoint(Config.model_path)
    start_of_number = model_path.index('-')+1
    document_index = int(model_path[start_of_number:])

with tf.Session(graph=new_graph) as session:
  # Initialized all the variables in our graph:
  if restore:
      saver.restore(session,model_path)
      print("Restored")
  else:
      init.run()
      print('Initialized')
  step = 0
  while epoch < epochs:
      step += 1
      batch_inputs, batch_doc ,batch_labels = generate_batch(
          Config.batch_size, Config.skip_window)
      feed_dict = {train_inputs: batch_inputs,train_document:batch_doc ,train_labels: batch_labels}


      _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
      if step % 2000==0:
          print("step %i,epoch %i,document %i"%(step,epoch,document_index))
      if step % 50000 == 0 :
          saver.save(session,Config.model_path+'doc',global_step=document_index)
