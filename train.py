import numpy as np
import tensorflow as tf
import collections
import math
from six.moves import xrange
import time

import Config



def read_data_preprocessed(dataset_name=Config.dataset_name,train=False):
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


document_index = 0 # Index indicating which document is being used
data_index = 0 # Index in corresponding document

def generate_batch(batch_size=Config.batch_size,skip_window=Config.skip_window): # CBOW batch
    global document_index
    global data_index
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
            if document_index == len(data):
                document_index = 0
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

# =============================================================
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

# Build the graph.
graph = tf.Graph()

with graph.as_default():

    train_inputs = tf.placeholder(tf.int32,shape=[Config.batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[Config.batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device('/device:GPU:0'):
    # with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform([Config.vocabulary_size, Config.embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs) # 128x512 batch_size x embedding_size

        nce_weights = tf.Variable(
            tf.truncated_normal([Config.vocabulary_size, Config.embedding_size],
                                stddev=1.0 / math.sqrt(Config.embedding_size)))
        nce_biases = tf.Variable(tf.zeros([Config.vocabulary_size]))
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
        inputs=embed,
        num_sampled=Config.num_sampled,
        num_classes=Config.vocabulary_size))
    # optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm

    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    init = tf.global_variables_initializer()

num_steps = Config.num_steps

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, _ ,batch_labels = generate_batch(
        Config.batch_size, Config.skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    if step % 10000 == 0:
         sim = similarity.eval()
         for i in xrange(valid_size):
             valid_word = reverse_dictionary[valid_examples[i]]
             top_k = 8  # number of nearest neighbors
             nearest = (-sim[i, :]).argsort()[1:top_k + 1]
             log_str = 'Nearest to %s:' % valid_word
             for k in xrange(top_k):
                 close_word = reverse_dictionary[nearest[k]]
                 log_str = '%s %s,' % (log_str, close_word)
             print(log_str)
  final_embeddings = normalized_embeddings.eval()


def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels, os.path.join(os.getcwd(), str(int(time.time()))+'tsne.png'))

except ImportError as ex:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  print(ex)
