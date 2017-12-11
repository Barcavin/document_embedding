import tensorflow as tf
import numpy as np
import Config
import math
import time
import os

logit_batch = 200
epochs = 20
units = 20
nclass = 2
learning_rate = 0.02
new_path = 'logit_'+Config.model_path

def get_doc_labels():
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


        with tf.device('/cpu:0'):
            embeddings = tf.Variable(tf.random_uniform([Config.vocabulary_size, Config.embedding_size], -1.0, 1.0),name='embedding')
            embed = tf.nn.embedding_lookup(embeddings, train_inputs) # 128x512 batch_size x embedding_size

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
        # optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        optimizer = tf.train.AdamOptimizer().minimize(loss)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = tf.Variable(embeddings / norm)
        doc_norm = tf.sqrt(tf.reduce_sum(tf.square(document_embeddings), 1, keep_dims=True))
        doc_normalized_embeddings = tf.Variable(document_embeddings / doc_norm)

        saver = tf.train.Saver()
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
    document_label -=1 # 0,1


    return doc,document_label
train_index = 0
epoch = 0
def generate_batch(batch_size,doc,document_label):
    global train_index
    global epoch
    i = train_index
    beyond = False
    if train_index + batch_size >= doc.shape[0]:
        beyond = True
        train_index = train_index + batch_size - doc.shape[0]
        epoch += 1
    else:
        train_index += batch_size
    if beyond:
        return np.vstack((doc[i:,],doc[:train_index,])) , np.hstack((document_label[i:],document_label[:train_index]))
    else:
        return doc[i:train_index,],document_label[i:train_index]

doc,document_label = get_doc_labels()
document_index = doc.shape[0]

logit_graph = tf.Graph()
with logit_graph.as_default():
    train_inputs = tf.placeholder(tf.float32,shape = [None,Config.document_embedding_size],name = "vector")
    train_labels = tf.placeholder(tf.int32, shape = [None],name = "label")

    W_1 = tf.Variable(tf.random_uniform([Config.document_embedding_size, units], -1.0, 1.0))
    b_1 = tf.Variable(tf.zeros([units]))

    W_2 = tf.Variable(tf.truncated_normal([units, nclass],
                        stddev=1.0 / math.sqrt(nclass)))
    b_2 = tf.Variable(tf.zeros([nclass]))

    h = tf.nn.tanh(tf.matmul(train_inputs,W_1)+b_1) #logit_batch x units
    y = tf.matmul(h,W_2) + b_2
    one_hot_label = tf.one_hot(train_labels,nclass)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels = one_hot_label,
            logits = y))
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    init = tf.global_variables_initializer()
    prediction = tf.reshape(tf.cast(tf.argmax(y,1),tf.int32),[-1])
    # accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction,train_labels), tf.int32))
    saver = tf.train.Saver(max_to_keep = 3)

# ===================================================================================================
restore = False
##if os.path.exists(new_path):
##    restore = True
##    model_path  = tf.train.latest_checkpoint(new_path)

# with tf.Session(graph=logit_graph) as session:
session = tf.Session(graph=logit_graph)
if restore:
    saver.restore(session,model_path)
    print("Restored")
else:
    init.run(session=session)
    print("Initialized")

average_loss = 0
step = 0
while epoch < epochs:
    batch_inputs,batch_labels = generate_batch(logit_batch,doc,document_label)
    feed_dict = {train_inputs:batch_inputs,train_labels:batch_labels}
    _ , loss_val = session.run([optimizer, loss],feed_dict = feed_dict)
    average_loss += loss_val
    step += 1
    if step % 100 == 0:
        average_loss /= 100
        print('Average loss at step ', step, ': ', average_loss,", epoch:",epoch)
        average_loss = 0
##        if step % 500 ==0:
##            saver.save(session,new_path,global_step=epoch)
print("Complete Train")
print("Calculate accuracy")
feed_dict = {train_inputs:doc,train_labels:document_label}
pred = session.run(prediction,feed_dict = feed_dict)
acc = np.sum(pred==document_label)
print("The accuracy is %f" % acc[0])
