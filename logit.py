import tensorflow as tf
import numpy as np
import Config
import math
import time
import os

logit_batch = 400
epochs = 2000
units = 30
nclass = 2
learning_rate = 0.02

holding = Config.holding

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



    result = lines2struct()
    document_label = [int(label[1]) for label in result][:holding]
    del result,lines2struct,read_data_preprocessed
    import get_fixed_doc
    doc = get_fixed_doc.get_fixed_doc()[:holding]

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
    if train_index + batch_size >= holding:
        beyond = True
        train_index = train_index + batch_size - holding
        epoch += 1
    else:
        train_index += batch_size
    if beyond:
        return np.vstack((doc[i:,],doc[:train_index,])) , np.hstack((document_label[i:],document_label[:train_index]))
    else:
        return doc[i:train_index,],document_label[i:train_index]

doc,document_label = get_doc_labels()

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
with tf.Session(graph=logit_graph) as session:
    init.run(session=session)
    print("Initialized")
    plot_result = list()
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
            plot_result.append([step,average_loss])
            average_loss = 0
    print("Complete Train")
    print("Saving the model")
    # saver.save(session,'logit/')
    print("Calculate accuracy")
    feed_dict = {train_inputs:doc}
    pred = session.run(prediction,feed_dict = feed_dict)
    acc = np.mean(pred==document_label)
    print("The accuracy is %f" % acc)
