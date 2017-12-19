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


result = lines2struct()
document_label = [int(label[1]) for label in result][:Config.holding]
del result,lines2struct,read_data_preprocessed

import get_fixed_doc
doc = get_fixed_doc.get_fixed_doc()

document_label = np.array(document_label)
document_label -=1
from sklearn.cluster import KMeans
est = KMeans(n_clusters=2)
est.fit(doc)
print(np.mean(est.labels_ == document_label))
