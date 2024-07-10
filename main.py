import os
import json
import time
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors


class Mlsmote():
    def __init__(self, data, labels, n, k):
        
#         args:
#         data : the input feature of data, which should be a ndarray, you can use np.array to transfer a list into ndarray
#                the shape of this input should be (data_size, .......)
#                which means the 1st dimesion of your input should be the number of your data instance
#                for example:
#                if it's a deep learning task, the 1st dimesion maybe the batch size    
#                  [64, 3, 224, 224]
#         labels : the labels for your input, should be a ndarray and data type needs to be int
#                  the shape of this input should be (data_size, class_number)
#         n : the nearest neighbours you want use when generating a new sample
#         k : the number of new samples you want to generate with a single tail-class data instance

        super(Mlsmote, self).__init__()

        self.data = data
        self.labels = labels
        self.N = n
        self.K = k
        self.neg_dict = {}
        self.cls_num = self.labels.shape[1]
        
        self.tail_label = self.get_tail_label(self.labels)
        
        # create a dict to store the neighbour indices for datas in each tail class

        for label in self.tail_label:
            self.neg_dict[str(label)] = []

        print("self.N is {}".format(self.N))
        print("self.tail cls is {}".format(self.tail_label))


    def get_tail_label(self, labels):
    
    # Give tail label colums of the given label
    
        irpl = np.zeros(self.cls_num)
        
        for cls in range(self.cls_num):
            irpl[cls] = sum(labels[:, cls])
            print("cls : {} has {} samples".format(cls, irpl[cls]))

        irpl = max(irpl)/irpl
        mir = np.average(irpl)
        tail_label = []

        for i in range(self.cls_num):
            if irpl[i] > mir:
                tail_label.append(i)
        return tail_label
        
# get the indices of all instances within a certain class
        
    def getclassdata(self, class_):
        data_index = []
        for i in range(self.data.shape[0]):
            ins_data_label = self.labels[i]
            if ins_data_label[class_] == 0:
                continue
            else:
                data_index.append(i)
        return data_index

# calculate the k-nearest-neighbour

    def knn(self):
        start_time = time.time()

        for class_ in self.tail_label:
            # choose a class from tail classes
            # get all the data and labels of this class
            data_index = self.getclassdata(class_)
            data_ = self.data[data_index]
            label_ = self.labels[data_index]
            
            # reshape the data, to fit the input of sklearn
            data_ = data_.reshape(data_.shape[0], -1)

            nbrs = NearestNeighbors(n_neighbors = self.K, algorithm = 'ball_tree') \
                .fit(data_)
            
            # if return_distance is True, you can also get the distances between each data points
            # this indices is not the direct index for the data
            # for example
            # if I chose class 3, and the data index is like:
            # [3, 4, 15, 32, 111, ......] this is the indices of samples whose 3rd class is positive
            # and the indices below maybe liks [1, 3, 5, 12, 34, .....]
            # and indices[2] = 3, 3 means the 3rd samples in the data_, which shoule be data[15]
            indices = nbrs.kneighbors(data_, return_distance = False)
            
            self.neg_dict[str(class_)] =  indices

        end_time = time.time()

        print("time total knn {}".format(end_time - start_time))
        
 # generating relevant labels for new sample

    def syn_label(self, ins_nn, data_index):
        syn_label = np.zeros((self.cls_num))
        for i in range(len(ins_nn)):
            # ins_nn is list, which contains the index of nearest-neighbour of this sample
            ins_n_label = self.labels[data_index[ins_nn[i]]]
            syn_label += ins_n_label
        for i in range(self.cls_num):
            if syn_label[i] >= (self.K / 2):
                syn_label[i] = 1
            else:
                syn_label[i] = 0
        return syn_label

    def MlS(self):
        self.knn()
        new_feature = []
        new_label = []
        for class_ in self.tail_label:
            data_index = self.getclassdata(class_)
            data_ = self.data[data_index]
            label_ = self.labels[data_index]

            for i in range(len(data_index)):
                ins_data_base = data_[i]
                ins_nn = self.neg_dict[str(class_)][i][1:]
                label_syn = self.syn_label(ins_nn, data_index)
                
                for times in range(self.N):
                    nei_ref = random.randint(0, len(ins_nn) - 1)
                    delta = random.uniform(0, 1)
                    ins_n_data = self.data[data_index[ins_nn[nei_ref]]]
                    
                    gap = ins_n_data - ins_data_base
                    new_ins_feature = ins_data_base + delta * gap
                    
                    new_feature.append(new_ins_feature)
                    new_label.append(label_syn)
                    
        
        new_shape = [-1]
        
        for i in range(1, len(self.data.shape)):
            new_shape.append(self.data.shape[i])

        new_feature = np.array(new_feature).reshape(new_shape)
        new_label = np.array(new_label)

        data_wrap = np.concatenate((self.data, new_feature), axis = 0)
        labels_warp = np.concatenate((self.labels, new_label), axis = 0)

        return data_wrap, labels_warp
