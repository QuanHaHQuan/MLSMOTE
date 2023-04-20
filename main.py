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
    def __init__(self, data, labels, tail_list, n, k):
        super(Mlsmote, self).__init__()

        self.data = data
        self.N = n
        self.K = k
        self.labels = labels
        self.tail_list = tail_list
        self.neg_dict = {}
        self.cls_num = self.labels.shape[1]

        for label in tail_list:
            self.neg_dict[str(label)] = []

        print("self.N is {}".format(self.N))
        
    def getclassdata(self, class_):
        data_index = []
        for i in range(self.data.shape[0]):
            ins_data_label = self.labels[i]
            if ins_data_label[class_] == 0:
                continue
            else:
                data_index.append(i)
        return data_index

    def knn(self):
        start_time = time.time()

        for class_ in self.tail_list:
            data_index = self.getclassdata(class_)
            data_ = self.data[data_index]
            label_ = self.labels[data_index]

            data_ = data_.reshape(data_.shape[0], -1)

            nbrs = NearestNeighbors(n_neighbors = self.K, algorithm = 'ball_tree') \
                .fit(data_)
            indices = nbrs.kneighbors(data_, return_distance = False)
            
            self.neg_dict[str(class_)] =  indices

        end_time = time.time()

        print("time total knn {}".format(end_time - start_time))

    def syn_label(self, ins_nn, data_index):
        syn_label = np.zeros((self.cls_num))
        for i in range(len(ins_nn)):
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
        for class_ in self.tail_list:
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
                    
        new_feature = np.array(new_feature)
        new_label = np.array(new_label)

        data_wrap = np.concatenate((self.data, new_feature), axis = 0)
        labels_warp = np.concatenate((self.labels, new_label), axis = 0)

        return data_wrap, labels_warp
