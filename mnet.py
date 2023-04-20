import os
import json
import time
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import transforms
from torchvision.models import resnet50

class Mnet(nn.Module):
    def __init__(self, ann_path, img_path, save_path_f, save_path_k, label_path, tail_list, n):
        super(Mnet, self).__init__()

        self.ann_path = ann_path
        self.img_path = img_path
        self.save_path_f = save_path_f
        self.save_path_k = save_path_k
        self.N = n

        self.label_path = label_path

        self.label = pd.read_csv(self.label_path, sep=',', header='infer', index_col=0)
        self.label = self.label.replace(-1, 1)
        self.label = self.label.replace(0, 0)
        self.label = self.label.fillna(0)

        self.num_classes = 14

        self.cnn_model = resnet50(pretrained=True)
        modules = list(self.cnn_model.children())[:-2]
        self.feature_extractor = nn.Sequential(*modules).cuda()
        # print(self.feature_extractor[0].weight.data)
        self.visual_feats_dim = 2048
        # self.hidden_size = 512
        self.hidden_size = 768
        self.hidden_dropout_prob = 0.1
        self.vis_embed = nn.Sequential(nn.Linear(self.visual_feats_dim, self.hidden_size * 2),
                                  nn.ReLU(),
                                  nn.Linear(self.hidden_size * 2, self.hidden_size),
                                  nn.ReLU(),
                                  nn.Dropout(self.hidden_dropout_prob))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls = nn.Sequential(nn.Linear(self.hidden_size, self.num_classes), nn.Sigmoid())
        # self.cls = nn.Sequential(nn.Linear(2048, self.num_classes), nn.Sigmoid())

        self.json_data = json.loads(open(self.ann_path, 'r').read())
        self.split = 'train'
        self.examples = self.json_data[self.split]

        self.tail_list = tail_list
        self.neg_dict = {}
        for label in tail_list:
            self.neg_dict[str(label)] = []

        self.is_train = False

        self.trans = transforms.Compose([
            transforms.Resize(256),       # 图像本身就是256*256
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        print("self.N is {}".format(self.N))

    def ef(self, batch_img_path_list, img_path):
        batch_image = []
        for ins_data_img_path in batch_img_path_list:
            ins_img_path = os.path.join(img_path, ins_data_img_path)
            ins_image = Image.open(ins_img_path).convert('RGB')
            ins_image = self.trans(ins_image)
            batch_image.append(ins_image)

        batch_image = torch.stack(batch_image, 0).cuda()
        vis_fea = self.vis_embed(self.feature_extractor(batch_image).reshape(-1, 7, 7, 2048))
        # vis_fea = (self.feature_extractor(batch_image).reshape(-1, 7, 7, 2048))
        vis_fea = vis_fea.tolist()

        for img_no_ef in range(len(batch_img_path_list)):
            ins_img_path = os.path.join(self.save_path_f, batch_img_path_list[img_no_ef])
            ins_img_path_file = ins_img_path[:-6]
            if not os.path.exists(ins_img_path_file):
                os.makedirs(ins_img_path_file)

            ins_img_path_ef = ins_img_path.replace("png", "npy")
            np.save(ins_img_path_ef, np.array(vis_fea[img_no_ef]))

    def convert(self, labels_raw):
        labels_pre = np.zeros_like(labels_raw)
        for i in range(len(labels_raw)):
            if labels_raw[i] == 3 or labels_raw[i] == 1:
                labels_pre[i] = 1
            else:
                labels_pre[i] = 0
        return labels_pre

    def getinslabel(self, label):
        ins_label = []
        for i in range(len(self.examples)):
            ins_data = self.examples[i]
            ins_data_img_path = ins_data['image_path']
            ins_data_label = ins_data['labels']
            if ins_data_label[label] == 0:
                continue
            ins_label.append(ins_data)
        return ins_label

    def getlabelindex(self, label, batch_labels):
        label_index = []
        for i in range(len(batch_labels)):
            ins_data_label = batch_labels[i]
            if ins_data_label[label] == 0:
                continue
            else:
                label_index.append(i)
        return label_index

    def knn(self):
        batch_img_path_list = []
        img_count = 0

        for i in range(len(self.examples)):
            self.examples[i]['labels'] = self.convert(self.examples[i]['labels'])

        start_time = time.time()
        for i in range(len(self.examples)):
            if i % 500 == 0:
                print("{} images done_ef".format(i))
            ins_data = self.examples[i]
            ins_data_img_path = ins_data['image_path']
            ins_data_label = ins_data['labels']
            if sum(torch.tensor(ins_data_label)[self.tail_list]) == 0:
                if i == (len(self.examples) - 1):
                    self.ef(batch_img_path_list, self.img_path)
                    batch_img_path_list = []
                else:
                    continue
            for img_no in range(len(ins_data_img_path)):
                ins_data_img_path_per = ins_data_img_path[img_no]
                batch_img_path_list.append((ins_data_img_path_per))
                img_count += 1

            if len(batch_img_path_list) >= 64 or i == (len(self.examples) - 1):
                self.ef(batch_img_path_list, self.img_path)
                batch_img_path_list = []

        print("total {} images".format(img_count))

        end_time = time.time()
        print("time total {}".format(end_time - start_time))

        start_time = time.time()

        for label in self.tail_list:
            vis_fea_knn_img = []
            vis_fea_knn = []
            fea_knn_no = 0
            label_data = self.getinslabel(label)
            for i in range(len(label_data)):
                ins_data = label_data[i]
                ins_data_img_path = ins_data['image_path']
                ins_data_label = ins_data['labels']
                for img_no in range(len(ins_data_img_path)):
                    ins_data_img_path_per = ins_data_img_path[img_no]
                    ins_img_path_fea = os.path.join(self.save_path_f, ins_data_img_path_per) \
                        .replace('png', 'npy')
                    vis_fea_knn_img.append(ins_img_path_fea[38:])
                    ins_img_data_fea = np.load(ins_img_path_fea)
                    vis_fea_knn.append(ins_img_data_fea.tolist())

            self.neg_dict[str(label)] = vis_fea_knn_img

            vis_fea_knn_np = (np.array(vis_fea_knn)).reshape(len(vis_fea_knn), -1)

            nbrs = NearestNeighbors(n_neighbors = 10, algorithm = 'ball_tree') \
                .fit(vis_fea_knn_np)
            indices = nbrs.kneighbors(vis_fea_knn_np, return_distance = False)

            for i in range(len(label_data)):
                ins_data = label_data[i]
                ins_data_img_path = ins_data['image_path']
                ins_data_label = ins_data['labels']
                for img_no in range(len(ins_data_img_path)):
                    ins_data_img_path_per = ins_data_img_path[img_no]
                    ins_img_path_fea_save = os.path.join(self.save_path_k, ins_data_img_path_per) \
                        .replace('.png', '_'+str(label)+'_knn.npy')

                    np.save(ins_img_path_fea_save, indices[fea_knn_no, 1:])
                    fea_knn_no += 1

        end_time = time.time()

        print("time total knn {}".format(end_time - start_time))

    #def syn_label(self, ins_k, label):
    def syn_label(self, ins_k, label):
        cls_num = 14
        k = 9
        syn_label = np.zeros((cls_num))
        for i in range(len(ins_k)):
            ins_index = ((self.neg_dict[str(label)])[ins_k[i]])[:-6]
            ins_data_label = np.array(self.label.loc[ins_index])[0:14]
            syn_label += ins_data_label
        for i in range(cls_num):
            if syn_label[i] >= 4:
                syn_label[i] = 1
            else:
                syn_label[i] = 0
        return syn_label

    def mls(self, inputs, labels, ids):
        tail_ins_list = []
        new_feature = []
        new_label = []
        batch_size = inputs.shape[0]
        for label in self.tail_list:
            label_index = self.getlabelindex(label, labels)
            for i in label_index:
                ins_id_path = ids[i]
                ins_path_f = os.path.join(self.save_path_f, ins_id_path).replace('png', 'npy')
                ins_f = np.load(ins_path_f)
                ins_path_k = os.path.join(self.save_path_k, ins_id_path).replace('.png',
                                                                                 '_'+str(label)+'_knn.npy')
                ins_k = np.load(ins_path_k)
                label_syn = self.syn_label(ins_k, label)
                for times in range(self.N):
                    nei_ref = random.randint(0, len(ins_k) - 1)
                    delta = random.uniform(0, 1)
                    nei_img_path = self.neg_dict[str(label)][nei_ref]
                    nei_path_f = os.path.join(self.save_path_f, nei_img_path).replace('png', 'npy')
                    nei_f = np.load(nei_path_f)
                    gap = nei_f - ins_f
                    new_ins_feature = ins_f + delta * gap
                    new_feature.append(new_ins_feature)
                    new_label.append(label_syn)
        new_feature = torch.tensor(new_feature).cuda()
        new_label = torch.tensor(new_label).cuda()

        inputs = torch.cat((inputs, new_feature), dim = 0)
        labels = torch.cat((labels, new_label), dim = 0)

        return inputs, labels

    def forward(self, x = None, epoch = None, labels = None, ids = None):

        x = self.vis_embed(self.feature_extractor(x).reshape(-1, 7, 7, 2048))
        # if self.is_train:
        #     x, labels = self.mls(x, labels, ids)

        x = self.avgpool(x.reshape(-1, self.hidden_size, 7, 7))
        x = torch.flatten(x, 1)
        x = x.to(torch.float32)
        x = self.cls(x)

        if self.is_train:
            labels = labels.to(torch.float32)
            return x, labels
        else:
            return x

