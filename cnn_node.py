# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 20:23:47 2020

@author: rasool
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Input, Flatten, MaxPooling2D
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import cv2

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from cnn_base import CNN
import keras

class CnnNode (object):
    def __init__(self, num_classes, labels = [], input_shape=(32,32,3), max_leafes=5, node_type='root', ds_name='cifar10'):
        self.net = CNN(num_classes, input_shape, node_type, ds_name)
        self.num_classes = num_classes
        self.childrens = [label for label in labels]
        self.childrens_leaf = [True for _ in range(num_classes)]
        self.labels = labels
        self.max_leafes = max_leafes
        self.labels_transform = {}
        for nc in range(num_classes):
            self.labels_transform[labels[nc]] = []
            self.labels_transform[labels[nc]].append(labels[nc])
        
    
    def get_num_leafnodes(self):
        count = 0
        for is_leaf in self.childrens_leaf:
            if is_leaf:
                count = count + 1
        return count
    
    def remove_leaf(self, label):
        childrens = []
        childrens_leaf = []
        labels = []
        del self.labels_transform[label]
        self.num_classes = (self.num_classes - 1)
        position_in_net = -1
        
        for i in range(len(self.labels)):
            if self.labels[i] != label:
                childrens.append(self.childrens[i])
                childrens_leaf.append(self.childrens_leaf[i])
                labels.append(self.labels[i])
            else:
                position_in_net = i
                
        self.childrens = childrens
        self.childrens_leaf = childrens_leaf
        self.labels = labels
        self.net.remove_class(position_in_net)
        
    def add_leaf(self, label):
        self.childrens.append(label)
        self.childrens_leaf.append(True)
        self.num_classes = (self.num_classes + 1)
        self.labels.append(label)
        self.labels_transform[label] = []
        self.labels_transform[label].append(label)
        self.net.add_class()
    
    def predict(self, imgs):
        vector_output = self.net.pred(imgs)
        return vector_output
    
    def inference(self, imgs):
        vector_output = self.predict(imgs)
        out = np.array([idx for idx in np.argmax(vector_output, axis=1)])
        output = np.array([-1 for _ in range(imgs.shape[0])])

        for i, o in zip(range(len(out)), out):
            if self.childrens_leaf[o]:
                output[i] = self.labels[o]            

        # Send images to branches nodes
        for child_id in range(len(self.childrens_leaf)):
            if not self.childrens_leaf[child_id]:
                output[out==child_id] = self.childrens[child_id].inference(imgs[out==child_id])


        return output
    
    def train(self, X, Y, X_test, Y_test):
        Y_to_this_nivel = Y.copy()
        Y_test_to_this_nivel = Y_test.copy()
        print(1, self.labels_transform)
        for idx in range(len(self.labels)):
            for label in self.labels_transform[self.labels[idx]]:
                Y_to_this_nivel[Y==label] = idx
                Y_test_to_this_nivel[Y_test == label] = idx;

        num_clss = len(np.unique(Y_to_this_nivel));
        # Y_true = pd.get_dummies(pd.Series(Y_to_this_nivel)).values
        Y_true = keras.utils.to_categorical(Y_to_this_nivel, num_clss)
        # Y_test_true = pd.get_dummies(pd.Series(Y_test_to_this_nivel)).values
        Y_test_true = keras.utils.to_categorical(Y_test_to_this_nivel, num_clss)

        self.net.train(X, Y_true, X_test, Y_test_true)
        
        # Send images to branches nodes
        for child_id in range(len(self.childrens_leaf)):
            if not self.childrens_leaf[child_id]:
                self.childrens[child_id].train(X[Y_to_this_nivel==child_id], Y[Y_to_this_nivel==child_id],
                                               X_test[Y_test_to_this_nivel==child_id], Y_test[Y_test_to_this_nivel==child_id])

