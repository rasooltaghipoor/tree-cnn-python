# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 20:40:00 2020

@author: rasool
"""

import numpy as np
import pandas as pd
import cv2

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tree_cnn import TreeCNN

import keras
from keras.datasets import cifar100

# cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# range index from 0 to 9

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

y_train = y_train.flatten()
y_test = y_test.flatten()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

rnds = np.random.permutation(100)

num_initial_classes = 10
addition_factor = 10
# init_classes = [i for i in range(num_initial_classes)]
init_classes = [rnds[i] for i in range(num_initial_classes)]
tree = TreeCNN(init_classes, ds_name='cifar100')

# old_classes_X = x_train[y_train<num_initial_classes].astype(float)
# old_classes_Y = y_train[y_train<num_initial_classes]
old_classes_X = x_train[np.in1d(y_train, rnds[0:num_initial_classes])].astype(float)
old_classes_Y = y_train[np.in1d(y_train, rnds[0:num_initial_classes])]

# old_classes_Xtest = x_test[y_test<num_initial_classes].astype(float)
# old_classes_Ytest = y_test[y_test<num_initial_classes]
old_classes_Xtest = x_test[np.in1d(y_test, rnds[0:num_initial_classes])].astype(float)
old_classes_Ytest = y_test[np.in1d(y_test, rnds[0:num_initial_classes])]

tree.train(old_classes_X, old_classes_Y, old_classes_Xtest, old_classes_Ytest)

Y_hat = tree.inference(old_classes_Xtest)
Y_true = old_classes_Ytest
print(Y_hat)
print(np.sum(Y_true == Y_hat) / len(Y_true))

for cn in range(9):
    # new_classes_X = [x_train[y_train == i].astype(float) for i in range(num_initial_classes, num_initial_classes + 10)]
    # new_classes_Y = [i for i in range(num_initial_classes, num_initial_classes + 10)]
    new_classes_X = [x_train[y_train == rnds[i]].astype(float) for i in range(num_initial_classes, num_initial_classes + addition_factor)]
    new_classes_Y = [rnds[i] for i in range(num_initial_classes, num_initial_classes + addition_factor)]

    # merged_classes_X = x_train[y_train < num_initial_classes + 10].astype(float)
    # merged_classes_Y = y_train[y_train < num_initial_classes + 10]
    merged_classes_X = x_train[np.in1d(y_train, rnds[0:num_initial_classes + addition_factor])].astype(float)
    merged_classes_Y = y_train[np.in1d(y_train, rnds[0:num_initial_classes + addition_factor])]

    # merged_classes_Xtest = x_test[y_test < num_initial_classes + 10].astype(float)
    # merged_classes_Ytest = y_test[y_test < num_initial_classes + 10]
    merged_classes_Xtest = x_test[np.in1d(y_test, rnds[0:num_initial_classes + addition_factor])].astype(float)
    merged_classes_Ytest = y_test[np.in1d(y_test, rnds[0:num_initial_classes + addition_factor])]

    # tree.addTasks(new_classes_X, [i for i in range(num_initial_classes, num_initial_classes + 10)])
    tree.addTasks(new_classes_X, new_classes_Y)

    # Retraing with all classes
    tree.train(merged_classes_X, merged_classes_Y, merged_classes_Xtest, merged_classes_Ytest)

    num_initial_classes += addition_factor

    Y_hat = tree.inference(merged_classes_Xtest)
    Y_true = merged_classes_Ytest  # y_train[y_train < num_initial_classes]
    print(Y_hat)
    print(np.sum(Y_true == Y_hat) / len(Y_true))

# ### Re-evaluate in old classes

# Y_hat = tree.inference(old_classes_X)
# Y_true = y_train[y_train<num_initial_classes]
# print(Y_hat)
# np.sum(Y_true==Y_hat)/len(Y_true)
#
# # ### Re-evaluate in all classes
#
# Y_hat = tree.inference(merged_classes_X)
# Y_true = y_train[y_train<7]
# print(Y_hat)
# np.sum(Y_true==Y_hat)/len(Y_true)
