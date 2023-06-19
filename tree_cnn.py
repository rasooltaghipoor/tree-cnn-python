# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 20:23:47 2020

@author: rasool
"""

import numpy as np

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from cnn_node import CnnNode

class NodeList (object):
    def __init__(self, label, values, nodes, class_position):
        self.label = label
        self.values = values
        self.nodes = nodes
        self.class_position = class_position
    def __str__(self):
        return {'label': self.label, 'values': self.values, 'nodes':self.nodes, 'class_position': self.class_position}
    def __unicode__(self):
        return {'label': self.label, 'values': self.values, 'nodes':self.nodes, 'class_position': self.class_position}
    def __repr__(self):
        return 'label: ' + str(self.label) + ' values: ' + str(self.values) + ' nodes: ' + str(self.nodes)+' class_pos: '+ str(self.class_position)


class TreeCNN (object):
    def __init__(self, initial_labels, alpha=0.1, beta=0.1, input_shape=(32,32,3), max_leafnodes=1000, ds_name='cifar10'):
        num_class_initial = len(initial_labels)
        self.root = CnnNode(num_class_initial, labels=initial_labels, input_shape=input_shape, node_type='root', ds_name=ds_name)
        self.alpha = alpha
        self.beta = beta
        self.max_leafnodes = max_leafnodes
        self.ds_name = ds_name
    
    def addTasks(self, imgs_of_classes=[], labels=[]):
        self.growTreeCNN(self.root, imgs_of_classes, labels)
        
    def train(self, X, Y, X_test, Y_test):
        self.root.train(X, Y, X_test, Y_test)
        
    def inference(self, X):
        return self.root.inference(X)
        
        
    def growTreeCNN(self, operation_node, imgs_of_classes=[], labels=[]):
        def get_Oavg_matrix(node, imgs_of_classes_, labels_):
            Oavg = np.zeros(shape=(node.num_classes, 0))
            for imgs, label in zip(imgs_of_classes_, labels_):
                net_out = node.predict(imgs)
                Oavg_i = np.average(net_out, axis=0)
                Oavg = np.concatenate(( Oavg, Oavg_i.reshape((Oavg_i.shape[0], 1)) ), axis=1)
            return Oavg
        
        def get_loglikelihood_matrix(Oavg):
            return (np.power(np.e, Oavg) / np.sum(np.power(np.e, Oavg), axis=0))
        
        def generate_listS(llh, labels_in):
            listS = []
            for i in range(llh.shape[1]):
                label = labels_in[i]
                values = []
                nodes = []

                col = llh[:,i].copy()
                for _ in range(3):
                    max_idx = np.argmax(col)
                    values.append(col[max_idx])
                    nodes.append(max_idx)
                    col[max_idx] = -100

                listS.append(NodeList(label, values, nodes, i))

            # Sort List S by value of S[i].values[0]
            listS.sort(key=lambda node_list: node_list.values[0])
            
            return listS

        llh = get_loglikelihood_matrix(get_Oavg_matrix(operation_node, imgs_of_classes, labels))
        
        listS = generate_listS(llh, labels)
        new_labels = labels.copy()
        
        branches_dest = {}
        while len(listS) > 0:
            nodeList = listS[0]
            rows_to_remove_in_llh = []
            # adding new class to node0. if node0 is a leaf, merge node0 and new class and form a new branch node
            if nodeList.values[0] - nodeList.values[1] > self.alpha:
                # in this case a new branch node created and 2 nodes will be it's children
                if operation_node.childrens_leaf[nodeList.nodes[0]]:
                    print('\n***** a new branch is created dadash *****\n')
                    operation_node.childrens_leaf[nodeList.nodes[0]] = False
                    old_label = operation_node.labels[nodeList.nodes[0]]
                    new_label = nodeList.label
                    branch_node = CnnNode(2, labels=[old_label, new_label], node_type='branch', ds_name=self.ds_name)
                    operation_node.childrens[nodeList.nodes[0]] = branch_node
                else:
                    print('\n***** an existing branch node updated *****\n')
                    if nodeList.nodes[0] not in branches_dest:
                        branches_dest[nodeList.nodes[0]] = []
                    branches_dest[nodeList.nodes[0]].append(nodeList.label)
                    # ****** me ******
                    operation_node.childrens[nodeList.nodes[0]].add_leaf(nodeList.label)

                operation_node.labels_transform[operation_node.labels[nodeList.nodes[0]]].append(nodeList.label)

            # node0 and node1 are close to new class, so merge them if possible and add new class to them
            elif nodeList.values[1] - nodeList.values[2] > self.beta:
                print('\n***** new class is close to node0 and node1, maybe they are merged ******\n')
                left_is_leafnode = operation_node.childrens_leaf[nodeList.nodes[0]]
                right_is_leafnode = operation_node.childrens_leaf[nodeList.nodes[1]]
                has_space_in_left = left_is_leafnode or (operation_node.childrens[nodeList.nodes[0]].get_num_leafnodes() < (self.max_leafnodes - 1))
                
                if right_is_leafnode and has_space_in_left: # if Merge
                    if operation_node.childrens_leaf[nodeList.nodes[0]]: # if left is a leaf
                        # merge node0 and node1 and create a branch node
                        operation_node.childrens_leaf[nodeList.nodes[0]] = False
                        old_label = operation_node.labels[nodeList.nodes[0]]
                        new_label = operation_node.labels[nodeList.nodes[1]]
                        branch_node = CnnNode(2, labels=[old_label, new_label], node_type='branch', ds_name=self.ds_name)
                        operation_node.childrens[nodeList.nodes[0]] = branch_node
                    else:
                        # add node1 to node0 that is a branch node
                        operation_node.childrens[nodeList.nodes[0]].add_leaf(operation_node.labels[nodeList.nodes[1]])
                        if nodeList.nodes[0] not in branches_dest:
                            branches_dest[nodeList.nodes[0]] = []
                        branches_dest[nodeList.nodes[0]].append(nodeList.label)

                    operation_node.labels_transform[operation_node.labels[nodeList.nodes[0]]].append(operation_node.labels[nodeList.nodes[1]])
                    # add new class to node0 branch ****** me ******
                    operation_node.childrens[nodeList.nodes[0]].add_leaf(nodeList.label)
                    operation_node.labels_transform[operation_node.labels[nodeList.nodes[0]]].append(nodeList.label)

                    operation_node.remove_leaf(operation_node.labels[nodeList.nodes[1]])
                    rows_to_remove_in_llh.append(nodeList.nodes[1])
                else:
                    if left_is_leafnode:
                        # add new class to left node (node0 that is a leaf) and create a new branch
                        operation_node.childrens_leaf[nodeList.nodes[0]] = False
                        old_label = operation_node.labels[nodeList.nodes[0]]
                        new_label = nodeList.label
                        branch_node = CnnNode(2, labels=[old_label, new_label], node_type='branch', ds_name=self.ds_name)
                        operation_node.childrens[nodeList.nodes[0]] = branch_node
                        # ***** me *****
                        operation_node.labels_transform[operation_node.labels[nodeList.nodes[0]]].append(nodeList.label)
                    elif right_is_leafnode:
                        # add new class to right node (node1 that is a leaf) and create a new branch
                        operation_node.childrens_leaf[nodeList.nodes[1]] = False
                        old_label = operation_node.labels[nodeList.nodes[1]]
                        new_label = nodeList.label
                        branch_node = CnnNode(2, labels=[old_label, new_label], node_type='branch', ds_name=self.ds_name)
                        operation_node.childrens[nodeList.nodes[1]] = branch_node
                        # ***** me *****
                        operation_node.labels_transform[operation_node.labels[nodeList.nodes[1]]].append(nodeList.label)
                    else:
                        if operation_node.childrens[nodeList.nodes[0]].get_num_leafnodes() < operation_node.childrens[nodeList.nodes[1]].get_num_leafnodes():
                            # add new class to left node (node0) that is a branch
                            if nodeList.nodes[0] not in branches_dest:
                                branches_dest[nodeList.nodes[0]] = []
                            branches_dest[nodeList.nodes[0]].append(nodeList.label)
                            # ***** me *****
                            operation_node.childrens[nodeList.nodes[0]].add_leaf(nodeList.label)
                            operation_node.labels_transform[operation_node.labels[nodeList.nodes[0]]].append(nodeList.label)
                        else:
                            # add new class to right node (node1) that is a branch
                            if nodeList.nodes[1] not in branches_dest:
                                branches_dest[nodeList.nodes[1]] = []
                                print(nodeList)
                            branches_dest[nodeList.nodes[1]].append(nodeList.label)
                            # ***** me *****
                            operation_node.childrens[nodeList.nodes[1]].add_leaf(nodeList.label)
                            operation_node.labels_transform[operation_node.labels[nodeList.nodes[1]]].append(nodeList.label)
            else:
                print('\n***** a new leaf node is created *****\n')
                operation_node.add_leaf(nodeList.label)
                
            
            # Clean likelihood matrix and labels to recreate listS to next iteration
            # Delete column already inserted
            llh = np.delete(llh, nodeList.class_position, axis=1)
            # Delete rows that was merged
            for r in rows_to_remove_in_llh:
                llh = np.delete(llh, r, axis=0)
            # Delete rows that represent full nodes (just branch/child nodes)
            deleted = 0
            for i in range(len(operation_node.childrens_leaf)):
                if not operation_node.childrens_leaf[i] and (operation_node.childrens[i].get_num_leafnodes() >= self.max_leafnodes):
                    llh = np.delete(llh, (i - deleted), axis=0)
                    deleted = deleted + 1
            # Update labels
            del new_labels[nodeList.class_position]
            listS = generate_listS(llh, new_labels)
        
        # # Send to sub-nivels
        # for k, v in branches_dest.items():
        #     imgs_to_send = []
        #     labels_to_send = []
        #     for idx, label in zip(range(len(labels)), labels):
        #         if label in v:
        #             imgs_to_send.append(imgs_of_classes[idx])
        #             labels_to_send.append(labels[idx])
        #
        #     self.growTreeCNN(operation_node.childrens[k], imgs_to_send, labels_to_send)
        