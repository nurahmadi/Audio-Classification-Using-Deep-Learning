#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 20:22:05 2020

@author: na5815
"""
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def make_dir(dir_name):
    """
    create new directory if it does not exist
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def flatten(lists):
    """
    flatten the list of lists
    """
    flat_list = [val for sublist in lists for val in sublist]
    return flat_list

class change_label:
    """
    change label from one to another representation
    """
    def __init__(self):
        self.le = LabelEncoder()
    
    def str2int(self,y_str):
        # change label from string to integer
        y_int = self.le.fit_transform(y_str)
        return y_int
    
    def int2str(self,y_int):
        # change label from string to integer
        y_str = self.le.inverse_transform(y_int).tolist()
        return y_str
    
    def int2bin(self,y_int):
        # change label from integer to binary matrix
        y_bin = to_categorical(y_int)
        return y_bin
    
    def bin2int(self,y_bin):
        # change label from binary matrix to integer
        y_int = np.argmax(y_bin,axis=1)
        return y_int
    
    def str2bin(self,y_str):
        # change label from list of string to binary matrix
        y_bin = to_categorical(self.le.fit_transform(y_str))
        return y_bin
    
    def bin2str(self,y_bin):
        # change label from binary matrix to string
        y_str = self.le.inverse_transform(np.argmax(y_bin,axis=1)).tolist()
        return y_str