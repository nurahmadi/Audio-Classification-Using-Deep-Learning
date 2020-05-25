# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 08:09:51 2020

@author: na5815
"""
import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from preprocessing import pad_input, extract_mel, extract_mfcc, normalise_feature
from utils import change_label, make_dir
import pickle
import h5py

# open training, validation, and testing sets from csv files
train_csv = pd.read_csv("training.csv",index_col=0)
valid_csv = pd.read_csv("validation.csv",index_col=0)
test_csv = pd.read_csv("testing.csv",index_col=0)

"""
set_dirnames = ['train','valid','test']
for dname in set_dirnames: 
    make_dir(dname)
"""

# extracting the mel and mfcc features
feature_dir = 'features'
mel_train = []
mel_valid = []
mel_test = []

mfcc_train = []
mfcc_valid = []
mfcc_test = []

change = change_label()

for i in range(train_csv.shape[0]):
    sr, audio = wavfile.read(train_csv.iloc[i,0])
    audio = pad_input(audio)
    mel = normalise_feature(extract_mel(audio))
    mfcc = normalise_feature(extract_mfcc(audio))
    mel_train.append(mel.T)
    mfcc_train.append(mfcc.T)
mel_train = np.asarray(mel_train)
mfcc_train = np.asarray(mfcc_train)
y = train_csv.iloc[:,1].to_list()
y_train = change.str2bin(y)
#train_file = os.path.join(feature_dir,'mel_mfcc_train.pkl')
train_file = os.path.join(feature_dir,'mel_mfcc_train.h5')
print ("Storing features into a file: "+train_file)
#with open(train_file, 'wb') as f:
#    pickle.dump([mel_train, mfcc_train, y_train], f)
with h5py.File(train_file,'w') as f:
    f['mel_train'] = mel_train
    f['mfcc_train'] = mfcc_train
    f['y_train'] = y_train


for i in range(valid_csv.shape[0]):
    sr, audio = wavfile.read(valid_csv.iloc[i,0])
    audio = pad_input(audio)
    mel = normalise_feature(extract_mel(audio))
    mfcc = normalise_feature(extract_mfcc(audio))
    mel_valid.append(mel.T)
    mfcc_valid.append(mfcc.T)
mel_valid = np.asarray(mel_valid)
mfcc_valid = np.asarray(mfcc_valid)
y = valid_csv.iloc[:,1].to_list()
y_valid = change.str2bin(y)
#valid_file = os.path.join(feature_dir,'mel_mfcc_valid.pkl')
valid_file = os.path.join(feature_dir,'mel_mfcc_valid.h5')
print ("Storing features into a file: "+valid_file)
#with open(valid_file, 'wb') as f:
#    pickle.dump([mel_valid, mfcc_valid, y_valid], f)
with h5py.File(valid_file,'w') as f:
    f['mel_valid'] = mel_valid
    f['mfcc_valid'] = mfcc_valid
    f['y_valid'] = y_valid

for i in range(test_csv.shape[0]):
    sr, audio = wavfile.read(test_csv.iloc[i,0])
    audio = pad_input(audio)
    mel = normalise_feature(extract_mel(audio))
    mfcc = normalise_feature(extract_mfcc(audio))
    mel_test.append(mel.T)
    mfcc_test.append(mfcc.T)
mel_test = np.asarray(mel_test)
mfcc_test = np.asarray(mfcc_test)
y = test_csv.iloc[:,1].to_list()
y_test = change.str2bin(y)
#test_file = os.path.join(feature_dir,'mel_mfcc_test.pkl')
test_file = os.path.join(feature_dir,'mel_mfcc_test.h5')
print ("Storing features into a file: "+test_file)
#with open(test_file, 'wb') as f:
#    pickle.dump([mel_test, mfcc_test, y_test], f)
with h5py.File(test_file,'w') as f:
    f['mel_test'] = mel_test
    f['mfcc_test'] = mfcc_test
    f['y_test'] = y_test
