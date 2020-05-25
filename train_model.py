# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 14:49:12 2020

@author: na5815
"""
import os
import numpy as np
import h5py
import pandas as pd
from scipy.io import wavfile
from preprocessing import pad_input, extract_mel, extract_mfcc, normalise_feature
from model import CNN_TF, LSTM_TF
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
from utils import change_label, make_dir
from plotting import make_confusion_matrix, plot_training
import matplotlib.pyplot as plt

feature_dir = 'features'
result_dir = 'results'
make_dir(result_dir)
feature = 'mel' #['mel','mfcc']

# Loading feature from training, validation, and testing sets
train_file = os.path.join(feature_dir,'mel_mfcc_train.h5')
print ("Loading feature from file: "+train_file)
with h5py.File(train_file,'r') as f:
    x_train = f[feature+'_train'][()]
    y_train = f['y_train'][()]
    
valid_file = os.path.join(feature_dir,'mel_mfcc_valid.h5')
print ("Loading feature from file: "+valid_file)
with h5py.File(valid_file,'r') as f:
    x_valid = f[feature+'_valid'][()]
    y_valid = f['y_valid'][()]
    
test_file = os.path.join(feature_dir,'mel_mfcc_test.h5')
print ("Loading feature from file: "+test_file)
with h5py.File(test_file,'r') as f:
    x_test = f[feature+'_test'][()]
    y_test = f['y_test'][()]

test_csv = pd.read_csv("testing.csv",index_col=0)
y = test_csv.iloc[:,1].to_list()
change = change_label()
y_test_int = change.str2int(y)
labels = change.le.classes_.tolist()
    
# Building the model
model_type = 'lstm' # ['cnn','lstm']
if model_type=='cnn':
    x_train = np.expand_dims(x_train,axis=3)
    x_valid = np.expand_dims(x_valid,axis=3)
    x_test = np.expand_dims(x_test,axis=3)
    args = {'filters':16,'kernel_size':2,'pool_size':2,'dropout':0.2,'activation':'relu',
            'input_shape':(x_train.shape[1], x_train.shape[2], x_train.shape[3]),'num_labels':y_train.shape[1],
            'loss':'categorical_crossentropy','optimizer':'adam','metrics':'accuracy'}
    model_tf = CNN_TF(**args)
elif model_type=='lstm':
    args = {'units':100,'dropout':0.2,'input_shape':(x_train.shape[1], x_train.shape[2]),
            'num_labels':y_train.shape[1],'loss':'categorical_crossentropy','optimizer':'adam','metrics':'accuracy'}
    model_tf = LSTM_TF(**args)

model_tf.build()
earlystop =  EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
model_file = os.path.join(result_dir,feature+'_best_'+model_type+'_model.h5')
checkpoint = ModelCheckpoint(filepath=model_file, monitor='val_loss', mode='min',
                               verbose=1, save_best_only=True)
callbacks = [earlystop,checkpoint]
batch_size = 64
epochs = 50
start_time = datetime.now()
log = model_tf.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,callbacks=callbacks,
              validation_data=(x_valid,y_valid))
train_time = datetime.now() - start_time
print("Training completed in time: ", train_time)
plot_training(log)

eval_score = model_tf.evaluate(x_valid,y_valid,batch_size=64)
y_pred_prob = model_tf.predict(x_test,batch_size=64)
y_pred_int = change.bin2int(y_pred_prob)
y_pred_str = change.bin2str(y_pred_prob)

test_report = classification_report(y_test_int, y_pred_int, output_dict=True)
df_test_report = pd.DataFrame(test_report).transpose()
print(df_test_report)
report_file = os.path.join(result_dir,feature+'_'+model_type+'_test.csv')
print ("Storing report into a file: "+report_file)
df_test_report.to_csv(report_file)

conf_mat = confusion_matrix(y_test_int,y_pred_int)
make_confusion_matrix(conf_mat, figsize=(8,8),cbar=False,title="Confusion matrix on testing set")

filename = test_csv.iloc[0,0]
def predict_audio(filename,feature='mfcc',model_type='lstm'):
    sr, audio = wavfile.read(filename)
    audio = pad_input(audio)
    if feature=='mfcc':
        x_feature = normalise_feature(extract_mfcc(audio))
    elif feature=='mel':
        x_feature = normalise_feature(extract_mel(audio))
    if model_type=='lstm':
        x_feature = np.array(x_feature.T)[np.newaxis,:,:]
    elif model_type=='cnn':
        x_feature = np.array(x_feature.T)[np.newaxis,:,:,np.newaxis]
    y_prob = model_tf.predict(x_feature,batch_size=1)
    y_str = change.bin2str(y_prob)
    
    print("Predicted label: {}".format(y_str[0])) 
    for i in range(y_prob.shape[1]): 
        print("Probability of {} \t: {:.6f}".format(labels[i],y_prob[0,i]))
        
predict_audio(test_csv.iloc[2,0],feature=feature,model_type=model_type)    
    
    