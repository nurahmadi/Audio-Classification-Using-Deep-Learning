"""
Created on Sun Mar 15 15:18:18 2020

@author: na5815
"""
import os
import glob
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
#from tqdm import tqdm
import librosa
import librosa.display
import numpy as np
import pandas as pd
from preprocessing import pad_input, extract_stft, extract_mel, extract_mfcc, split_data_index
from utils import flatten, make_dir

dir_name = "raw_data"
make_dir(dir_name)

#os.listdir(dir_name) # returns list of files/directorys within raw_data
# save directory name corresponding to label name 
label_names = [label for label in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name,label))]
# sort label name
label_names.sort()
# count number of files within each label directory
label_length = []
label_plots = []
audio_length = []
audio_chan = []
num_not1s = 0
for label in label_names:
    file_names = os.path.join(dir_name,label,"*.wav")
    print(file_names)
    #file_list = [filename for filename in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, filename))]
    file_lists = glob.glob(file_names)
    label_length.append(len(file_lists))
    label_plots.append(file_lists[0])
    # count the number of files with length not equal to 1 s
    #for f in file_list:
    for i in range(1):
        sr, audio = wavfile.read(file_lists[i])
        audio_length.append(len(audio))
        audio_chan.append(audio.ndim)

num_files = sum(label_length)        
audio_length = np.asarray(audio_length)
min_length = np.min(audio_length) #6688
max_length = np.max(audio_length) #16000
num_more1s = np.where(audio_length > sr)[0]
num_less1s = np.where(audio_length < sr)[0]

label_prob = [length/sum(label_length) for length in label_length]
# plot class distribution
fig, ax = plt.subplots()
ax.pie(label_length, labels=label_names, autopct='%1.1f%%',
        shadow=False, startangle=90, counterclock=False)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax.set_title('Class distribution')

audio_plots = []
for i,file in enumerate(label_plots):
    sr, audio = wavfile.read(file)
    audio_plots.append(pad_input(audio))
    print("sampling: {}, audio length: {}, paded audio length: {}".format(sr,len(audio),len(audio_plots[i])))

title_plots = ['raw','stft','mel','mfcc']
nrow = len(audio_plots)
ncol = len(title_plots)
fig = plt.figure(figsize=(10,16))
for r in range(5):
    plt.subplot(nrow,ncol,ncol*r+1)
    librosa.display.waveplot(audio_plots[r], sr=sr)
    plt.title("{}: {}".format(title_plots[0],label_names[r]))
    plt.xlim(0,1)
    plt.locator_params(axis='x', nbins=3)
    # stft spectrogram
    stft = extract_stft(audio_plots[r])
    plt.subplot(nrow,ncol,ncol*r+2)
    librosa.display.specshow(stft,x_axis='time',y_axis='log',sr=sr)
    plt.title("{}: {}".format(title_plots[1],label_names[r]))
    plt.locator_params(axis='x', nbins=3)
    plt.colorbar(format='%+2.0f dB')
    # mel spectrogram
    mel = extract_mel(audio_plots[r])
    plt.subplot(nrow,ncol,ncol*r+3)
    librosa.display.specshow(mel,x_axis='time',y_axis='mel',sr=sr)
    plt.title("{}: {}".format(title_plots[2],label_names[r]))
    plt.locator_params(axis='x', nbins=3)
    plt.colorbar(format='%+02.0f dB')
    # mfcc
    plt.subplot(nrow,ncol,ncol*r+4)
    mfcc = extract_mfcc(audio_plots[r])
    librosa.display.specshow(mfcc,x_axis='time',sr=sr)
    plt.title("{}: {}".format(title_plots[3],label_names[r]))
    plt.xlim(0,1)
    plt.locator_params(axis='x', nbins=3)
    plt.colorbar()
#plt.subplots_adjust(hspace=0.1,wspace=0.4)
plt.tight_layout()

# split data into training, validation, and testing tests
train_files = []
valid_files = []
test_files = []

train_labels = []
valid_labels = []
test_labels = []
for label in label_names:
    file_names = os.path.join(dir_name,label,"*.wav")
    file_lists = np.asarray(glob.glob(file_names))
    file_idxs = np.arange(len(file_lists))
    train_idx,valid_idx,test_idx = split_data_index(file_idxs,train_size=0.8,valid_size=0.1)
    # append training, validation, and testing lists of files
    train_files.append(file_lists[train_idx].tolist())
    valid_files.append(file_lists[valid_idx].tolist())
    test_files.append(file_lists[test_idx].tolist())
    # append training, validation, and testing lists of files
    train_labels.append([label]*len(train_idx))
    valid_labels.append([label]*len(valid_idx))
    test_labels.append([label]*len(test_idx))

train_files = flatten(train_files)
valid_files = flatten(valid_files)
test_files = flatten(test_files)
train_labels = flatten(train_labels)
valid_labels = flatten(valid_labels)
test_labels = flatten(test_labels)

df_train = pd.DataFrame(list(zip(train_files,train_labels)),columns =['Training Files', 'Training Labels'])        
df_valid = pd.DataFrame(list(zip(valid_files,valid_labels)),columns =['Validation Files', 'Validation Labels'])   
df_test = pd.DataFrame(list(zip(test_files,test_labels)),columns =['Testing Files', 'Testing Labels'])   

# store training, validation, and testing sets into files
df_train.to_csv("training.csv")
df_valid.to_csv("validation.csv")
df_test.to_csv("testing.csv")

