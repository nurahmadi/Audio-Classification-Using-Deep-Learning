"""
List of preprocessing functions
"""
import librosa
import numpy as np
from scipy.stats import zscore

def pad_input(audio,L=16000):
    """
    Pad audio time series with zeros
    
    Parameters
    ----------
    audio : ndarray
        audio time series
    L : int
        length of padded audio time series

    Returns
    -------
    audio_pad : ndarray
        padded audio time series
    """
    if len(audio) < L:
        audio_pad = np.pad(audio, (0, L-len(audio)), 'constant')
        audio_pad = audio_pad.astype(np.float32)
    else: 
        audio_pad = audio.astype(np.float32)
    
    return audio_pad

def extract_stft(audio,n_fft=2048,hop_length=None,win_length=None):
    """
    Extract log STFT spectrogram features
    
    Parameters
    ----------
    audio : ndarray
        audio time series.
    n_fft : int
        length of FFT. Detaults to 2048.
    hop_length : int
        step size (in samples) between consecutive windowed processing. Detaults to win_length/4.
    win_length : int
        window length used for FFT computation. Detaults to n_fft.

    Returns
    -------
    stft : ndarray
        Log magnitude of short-term Fourier transform coefficients.
    """
    stft = np.abs(librosa.stft(audio,n_fft=n_fft,hop_length=hop_length,win_length=win_length))
    stft = librosa.amplitude_to_db(stft,ref=np.max)
    return stft

def extract_mel(audio,sr=16000,n_mels=40,**kwargs):
    """
    Extract Mel spectrogram features
    
    Parameters
    ----------
    audio : ndarray
        audio time series.
    sr : number > 0
        sampling rate of audio time series.
    n_mels : 
        number of Mel bands. Default to 128.
    kwargs : additional keyword arguments
        arguments to extract_fft

    Returns
    -------
    mel : ndarray
        Mel spectrogram.
    """
    mel = librosa.feature.melspectrogram(audio,sr=sr,n_mels=n_mels,**kwargs)
    #mel = librosa.power_to_db(mel,ref=np.max)
    return mel

def extract_mfcc(audio,sr=16000,n_mfcc=40,**kwargs):
    """
    Extract Mel-frequency cepstral coefficients (MFCC) features
    
    Parameters
    ----------
    audio : ndarray
        audio time series.
    sr : number > 0
        sampling rate of audio time series.
    n_mfcc : 
        number of MFCCs. Default to 40.
    kwargs : additional keyword arguments
        arguments to extract_fft

    Returns
    -------
    mfcc : ndarray
        MFCC sequence.
    """
    mfcc = librosa.feature.mfcc(audio,sr=sr,n_mfcc=n_mfcc,**kwargs)
    return mfcc

def normalise_feature(feature,mode='z_score'):
    """
    Extract Mel-frequency cepstral coefficients (MFCC) features
    
    Parameters
    ----------
    feature : ndarray
        feature time series.
    mode : string ['z_score','min_max','mean']
        normalisation mode.

    Returns
    -------
    norm_feature : ndarray
        normalised feature
    """
    if mode=='z_score':
        norm_feature = zscore(feature,axis=1)
    elif mode=='min_max':
        norm_feature = (feature - np.min(feature,axis=1))/(np.max(feature,axis=1)-np.min(feature,axis=1))
    elif mode=='mean':
        mean_feature = np.mean(feature,axis=1)
        norm_feature = feature-mean_feature
    else:
        print("mode is not supported. return feature with z_score mode")
        norm_feature = zscore(feature,axis=1)
        
    return norm_feature
    
def split_data_index(index,train_size=0.8,valid_size=0.1):
    """
    Split data by its index into training, validation, and testing sets
    
    Parameters
    ----------
    index : ndarray
        index of data to be split.
    train_size : float
        proportion of data to be split for training set. The default is 0.8.
    valid_size : float
        proportion of data to be split for validation set. The default is 0.1.

    Returns
    -------
    train_idx, valid_idx, test_idx : ndarray
        the indices of training, validation, and testing sets after being split.

    """
    
    train_idx, valid_idx, test_idx = np.split(index,[int(train_size*len(index)),
                                                     int((train_size+valid_size)*len(index))])
    return train_idx, valid_idx, test_idx