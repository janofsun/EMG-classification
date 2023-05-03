#from google.colab import drive
#drive.mount('/content/drive')
#from __future__ import print_function
import numpy as np
import os
import torch
from copy import deepcopy
from itertools import chain
from sklearn.model_selection import train_test_split
import scipy.signal as s
import matplotlib.pyplot as plt
from util import *
import torch.nn.utils.rnn as rnn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from PyEMD import EMD
from PyEMD import EEMD
"""
Author:Xuefan Zha
Data source: CDI
Data structure explain:
for every file
#1 channel number: two channel in total, used for seek
#2 total number of record
#3 observation interval
#4 sampling rate:1200HZ
#5 A-D upper voltage
#6 A-D lower voltage
#7 amplifier gain
#8 amplifier low band
#9 amplifier high band
for every epoch:
#1 record label
#2 epoch start time
#3 epoch length(number of data point)
#4- data for each epoch
"""

def eemd_preprocess(record_data, signal):
  # EEMD
  eemd = EEMD()
  # Say we want detect extrema using parabolic method
  emd = eemd.EMD
  emd.extrema_detection="parabol"
  fs = np.linspace(0,1,1200)
  for i in range(record_data.shape[0]):
      # Execute EEMD on S
      eIMFs = eemd.eemd(record_data[i], fs)
      record_data[i] = eIMFs[0]
      signal[1200*i:1200*(i+1)] = eIMFs[0]
  return record_data,signal


def spectro(record_data, signal,length_of_win,fft_length):
    # normspectro(data_path, 100, 256)
    fs = 1200
    win = s.get_window('hann', length_of_win) # the number of samples in the window// win:ndarray[length_of_win]
    raw = []
    for i in range(record_data.shape[0]):
        [f, t, stft_signal] = s.spectrogram(signal[i*1200:(i+1)*1200+1],fs,
                                    window = win,
                                    nperseg = length_of_win,
                                    noverlap = 0.8*length_of_win,
                                    nfft = fft_length)
        raw.append(10*np.log10(abs(stft_signal)))
    raw = np.asarray(raw)
    return raw # raw

#turn into spectrogram
def normspectro(datapath,length_of_win,fft_length,start,end):
    # normspectro(data_path, 100, 256, 0, 100)
    epoch_time, record_data, signal = readdata(datapath)
    # len(signal) == record_data.shape[0] * record_data.shape[1]
    print("Record data shape:{}\n".format(record_data.shape))
    fs = 1200
    win = s.get_window('hann', length_of_win) # the number of samples in the window// win:ndarray[length_of_win]
    norm_signal = []
    for i in range(start, end):
        [f, t, stft_signal] = s.spectrogram(signal[i*1200+40:(i+1)*1200], fs,
                                    window=win,
                                    nperseg=length_of_win,
                                    noverlap=0.8*length_of_win,
                                    nfft = fft_length)
        stft_signal = abs(stft_signal)
        norm_signal.append(stft_signal)
    norm_signal = np.hstack(norm_signal)
    reference_mean_norm = 10*np.log10(np.mean(norm_signal,axis = 1))
    reference_std_norm = np.std(norm_signal,axis = 1)
    raw = []
    spectrogram_mean = []
    spectrogram_std = []
    for i in range(record_data.shape[0]):
        [f, t, stft_signal] = s.spectrogram(signal[i*1200+40:(i+1)*1200+1],fs,
                                    window = win,
                                    nperseg = length_of_win,
                                    noverlap = 0.8*length_of_win,
                                    nfft = fft_length)
        raw.append(10*np.log10(abs(stft_signal)))
        s_mean = 10*np.log10(abs(stft_signal)) - np.tile(reference_mean_norm,(abs(stft_signal).shape[1],1)).T
        s_std = abs(stft_signal) - np.tile(reference_std_norm,(abs(stft_signal).shape[1],1)).T
        spectrogram_mean.append(s_mean)
        spectrogram_std.append(s_std)
    raw = np.asarray(raw)
    spectrogram_mean = np.asarray(spectrogram_mean)
    spectrogram_std = np.asarray(spectrogram_std)
    return t*1200, f, spectrogram_mean, spectrogram_std # raw
'''
#turn into spectrogram
def normspectro(datapath,length_of_win,fft_length):
    epoch_time,record_data,signal = readdata(datapath)
    fs = 1200
    win = s.get_window('hann',length_of_win)
    norm_signal = template_spectro(length_of_win,fft_length)
    reference_mean_norm = 10*np.log10(np.mean(norm_signal,axis = 1))
    reference_std_norm = np.std(norm_signal,axis = 1)
    raw = []
    spectrogram_mean = []
    spectrogram_std = []
    for i in range(record_data.shape[0]):
        [f,t,stft_signal] = s.spectrogram(signal[i*1200+40:(i+1)*1200+1],fs,
                                    window = win,
                                    nperseg = length_of_win,
                                    noverlap = 0.8*length_of_win,
                                    nfft = fft_length)
        #print('raw:',stft_signal.shape)
        raw.append(10*np.log10(abs(stft_signal)))
        s_mean = 10*np.log10(abs(stft_signal)) - np.tile(reference_mean_norm,(abs(stft_signal).shape[1],1)).T
        s_std = abs(stft_signal) - np.tile(reference_std_norm,(abs(stft_signal).shape[1],1)).T
        ##################no normal
        #s_mean = 10*np.log10(abs(stft_signal))
        #s_std = abs(stft_signal)
        spectrogram_mean.append(s_mean)
        spectrogram_std.append(s_std)
    raw = np.asarray(raw)
    spectrogram_mean = np.asarray(spectrogram_mean)
    spectrogram_std = np.asarray(spectrogram_std)
    return t*1200,raw,spectrogram_mean,spectrogram_std


#for only one file
'''
def threshold(spectrogram,threshold):
    #return index of time
    #spectrogram is 1* time
    f_mean = np.mean(spectrogram[:,22:,:],axis = 1)
    index = []
    for i in range(spectrogram.shape[0]):
        #judge = f_mean[i,:]>threshold
        if any(f_mean[i,:]>threshold):
            index.append(i+1)
    #total = np.arange(len(f_mean))
    #index = total[abs(f_mean)>threshold]
    return np.asarray(index)
'''

def threshold(total_spectrogram,threshold):
    #return index of time
    #spectrogram is 1* time
  total_index = []
  for j in range(len(total_spectrogram)):
    spectrogram = total_spectrogram[j]
    f_mean = np.mean(spectrogram[:,22:,:],axis = 1)
    index = []
    for i in range(spectrogram.shape[0]):
        #judge = f_mean[i,:]>threshold
        if any(f_mean[i,:]>threshold):
            index.append(i+1)
    #total = np.arange(len(f_mean))
    #index = total[abs(f_mean)>threshold]
    total_index.append(np.asarray(index))
  return np.asarray(total_index)


def getlabel(label_path,total_burst_index,total_length,total_time):
  files= os.listdir(label_path)
  total_irr_index = []
  for file in files:
    f = np.load(label_path+"/"+file,allow_pickle=True)
    total_irr_index.append(f.item().get('irritation'))
  total_output = []
  for i in range(len(total_length)):
    length = total_length[i]
    time = total_time[i]
    irr_index = total_irr_index[i]
    burst_index = total_burst_index[i]
    output = np.zeros([length*len(time),2])   
    for s in burst_index:
        output[(s-1)*len(time):s*len(time),0] = 1
    for i in irr_index:
        output[(i-1)*len(time):i*len(time),1] = 1
    total_output.append(output)
  return np.asarray(total_output)      
    
#merge
'''
def get_data_loaders(total_data,total_label):
  total_train = []
  total_label_train = []
  total_length = []
  for i in range(len(total_data)):
    data = total_data[i]
    label = total_label[i]
    feature = np.reshape(data,(data.shape[0]*data.shape[2],data.shape[1]))
    print('patient data',feature.shape)
    #train,val,label_train,label_val = train_test_split(feature,label,test_size = 0.2,random_state = 42)
    #train = torch.from_numpy(feature)
    #val = torch.from_numpy(val)
    #label_train = torch.from_numpy(label)
    #label_val = torch.from_numpy(label_val)
    total_train.append(feature)
    total_label_train.append(label)
    total_length.append(len(label))
  total_train = pad_sequence([torch.from_numpy(total_train[i]) for i in range(len(total_train))])
  total_label_train = pad_sequence([torch.from_numpy(total_label_train[i]) for i in range(len(total_label_train))])
  total_train = total_train.view(total_train.size(1),total_train.size(0),total_train.size(2))
  total_label_train = total_label_train.view(total_label_train.size(1),total_label_train.size(0),total_label_train.size(2))
  #total_train = [torch.from_numpy(total_train[i]) for i in range(len(total_train))]
  #total_label_train = [torch.from_numpy(total_label_train[i]) for i in range(len(total_label_train))]
  total_length = torch.from_numpy(np.asarray(total_length))
  return total_train,total_label_train,total_length
'''
def get_data_loaders(total_data,total_label):
  total_train = []
  label_train = []
  total_length = []
  for i in range(len(total_data)):
    label = total_label[i]
    feature = total_data[i]
    #feature = np.reshape(data,(data.shape[0]*data.shape[2],data.shape[1]))
    print('patient data',feature.shape)
    total_train.append(feature)
    label_train.append(label)
    assert len(label) == len(feature)
    total_length.append(len(label))
  total_train = np.vstack(total_train)  
  label_train = np.vstack(total_label)
  label_train = label_train[:,0]
  zero_index = np.argwhere(label_train==0)
  irr_index = np.argwhere(label_train==1)
  evoked_index = np.argwhere(label_train==2)
  baseline_index = np.argwhere(label_train==3)
  strong_index = np.argwhere(label_train==4)
  weak_index = np.argwhere(label_train==5)
  arti_index = np.argwhere(label_train==6)
  print(len(zero_index),len(irr_index),len(evoked_index),len(baseline_index),len(strong_index),len(weak_index),len(arti_index))
  total_length = torch.from_numpy(np.asarray(total_length))
  total_train = torch.from_numpy(total_train)
  label_train = torch.from_numpy(label_train)
  return total_train,label_train,total_length

'''
# data_path = 'H:/project622/classification/code/rawdata/ai_0504.bin'
# time,raw,spectrogram_norm,spectrogram_std = normspectro(data_path,100,256)
# #power_mean = np.mean(spectrogram_norm,axis = 0)
# burst_index = threshold(spectrogram_norm,15)
# output = getlabel(irr_index,burst_index,6574,time)
'''


#plot raw data
#data_path = 'rawdata/ai_0504.bin'
#epoch_time,record_data,signal = readdata(data_path)

#plt.plot(signal[1200*804:1200*805+1])
#plt.show()

#a = timetoepoch(time[index],1200)
#win = s.get_window('hann',100)
#a,b,c = s.stft(signal,1200,window = win,nperseg = 100,noverlap = 80,nfft = 256)
'''#power_mean = np.mean(spectrogram_norm,axis = 0)
#f_mean = np.mean(spectrogram_norm,axis = 2)

#plt.plot(f_mean[0,:])
#plt.ylim([-20,30])'''