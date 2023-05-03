# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:02:57 2019

@author: Nancy
"""
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
import numpy as np
import os
import scipy.signal as s

def readdata(data_path):
    data = np.fromfile(data_path) # open .bin file using np.fromfile
    #this is public parameter
    channel_number = int(data[0])
    total_record = int(data[1])
    print("channel_number:{}\ntotal_record:{}\n".format(channel_number, total_record))
    record_label = [] # epoch_label : [1 2 3 ... 5744 5745 5746]
    epoch_time = []
    epoch_length = []
    record_data = []
    signal = np.zeros(total_record*1200)
    for i in range(total_record):
        record_label.append(int(data[1203*i + 9]))
        epoch_time.append(data[1203*i + 10])
        epoch_length.append(int(data[1203*i + 11]))
        start_point = 1203*i + 12
        end_point = int(start_point + 1200)
        record_data.append(data[start_point:end_point])
        signal[i*1200:(i+1)*1200] = data[start_point:end_point]
    record_data = np.asarray(record_data)
    assert (len(signal) == record_data.shape[0] * record_data.shape[1])
    return epoch_time,record_data,signal

def timetoepoch(time,fs):
    # time list
    epoch = [i//fs for i in time]
    ss = set(epoch)
    epoch_index = [s for s in ss]
    epoch_index = np.asarray(epoch_index)
    epoch_index = epoch_index[epoch_index>100]
    epoch_index.sort()
    return epoch_index

def epochtotime(epochnum,fs):
    epoch = np.arange(epochnum)
    time =  [i*fs for i in epoch]
    return time

def getlabel(irr_index,burst_index,length,time):
    output = np.zeros([length*len(time),2])
    
    for s in burst_index:
        output[(s-1)*len(time):s*len(time),0] = 1
    for i in irr_index:
        output[(i-1)*len(time):i*len(time),1] = 1
    return output
'''
def getalllabel(label_path,total_length,total_time):
  files= os.listdir(label_path)
  total_label = []
  for file in files:
    print(file)
    f = np.load(label_path+"/"+file,allow_pickle=True)
    total_label.append(f)
  total_output = []
  for i in range(len(total_length)):
    length = total_length[i]
    time = total_time[i]
    label = total_label[i]
    output = np.zeros([length*len(time),1])
    #art_index = np.argwhere(f[:,0]==9)
    burst_index = np.argwhere(f[:,0]==1)[:,0]
    for i in burst_index:
        output[i*len(time):(i+1)*len(time),0] = 1
    irr_index = np.argwhere(f[:,0]==2)[:,0]
    for i in irr_index:
        output[i*len(time):(i+1)*len(time),0] = 2
    evoked_index = np.argwhere(f[:,0]==4)[:,0]
    for i in evoked_index:
        output[i*len(time):(i+1)*len(time),0] = 3
    baseline_index = np.argwhere(f[:,0]==5)[:,0]
    for i in baseline_index:
        output[i*len(time):(i+1)*len(time),0] = 4
    strong_index = np.argwhere(f[:,0]==7)[:,0]
    for i in strong_index:
        output[i*len(time):(i+1)*len(time),0] = 5
    weak_index = np.argwhere(f[:,0]==8)[:,0]
    for i in weak_index:
        output[i*len(time):(i+1)*len(time),0] = 6
    total_output.append(output)
  return np.asarray(total_output)'''

def getalllabel(label_path,total_length,total_time):
  files= os.listdir(label_path)
  total_label = []
  irr_len = 0
  for file in files:
    print(file)
    f = np.load(label_path+"/"+file,allow_pickle=True)
    total_label.append(f)
  total_output = []
  for i in range(len(total_length)):
    length = total_length[i]
    time = total_time[i]
    label = total_label[i]
    output = np.zeros_like(label)
    #########
    
    #art_index = np.argwhere(f[:,0]==9)
    #burst_index = np.argwhere(label[:,0]==1)[:,0]
    #for i in burst_index:
    #    output[i:(i+1),0] = 1
    irr_index = np.argwhere(label[:,0]==2)[:,0]
    for j in irr_index:
        output[j:(j+1),0] = 1
    evoked_index = np.argwhere(label[:,0]==4)[:,0]
    for j in evoked_index:
        output[j:(j+1),0] = 0
    baseline_index = np.argwhere(label[:,0]==5)[:,0]
    for j in baseline_index:
        output[j:(j+1),0] = 1
    strong_index = np.argwhere(label[:,0]==7)[:,0]
    for j in strong_index:
        output[j:(j+1),0] = 1
    weak_index = np.argwhere(label[:,0]==8)[:,0]
    for j in weak_index:
        output[j:(j+1),0] = 1
    '''artifact_index = np.argwhere(label[:,0]==9)[:,0]
    for j in artifact_index:
        output[j:(j+1),0] = 6'''
    irr_len += len(irr_index) + len(baseline_index) + len(strong_index) + len(weak_index)
    total_output.append(output)
  return np.asarray(total_output),irr_len
#def findthreshold()

def get_one_patient(label_path,length,time):
    #label path is full name
    f = np.load(label_path,allow_pickle=True)
    label = f
    #output = np.zeros(length*len(time))
    output = np.zeros(length*len(time))
    irr_index = np.argwhere(f[:,0]==2)[:,0]
    for j in irr_index:
        output[j*len(time):(j+1)*len(time)] = 1
    evoked_index = np.argwhere(f[:,0]==4)[:,0]
    for j in evoked_index:
        output[j*len(time):(j+1)*len(time)] = 2
    baseline_index = np.argwhere(f[:,0]==5)[:,0]
    for j in baseline_index:
        output[j*len(time):(j+1)*len(time)] = 3
    strong_index = np.argwhere(f[:,0]==7)[:,0]
    for j in strong_index:
        output[j*len(time):(j+1)*len(time)] = 4
    weak_index = np.argwhere(f[:,0]==8)[:,0]
    for j in weak_index:
        output[j*len(time):(j+1)*len(time)] = 5
    artifact_index = np.argwhere(f[:,0]==9)[:,0]
    for j in artifact_index:
        output[j*len(time):(j+1)*len(time)] = 6
    #irr_len = len(irr_index) + len(baseline_index) + len(strong_index) + len(weak_index)
    return np.asarray(output)

def noartifact_label(label_path,length,time):
    #label path is full name
    f = np.load(label_path,allow_pickle=True)
    label = f
    arifact_index = np.argwhere(f[:,0]==9)[:,0]
    #output = np.zeros(length*len(time))
    output = np.zeros((length - len(arifact_index))*len(time))
    burst_index = np.argwhere(f[:,0]==1)[:,0]
    for j in burst_index:
        output[j*len(time):(j+1)*len(time)] = 1
    irr_index = np.argwhere(f[:,0]==2)[:,0]
    #print(irr_index)
    for j in irr_index:
        output[j*len(time):(j+1)*len(time)] = 2
    evoked_index = np.argwhere(f[:,0]==4)[:,0]
    for j in evoked_index:
        output[j*len(time):(j+1)*len(time)] = 3
    baseline_index = np.argwhere(f[:,0]==5)[:,0]
    for j in baseline_index:
        output[j*len(time):(j+1)*len(time)] = 4
    strong_index = np.argwhere(f[:,0]==7)[:,0]
    for j in strong_index:
        output[j*len(time):(j+1)*len(time)] = 5
    weak_index = np.argwhere(f[:,0]==8)[:,0]
    for j in weak_index:
        output[j*len(time):(j+1)*len(time)] = 6
    return np.asarray(output),arifact_index

def template_spectro(length_of_win,fft_length): 
    fs = 1200
    win = s.get_window('hann',length_of_win)
    norm_signal = []
    epoch_time,record_data,signal = readdata('rawdata/ai_0501.bin')
    for i in range(590,596):
        [f,t,stft_signal] = s.spectrogram(signal[i*1200+40:(i+1)*1200],fs,
                                    window = win,
                                    nperseg = length_of_win,
                                    noverlap = 0.8*length_of_win,
                                    nfft = fft_length)
        stft_signal = abs(stft_signal)
        norm_signal.append(stft_signal)
    '''epoch_time,record_data,signal = readdata('rawdata/ai_0502.bin')
    for i in range(2880,2890):
        [f,t,stft_signal] = s.spectrogram(signal[i*1200+40:(i+1)*1200],fs,
                                    window = win,
                                    nperseg = length_of_win,
                                    noverlap = 0.8*length_of_win,
                                    nfft = fft_length)
        stft_signal = abs(stft_signal)
        norm_signal.append(stft_signal)
    epoch_time,record_data,signal = readdata('rawdata/ai_0503.bin')
    for i in range(0,6):
        [f,t,stft_signal] = s.spectrogram(signal[i*1200+40:(i+1)*1200],fs,
                                    window = win,
                                    nperseg = length_of_win,
                                    noverlap = 0.8*length_of_win,
                                    nfft = fft_length)
        stft_signal = abs(stft_signal)
        norm_signal.append(stft_signal)
    epoch_time,record_data,signal = readdata('rawdata/ai_0505.bin')
    for i in range(79,85):
        [f,t,stft_signal] = s.spectrogram(signal[i*1200+40:(i+1)*1200],fs,
                                    window = win,
                                    nperseg = length_of_win,
                                    noverlap = 0.8*length_of_win,
                                    nfft = fft_length)
        stft_signal = abs(stft_signal)
        norm_signal.append(stft_signal)
    epoch_time,record_data,signal = readdata('rawdata/ai_0506.bin')
    for i in range(237,242):
        [f,t,stft_signal] = s.spectrogram(signal[i*1200+40:(i+1)*1200],fs,
                                    window = win,
                                    nperseg = length_of_win,
                                    noverlap = 0.8*length_of_win,
                                    nfft = fft_length)
        stft_signal = abs(stft_signal)
        #print(stft.shape)
        norm_signal.append(stft_signal)'''
    norm_signal = np.hstack(norm_signal)
    #print(norm_signal.shape)
    return norm_signal