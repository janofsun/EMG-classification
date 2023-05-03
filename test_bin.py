import matplotlib.pyplot as plt
import time
import scipy
import scipy.io
from scipy.fft import fft
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils import data
from torch.autograd import Variable
import pandas as pd
from util import *
from dataloader import *
from sklearn.model_selection import train_test_split

def save_model(path, epoch, model, optimizer):
    ckpt = {
    "epoch" : epoch,
    "model_state_dict" : model.state_dict(),
    "optimizer_state_dict" : optimizer.state_dict(),
    # "scheduler_state_dict" : scheduler.state_dict(),
    }
    print("saving model to:", path)
    torch.save(ckpt,path)

record_data0501 = np.load('dataset/rawdata/record_data0501.npy')
record_data0502 = np.load('dataset/rawdata/record_data0502.npy')
record_data0503 = np.load('dataset/rawdata/record_data0503.npy')
record_data0505 = np.load('dataset/rawdata/record_data0505.npy')
record_data0506 = np.load('dataset/rawdata/record_data0506.npy')
labels0501 = np.load('dataset/label/label0501.npy')
labels0502 = np.load('dataset/label/label0502.npy')
labels0503 = np.load('dataset/label/label0503.npy')
labels0505 = np.load('dataset/label/label0505.npy')
labels0506 = np.load('dataset/label/label0506.npy')

signal0501 = record_data0501.flatten()
signal0502 = record_data0502.flatten()
signal0503 = record_data0503.flatten()
signal0505 = record_data0505.flatten()
signal0506 = record_data0506.flatten()

def reference(signal,length_of_win,fft_length,start,end):
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
    return reference_mean_norm

def normspectrogram(record_data,signal,length_of_win,fft_length,start,end):
    # normspectro(data_path, 100, 256, 0, 100)
    # epoch_time, record_data, signal = readdata(datapath)
    # len(signal) == record_data.shape[0] * record_data.shape[1]
    # print("Record data shape:{}\n".format(record_data.shape))
    fs = 1200
    win = s.get_window('hann', length_of_win) # the number of samples in the window// win:ndarray[length_of_win]
    norm_signal = []
    for i in range(start, end):
        [f, t, stft_signal] = s.spectrogram(signal[i*1200:(i+1)*1200], fs,
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
        [f, t, stft_signal] = s.spectrogram(signal[i*1200:(i+1)*1200+1],fs,
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

class Netone(nn.Module):
    def __init__(self, num_classes, hidden_size, num_layers, middle_feature):
        super(Netone, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = False
        self.middle_feature = middle_feature

        self.embedding = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=(3, 1), padding=(1, 0), stride=1),
            nn.ReLU(),
            nn.Conv2d(2, 4, kernel_size=(3, 1), padding=(1, 0), stride=1),
            nn.ReLU())

        self.embedding_dim = 129*4
        # self.embedding_dim = 51*4
        # h0 = 256 num_layers = 1

        self.BiLSTM = nn.LSTM(input_size=self.embedding_dim,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              dropout=0.2,
                              bidirectional=self.bidirectional)

        self.MLP = nn.Sequential(
            nn.Linear(self.hidden_size, self.middle_feature),
            nn.ReLU(),
            nn.Linear(self.middle_feature, num_classes)
        )  # remove the mlp and generate the irritation state at each time step

    def forward(self, datas):
        batch, f, t = datas.shape
        # print('input:', batch, f, t)# overall patient data = [1, 129, 54]
        # datas = datas.view(batch, 1, t, f)  # [N, C, H, W]
        datas = datas.reshape(batch, 1, t, f)
        embedded_datas = self.embedding(datas)
        # print('after conv:',embedded_datas.size()) # [1, 4, 56, 129]
        # embedded_datas = embedded_datas.permute(3, 0, 1, 2).view(f, batch, -1)
        embedded_datas = embedded_datas.permute(2, 0, 1, 3).reshape(t, batch, -1)
        # print('before lstm:',embedded_datas.size())
        out, _ = self.BiLSTM(embedded_datas)
        # print('after lstm:',out.size()) # [56, 1, 256]
        # out = out[-1, :, :]  # 20,512
        out = self.MLP(out.squeeze())
        # print('output shape:', out.shape)
        return out

length_of_win,fft_length,start,end, fs = 100,256,0,100,1200
reference_0501 = reference(signal0501,length_of_win,fft_length,start,end)
reference_0502 = reference(signal0502,length_of_win,fft_length,start,end)
reference_0503 = reference(signal0503,length_of_win,fft_length,start,end)
reference_0505 = reference(signal0505,length_of_win,fft_length,start,end)
reference_0506 = reference(signal0506,length_of_win,fft_length,start,end)

time, raw, spectrogram_norm0501, spectrogram_std = normspectrogram(record_data0501, signal0501, 100, 256, 0, 100)
time, raw, spectrogram_norm0502, spectrogram_std = normspectrogram(record_data0502, signal0502, 100, 256, 2533,2633)
time, raw, spectrogram_norm0503, spectrogram_std = normspectrogram(record_data0503, signal0503, 100, 256, 0, 100)
time, raw, spectrogram_norm0505, spectrogram_std = normspectrogram(record_data0505, signal0505, 100, 256, 0, 100)
time, raw, spectrogram_norm0506, spectrogram_std = normspectrogram(record_data0506, signal0506, 100, 256, 0, 100)

# train_record_data = np.concatenate((spectrogram_norm0501, spectrogram_norm0502, spectrogram_norm0503, spectrogram_norm0505), axis=0)
# train_label = np.concatenate((labels0501, labels0502, labels0503, labels0505), axis=0)
# train_signal = np.concatenate((signal0501, signal0502, signal0503, signal0505))
# val_record_data, val_signal, val_label = record_data0506, signal0506, labels0506

train_record_data = np.concatenate((spectrogram_norm0501, spectrogram_norm0502, spectrogram_norm0506[start:end]), axis=0)
train_label = np.concatenate((labels0501, labels0502, labels0506[start:end]), axis=0)
train_signal = np.concatenate((signal0501, signal0502, signal0506[start*fs:end*fs]))
val_record_data, val_signal, val_label = record_data0506[end:], signal0506[end*1200:], labels0506[end:]

train_label = np.repeat(train_label, 56)
val_label = np.repeat(val_label, 56)

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,float)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

fs = 1200
length_of_win = 100
length_of_win_overlap = 20
fft_length = 256
win = s.get_window('hann', length_of_win)

val_epochs = val_record_data.shape[0] * ((fs-100)//length_of_win_overlap + 1)

eps = 1
#set hidden size
hidden_size = 256
n_class = 9
learning_rate = 0.01
max_epochs = 20
middle_features = 128
model = Netone(num_classes=n_class, hidden_size=hidden_size, num_layers=1, middle_feature=middle_features) #num_classes, hidden_size, num_layers, bidirectional, middle_feature
#model.cuda()
# criterion = nn.CrossEntropyLoss()
criterion = FocalLoss()
optimizer = torch.optim.Adam(model.parameters())

train_acc = []
val_acc = []

for epoch in range(max_epochs):
    correct_all = 0
    total_loss = 0
    train_length = 0
    predicted_train_csv = []
    real_train_csv = []

    for idx in range(train_record_data.shape[0]*56):
      train_raw = torch.from_numpy(train_record_data[idx//56,:,idx%56]).unsqueeze(0).unsqueeze(2)
      # print(train_raw.shape)
      label_raw = train_label[idx]
      label = torch.tensor(label_raw)
      batch_x = Variable(train_raw.float())
      batch_y = label.long()
      output = model(batch_x)
      optimizer.zero_grad()
      _, predicted_train = torch.max(output.data,0)
      # print(predicted_train)
      if label_raw == predicted_train: 
        correct_all += 1
    #   correct_all += sum(batch_y.eq(predicted_train))
      train_length += 1

      predicted_train_csv.append(predicted_train.cpu().detach().numpy())
      real_train_csv.append(int(label_raw))

      label_raw_tensor = torch.tensor([label_raw], dtype=torch.long)
      loss = criterion(output.unsqueeze(0), label_raw_tensor)#.float().reshape(-1, 1)
      total_loss += loss
      loss.backward()
      optimizer.step()

    print('\n************************************************************************************************')
    print("Epoch {}/{}:, Train Loss {:.04f}, Learning Rate {:.04f}".format(
        epoch+1,
        max_epochs,
        float(total_loss / train_length),
        float(optimizer.param_groups[0]['lr'])))
    print('train_acc:{}\n'.format(correct_all/train_length))

    np.save('logs/paperRes/val06_train_focal_pred_'+ str(correct_all / train_length) + '.npy', predicted_train_csv)
    np.save('logs/paperRes/val06_train_focal_real_'+ str(correct_all / train_length) + '.npy', real_train_csv)

    path = 'logs/bin_model/cnn_lstm_focal_bin_ep' + str(epoch) + ".pt"
    save_model(path, epoch, model, optimizer)

    if epoch%5==0:
        correct_all = 0
        predicted_val_csv = []
        real_val_csv = []
        val_length = 0
        # assert(val_signal.shape[0]//length_of_win == val_label.shape[0])
        for idx in range(val_epochs):
            # tic = time.perf_counter()
            start = idx//56
            shift = idx%56
            start = start*fs + shift*length_of_win_overlap
            end = start + length_of_win + 1
            val_signals = val_signal[start:end]
            [f, t, stft_signal] = s.spectrogram(val_signals, fs,
                                    window=win,
                                    nperseg=length_of_win,
                                    noverlap=0.8*length_of_win ,
                                    nfft=fft_length)

            val_raw = np.asarray(10 * np.log10(abs(stft_signal))) - np.tile(reference_0506,(abs(stft_signal).shape[1],1)).T
            val_raw = torch.from_numpy(val_raw).unsqueeze(0)
            # print(val_raw.shape)
            label_raw = val_label[idx]
            label_raw = torch.from_numpy(np.array(label_raw))
            batch_x = Variable(val_raw.float())  # .cuda()
            output = model(batch_x)  # ,hidden = model(batch_x,None)
            # toc = time.perf_counter()
            # print(f"Test time cost in {toc - tic:0.4f} seconds")
            _, predicted_val = torch.max(output.data, 0)
            # print("predicted_val shape:{}\nbatch_y shape:{}\n".format(predicted_val.shape, batch_y.shape))
            # if len(predicted_val)!=len(batch_y): 
            #   print("label shape:{}\nval_raw shape:{}\nlabel_raw shape:{}".format(label.shape, val_raw.shape, label_raw.shape))
            if label_raw == predicted_val: correct_all += 1
            val_length += 1
            predicted_val_csv.append(predicted_val.cpu().detach().numpy())
            real_val_csv.append(label_raw.cpu().detach().numpy())

        print('\n************************************************************************************************')
        print("Epoch {}/{}:\ntest_acc:{}\n".format(epoch+1, max_epochs, correct_all/val_length))

        np.save('logs/paperRes/val06_val_focal_pred_'+ str(correct_all / val_length) + '.npy', predicted_val_csv)
        np.save('logs/paperRes/val06_val_focal_real_'+ str(correct_all / val_length) + '.npy', real_val_csv)
