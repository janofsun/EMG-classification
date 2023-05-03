from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils import data
# from torchvision import datasets, transforms
from torch.autograd import Variable
import time
import numpy as np
import os
from copy import deepcopy
from itertools import chain
from sklearn.model_selection import train_test_split


class Net(nn.Module):
    def __init__(self, num_classes, hidden_size, num_layers, middle_feature):
        super(Net, self).__init__()
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

        self.embedding_dim = 129 * 4

        self.BiLSTM = nn.LSTM(input_size=self.embedding_dim,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              dropout=0.2,
                              bidirectional=self.bidirectional)

        self.MLP = nn.Sequential(
            nn.Linear(self.hidden_size, self.middle_feature),
            nn.ReLU(),
            nn.Linear(self.middle_feature, num_classes)
        )

    def forward(self, datas):
        batch, t, f = datas.shape
        print('input:', batch, f, t)
        datas = datas.view(batch, 1, f, t)
        print('reshape:', datas.size())
        embedded_datas = self.embedding(datas)
        print('after conv:', embedded_datas.size())
        embedded_datas = embedded_datas.permute(3, 0, 1, 2).view(t, batch, -1)
        print('before lstm:', embedded_datas.size())
        out, _ = self.BiLSTM(embedded_datas)
        # print('after lstm:',out.size())
        # out, _ =pad_packed_sequence(out)
        # out = out[-1,:,:]
        print('reshape:', out.size())
        out = self.MLP(out)
        return out


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

        # self.embedding_dim = 129
        self.embedding_dim = self.hidden_size

        self.BiLSTM = nn.LSTM(input_size=self.embedding_dim,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              dropout=0.2,
                              bidirectional=self.bidirectional,
                              batch_first=True)

        self.MLP = nn.Sequential(
            nn.Linear(self.hidden_size, self.middle_feature),
            nn.ReLU(),
            nn.Linear(self.middle_feature, num_classes)
        )  # remove the mlp and generate the irritation state at each time step

# only LSTM
    def forward(self, datas):
        # print('Before LSTM:{}'.format(datas.size()))
        out, _ = self.BiLSTM(datas)
        # print('After LSTM:{}'.format(out.size()))
        # out = out[-1, :, :]  # 20,512
        out = self.MLP(out)
        out = torch.squeeze(out)
        # print('output shape:{}'.format(out.shape))
        return out

# # CNN + LSTM
#     def forward(self, datas):
#         t, f = datas.shape
#         embedded_datas = self.embedding(datas)
#         print('After conv:{}\n'.format(embedded_datas.size())) # [20, 4, 54, 129]
#         embedded_datas = embedded_datas.permute(3, 0, 1, 2).view(f, 1, -1)
#         print('Before LSTM:{}'.format(datas.size()))
#         out, _ = self.BiLSTM(embedded_datas)
#         print('After LSTM:{}'.format(out.size()))
#         # out = out[-1, :, :]  # 20,512
#         out = self.MLP(out)
#         out = torch.squeeze(out)
#         # print('output shape:{}'.format(out.shape))
#         return out