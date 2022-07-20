import os
import numpy as np
import pandas as pd
import torch
# from BATM import BATM
from TFIDF import feature_select
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import time
import librosa
# from tool import EarlyStopping
import random
import warnings
import soundfile as sound
from torch.autograd import Variable
warnings.filterwarnings('ignore')
# c = np.zeros((1, 100))
# c = []
# for m in range(14400):
#     for i in range(400):
#         for j in range(1000):
#             context1 = np.load('/public/others/lengyan/ly-4/kd1000.npy')
#             context1 = context1.reshape(128,-1)
#             context2 = np.load('D:/logmel.npy')
#             context2 = context2.reshape(128,-1)
#             a = context1[:, j]
#             b = x[:,:,i]
#             distance = np.sqrt(np.sum(np.square(a - b)))
#             if distance == 0:
#                 c.append(str(j))# c[1,i] = j
# # a = np.load('D:/logmel.npy')
# # print(a.shape)

# data = pd.read_csv('D:/fold1_train.csv', encoding='ASCII', sep='\t')
# spec = np.asarray(data.iloc[:, 0])
# c = np.zeros((1, 400))
# spec1 = []
# for i in range(len(spec)):
#     stereo,sr = sound.read('D:/'+spec[i])
#     stereo = np.asfortranarray(stereo)
#     spec01 = librosa.feature.melspectrogram(stereo[:,0], sr=48000, n_fft=2048, hop_length=1024, n_mels=128)
#     spec01 = np.log(spec01+1e-8)
#     spec011 = spec01[:, 0:400]
#     for j in range(400):
#         for m in range(1000):
#             context = np.load('D:/kd1000.npy')
#             context = context.reshape(128,-1)
#             a = context[:, m]
#             b = spec011[:,j]
#             distance = np.sqrt(np.sum(np.square(a - b)))
#             if distance == 0:
#                 c[1,j] = m
#                 print(c)
#     spec1.append(c)
# np.save('D:/traintfidf.npy',c)
# print(spec1)
# a1 = []
# a = torch.zeros((1,5))
# b = torch.ones((1,5))
# e = torch.ones((1,5))
# f = torch.ones((1,5))
# # a.squeeze_(dim=1)
# c = a  +b+e+f
# c =c/3
# c = c[:,0:3]
# # a1.append(a)
# # a1.append(b)
# # list_add(a1),c
# print(c)
# xx = torch.randn(2,3, 4)
# x = torch.zeros(3, 4)
# xx = torch.sum(xx,dim=1)
# # x[1,:] = xx
# # print(xx)
# print(xx.shape)
a = np.load('D:/fold1_train400.npy')
print(a.shape)
# b = a[:,0:400]
# np.save('D:/fold1_evaluate400.npy',b)
