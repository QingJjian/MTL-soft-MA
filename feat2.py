import os
import numpy as np
import pandas as pd
import torch
from class10net import classifier
from BATM import Generator,Encoder,Discriminator
from TFIDF import feature_select
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import time
import math
import operator
from palmetto111 import Palmetto
from collections import defaultdict
import librosa
from tool import EarlyStopping
import random
import warnings
import soundfile as sound
from torch.autograd import Variable
warnings.filterwarnings('ignore') # scipy throws future warnings on fft (known bug)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def deltas(X_in):
    X_out = (X_in[:, :, 2:] - X_in[ :,:, :-2]) / 10.0
    X_out = X_out[:,:, 1:-1] + (X_in[ :,:, 4:] - X_in[:, :, :-4]) / 5.0
    return X_out
def gx(x):
    c = np.zeros((1, 400))
    for j in range(400):
        for m in range(1000):
            distance1 = []
            context = np.load('D:/kd1000.npy')
            context = context.reshape(128,-1)
            a = context[:, m]
            b = x[:,j]
            distance = np.sqrt(np.sum(np.square(a - b)))
            distance1.append(distance)
            # zuixiao = min(distance1)
            zuixiao = distance1.index(min(distance1))
            c[:,j] = zuixiao
    return c
def tfidf(list_words):
    # 总词频统计
    doc_frequency = defaultdict(int)
    for word_list in list_words:
        for i in word_list:
            doc_frequency[i] += 1

    # 计算每个词的TF值
    word_tf = {}  # 存储没个词的tf值
    for i in doc_frequency:
        word_tf[i] = doc_frequency[i] / sum(doc_frequency.values())

    # 计算每个词的IDF值
    doc_num = len(list_words)
    word_idf = {}  # 存储每个词的idf值
    word_doc = defaultdict(int)  # 存储包含该词的文档数
    for i in doc_frequency:
        for j in list_words:
            if i in j:
                word_doc[i] += 1
    for i in doc_frequency:
        word_idf[i] = math.log(doc_num / (word_doc[i] + 1))

    # 计算每个词的TF*IDF的值
    word_tf_idf = {}
    for i in doc_frequency:
        word_tf_idf[i] = word_tf[i] * word_idf[i]

    # 对字典按值由大到小排序
    dict_feature_select = sorted(word_tf_idf.items(), key=operator.itemgetter(1), reverse=True)
    return dict_feature_select
def k(x):
    c = np.zeros((1, 1000))
    for i in range(len(x)):
        a = list(x[i])
        a1 = int(a[0])
        a2 = a[1]
        c[:,a1] = a2
    return c
class Normalize(object):
    """Normalizes voice spectrogram (mean-varience)"""

    def __call__(self, spec):
        mu2 = spec.mean(axis=1).reshape(128, 1)
        sigma2 = spec.std(axis=1).reshape(128, 1)
        spec = (spec - mu2) / sigma2
        return spec

TRAIN_PATH = 'D:/fold1_train.csv'
TEST_PATH = 'D:/fold1_evaluate.csv'
DEVICE = 'cuda:0'
NUM_WORKERS = 8

# data = pd.read_csv(TRAIN_PATH, encoding='ASCII', sep='\t')
# specs = np.asarray(data.iloc[:, 0])
# print(specs.shape)
# label = np.asarray(data.iloc[:, 1])
# data_len = len(data.index)
# print('trainset',data_len)
# spec123 = np.empty((9185,1000))
# for i in range(data_len):
#     print('train',i)
#     single_sound_name = specs[i]
#     print(single_sound_name)
#     stereo,sr = sound.read('D:/'+single_sound_name)
#     stereo = np.asfortranarray(stereo)
#     spec = librosa.feature.melspectrogram(stereo[:,0], sr=48000, n_fft=2048, hop_length=1024, n_mels=128)
#     spec = np.log(spec+1e-8)
#     spec = gx(spec)
# #     # print('1',spec)
#     spec = tfidf(spec)
# #     # print('2',spec)
#     spec = k(spec)
#     spec = spec.astype(np.float32)
#     spec = torch.from_numpy(spec)
#     spec123[i] = spec
#     print(spec.shape)
# np.save('D:/topic1000train.npy',spec123)

# data = pd.read_csv(TEST_PATH, encoding='ASCII', sep='\t')
# specs = np.asarray(data.iloc[:, 0])
# print(specs.shape)
# label = np.asarray(data.iloc[:, 1])
# data_len = len(data.index)
# print('trainset',data_len)
# spec123 = np.empty((4185,1000))
# for i in range(data_len):
#     print('test',i)
#     single_sound_name = specs[i]
#     print(single_sound_name)
#     stereo,sr = sound.read('D:/'+single_sound_name)
#     stereo = np.asfortranarray(stereo)
#     spec = librosa.feature.melspectrogram(stereo[:,0], sr=48000, n_fft=2048, hop_length=1024, n_mels=128)
#     spec = np.log(spec+1e-8)
#     spec = gx(spec)
# #     # print('1',spec)
#     spec = tfidf(spec)
# #     # print('2',spec)
#     spec = k(spec)
#     spec = spec.astype(np.float32)
#     spec = torch.from_numpy(spec)
#     spec123[i,:] = spec
#     print(spec.shape)
# np.save('D:/topic1000test.npy',spec123)
a = np.load('D:/topic1000test.npy')
b = a[:,0:200]
np.save('D:/topic200test.npy',b)
c = a[:,0:400]
np.save('D:/topic400test.npy',c)
d = a[:,0:600]
np.save('D:/topic600test.npy',d)
e = a[:,0:800]
np.save('D:/topic800test.npy',e)
# print(a[:,0:300].shape)
