import matplotlib.pyplot as plt
# from kmeans_pytorch import kmeans, kmeans_predict
import numpy as np
import pandas as pd
import torch
import librosa
import soundfile as sound
from math import sqrt
from sklearn.cluster import KMeans
import os
from torch.autograd import Variable
# warnings.filterwarnings('ignore') # scipy throws future warnings on fft (known bug)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import soundfile as sf
import torch
from scipy.cluster.vq import *
import matplotlib.pyplot as plt

# TRAIN_PATH = '/public/others/lengyan/ly-2/new 2019/2019/bus.TXT'
# data_info = pd.read_csv(TRAIN_PATH, sep=',')
# spec = np.asarray(data_info.iloc[:, 0])
# spec11 = []
# spec22 = []
# for i in range(len(spec)):
#     single_sound_name = spec[i]
#     stereo, sr = sf.read('/public/others/lengyan/ly-2/new 2019/bus/'+single_sound_name)
#     stereo = np.asfortranarray(stereo)
#     spec01 = librosa.feature.melspectrogram(stereo[:, 0], sr=48000, n_fft=2048, hop_length=1024, n_mels=128)
#     spec02 = librosa.feature.melspectrogram(stereo[:, 1], sr=48000, n_fft=2048, hop_length=1024, n_mels=128)
#     spec01 = np.log(spec01 + 1e-8)
#     spec02 = np.log(spec02 + 1e-8)
#     spec1 = []
#     spec2 = []
#     for j in range(469):
#         spec01_j = spec01[:,j:j+1]
#         spec01_j = spec01_j.astype(np.float32)
#         # spec01_j = torch.from_numpy(spec01_j)
#         spec1.append(spec01_j)
#         # spec1 = np.array(spec1)
#         # spec11 = np.vstack(spec1)
#     for k in range(469):
#         spec02_k = spec02[:,k:k+1]
#         spec02_k = spec02_k.astype(np.float32)
#         # spec02_k = torch.from_numpy(spec02_k)
#         spec2.append(spec02_k)
#
#     spec11.append(spec1)
#     spec22.append(spec2)
#     # print(len(spec11))
# spec11 = np.array(spec11)
# spec22 = np.array(spec22)
# spec11 = spec11.reshape(-1,128)
# spec22 = spec22.reshape(-1,128)
# spec666 = np.concatenate((spec11,spec22),axis=0)

a = np.load('/public/others/lengyan/ly-2/new 2019/kd1000.npy')
spec666 = a.reshape(-1,128)

qq1 = []
AUD = []
for l in range(7):
    test_data= []
    batch = np.random.choice(spec666[0],300,replace=True, p=None)
    batch = np.squeeze(batch)
    print(batch.shape)
    for m in range(300):
        abc = int(batch[m])
        test_data.append(spec666[abc,:])

    test_data = np.array(test_data)

    # print(test_data.shape,test_data)

    k_means = KMeans(n_clusters=10, random_state=10)
    k_means.fit(test_data)

    qq=k_means.cluster_centers_
    # qq_mean = np.mean(qq,axis=0)
    # print(qq.shape)
    # AUD.append(k_means)shape
    qq1.append(qq)
# # AUD = np.array(AUD)
# # array = array.tolist()
qq = np.array(qq1)
np.save('/public/others/lengyan/ly-2/new 2019/k70.npy',qq)
# print(AUD)
# print(qq1)