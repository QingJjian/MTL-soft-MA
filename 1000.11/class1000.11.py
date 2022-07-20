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


class IdentificationDataset(Dataset):
    def __init__(self, csv_path,transform=None):
        """
        Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径
            transform: transform 操作
        """
        # Transforms
        #self.to_tensor = transforms.ToTensor()
        # 读取 csv 文件
        self.data_info = pd.read_csv(csv_path, encoding='ASCII', sep='\t')

        # 文件第一列包含图像文件的名称
        self.spec = np.asarray(self.data_info.iloc[:, 0])
        # 第二列是图像的 label
        self.label = np.asarray(self.data_info.iloc[:, 1])
        self.spec1 = np.load('./topic1000test.npy')
        # 第三列是决定是否进行额外操作
        # 计算 length
        self.data_len = len(self.data_info.index)
        self.transform = transform

    def __getitem__(self, index):
        # 从 pandas df 中得到文件名
        single_sound_name = self.spec[index]
        # 读取图像文件
        stereo,sr = sound.read('D:/'+single_sound_name)
        stereo = np.asfortranarray(stereo)
        spec = librosa.feature.melspectrogram(stereo[:,0], sr=48000, n_fft=2048, hop_length=1024, n_mels=128)
        spec = np.log(spec+1e-8)
        spec = spec[:, 0:400]
        # spec = gx(spec1)
        # # print('1',spec)
        # spec = tfidf(spec)
        # # print('2',spec)
        # spec = k(spec)
        spec = spec.astype(np.float32)
        spec = torch.from_numpy(spec)
        spec = np.expand_dims(spec,axis=0)
        spec1 = self.spec1[index]
        label = self.label[index]
        return spec, spec1, label

    def __len__(self):
        return self.data_len

class Normalize(object):
    """Normalizes voice spectrogram (mean-varience)"""

    def __call__(self, spec):
        mu2 = spec.mean(axis=1).reshape(128, 1)
        sigma2 = spec.std(axis=1).reshape(128, 1)
        spec = (spec - mu2) / sigma2
        return spec

if __name__ == "__main__":
    transforms = Compose([Normalize()])

    TRAIN_PATH = './fold1_evaluate2.csv'
    TEST_PATH = './fold1_evaluate1.csv'

    DEVICE = 'cuda:0'
    NUM_WORKERS = 4
    trainset = IdentificationDataset(TRAIN_PATH,transform=transforms)
    trainsetloader = torch.utils.data.DataLoader(trainset, batch_size=8, num_workers=NUM_WORKERS, shuffle=True)

    testset = IdentificationDataset(TEST_PATH,transform=transforms)
    testsetloader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=NUM_WORKERS)
    n_topic = 11
    hid_dim = 1024
    bow_dim = 1000
    generator = Generator(n_topic=n_topic,hid_dim=hid_dim,bow_dim=bow_dim)
    generator = generator.to(DEVICE)
    encoder = Encoder(bow_dim=bow_dim,hid_dim=hid_dim,n_topic=n_topic)
    encoder = encoder.to(DEVICE)
    discriminator = Discriminator(bow_dim=bow_dim,n_topic=n_topic,hid_dim=hid_dim)
    discriminator = discriminator.to(DEVICE)
    model = classifier()
    model.to(DEVICE)
    #model = torch.nn.DataParallel(model, DEVICE_ids=[0, 1, 2, 3])
    epochs = 300
    LR_INIT = 1e-2
    LR_LAST = 1e-5
    learning_rate=1e-4
    beta1=0.5
    beta2=0.999
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    gamma = 10 ** (np.log10(LR_LAST / LR_INIT) / (epochs - 1))
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.MSELoss()

    optim_G = torch.optim.Adam(generator.parameters(),lr=learning_rate,betas=(beta1,beta2))
    optim_E = torch.optim.Adam(encoder.parameters(),lr=learning_rate,betas=(beta1,beta2))
    optim_D = torch.optim.Adam(discriminator.parameters(),lr=learning_rate,betas=(beta1,beta2))
    optimizer = optim.SGD(model.parameters(), LR_INIT, MOMENTUM, weight_decay=WEIGHT_DECAY)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1, gamma=gamma)

    best_acc = 0.0
    patience = 50
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_eval_losses = []
    avg_train_acc = []
    avg_eval_acc = []
    Dloss_lst = []
    Gloss_lst = []
    Eloss_lst = []
    for e in range(epochs):
        print('epoch {}'.format(e + 1))
        epoch_start_time = time.time()
        # topics = torch.zeros((32,3))
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        topic1 = torch.zeros((1,11))
        train_loss_D = 0.0
        train_loss_G = 0.0
        train_loss_E = 0.0
        lr_scheduler.step()
        model.train()
        generator.train()
        encoder.train()
        discriminator.train()
        for iter_num, (spec,spec1, label)in enumerate(trainsetloader):
            spec,spec1,label = spec.to(DEVICE), spec1.to(DEVICE),label.to(DEVICE,dtype = torch.long)#将数据搬到cuda上
            spec1 = spec1.type(torch.cuda.FloatTensor)
            print(spec1.shape)
            # label =torch.LongTensor(label)
            optim_D.zero_grad()
            optim_G.zero_grad()
            optim_E.zero_grad()
            theta_fake = torch.from_numpy(np.random.dirichlet(alpha=1.0*np.ones(11)/11,size=(len(spec1)))).float().to(DEVICE)
            spec1.squeeze_(dim=1)
            loss_D = -1.0*torch.mean(discriminator(encoder(spec1).detach())) + torch.mean(discriminator(generator(theta_fake).detach()))
            # loss_D.backward(retain_graph=True)
            # optim_D.step()
            # optim_G.zero_grad()
            loss_G = -1.0*torch.mean(discriminator(generator(theta_fake)))
            # loss_G.backward(retain_graph=True)
            # optim_G.step()
            # optim_E.zero_grad()
            topic = encoder(spec1)
            topic1 = topic[:,0:11]
            loss_E = torch.mean(discriminator(topic))
            # loss_E.backward(retain_graph=True)
            # optim_E.step()
            optimizer.zero_grad()
            # Dloss_lst.append(loss_D.item())
            # Gloss_lst.append(loss_G.item())
            # Eloss_lst.append(loss_E.item())

            y1,y2 = model(spec)#预测
            loss1 = criterion1(y1,label)
            loss2 = criterion2(y2,topic1)
            loss = loss1 + 0.1*loss2 + 0.1*loss_D+ 0.1*loss_G+ 0.1*loss_E
            loss.backward()#误差的反向传播
            optim_D.step()
            optim_G.step()
            optim_E.step()
            optimizer.step()#参数更新
            pred = torch.max(y1, 1)[1]
            train_correct = (pred == label).sum()
            train_acc += train_correct.item()
            train_loss += loss.item()
            train_loss_D += loss_D.item()
            train_loss_G += loss_G.item()
            train_loss_E += loss_E.item()
            # print(f'Epoch {(e+1):>3d}\tIter {(iter_num+1):>4d}\tLoss_D:{loss_D.item():<.7f}\tLoss_G:{loss_G.item():<.7f}\tloss_E:{loss_E.item():<.7f}')
        print('Train Loss: {:.6f}, Acc: {:.6f}, Dloss: {:.6f}, Eloss: {:.6f}, Gloss: {:.6f}'.format(train_loss / (len(trainset)), train_acc / (len(trainset)),train_loss_D/ (len(trainset)),train_loss_G/ (len(trainset)),train_loss_E/ (len(trainset))))
        with torch.no_grad():
            generator.eval()
            encoder.eval()
            discriminator.eval()
            model.eval()
            for _, (spec,spec1, label) in enumerate(testsetloader):
                spec,spec1,label = spec.to(DEVICE), spec1.to(DEVICE),label.to(DEVICE,dtype = torch.long)
                spec1 = spec1.type(torch.cuda.FloatTensor)
                topic = encoder(spec1)
                topic1 = topic[:,0:11]
                y1,y2 = model(spec)
                loss1 = criterion1(y1,label)
                loss2 = criterion2(y2,topic1)
                loss = 0.8*loss1 + 0.2*loss2
                pred = torch.max(y1, 1)[1]
                num_correct = (pred == label).sum()
                val_acc += num_correct.item()
                val_loss += loss.item()
                topic1 = torch.mean(topic1,dim=0)
                # topics += topic1.item()
            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(val_loss / (len(testset)), val_acc / (len(testset))))
            print('topic:',topic1)
        avgtrain_acc = np.average(train_acc)
        avgeval_acc = np.average(val_acc)
        avg_train_acc.append(avgtrain_acc)
        avg_eval_acc.append(avgeval_acc)

        early_stopping(avgeval_acc, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    torch.save(model.state_dict(), 'class10.pkl')
    print('Model Saved!')

