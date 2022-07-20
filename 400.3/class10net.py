from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import warnings
warnings.filterwarnings('ignore')
# class ResidualBlock(nn.Module):
#     def __init__(self, inchannel, outchannel, stride):
#         super(ResidualBlock, self).__init__()
#         self.left = nn.Sequential(
#             nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(outchannel,),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(outchannel)
#         )
#         self.shortcut = nn.Sequential()
#         if stride != 1 or inchannel != outchannel:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(outchannel)
#             )
#
#     def forward(self, x):
#         out = self.left(x)
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
#
# class Dense(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(Dense, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, num_classes)
#
#     def forward(self, x):
#
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#
#         return out
class CNN(nn.Module):
    def __init__(self, inchannel, outchannel,kernel_size, stride,padding):
        super(CNN, self).__init__()
        self.cnn = nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.cnn(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class classifier(nn.Module):
    def __init__(self):
        super(classifier,self).__init__()

        self.cnn1 = nn.Sequential(CNN(1,64,3,1,1))
        self.cnn2 = nn.Sequential(CNN(64,64,3,1,1))
        self.maxpool1 = nn.MaxPool2d((1,2))
        self.cnn3 = nn.Sequential(CNN(64, 128,3,1,1))
        self.cnn4 = nn.Sequential(CNN(128, 128,3,1,1))
        self.maxpool2 = nn.MaxPool2d((1,2))
        self.cnn5 = nn.Sequential(CNN(128, 256,3,1, 1))
        self.dropout1 = nn.Dropout2d(0.3)
        self.cnn6 = nn.Sequential(CNN(256, 256,3,1, 1))
        self.maxpool3 = nn.MaxPool2d((1,2))
        self.cnn7 = nn.Sequential(CNN(256, 512, 3, 1, 1))
        self.dropout2 = nn.Dropout2d(0.3)
        self.cnn8 = nn.Sequential(CNN(512, 512, 3, 1, 1))
        self.dropout3 = nn.Dropout2d(0.3)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512 , 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.softmax=nn.Softmax(dim=1)

        self.fc3 = nn.Linear(512 , 1024)
        self.fc4 = nn.Linear(1024, 3)
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.maxpool1(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.maxpool2(x)
        x = self.cnn5(x)
        x = self.dropout1(x)
        x = self.cnn6(x)
        x = self.maxpool3(x)
        x = self.cnn7(x)
        x = self.dropout2(x)
        x = self.cnn8(x)
        x = self.dropout3(x)
        # x = self.maxpool4(x)
        # x = x.permute(0,2,1,3)
        x= self.gap(x)
        x = x.view(-1, 512)

        x1 = self.fc1(x)
        x1 = self.fc2(x1)
        x1 = self.softmax(x1)

        x2 = self.fc3(x)
        x2 = self.fc4(x2)
        x2 = self.sigmoid(x2)
        return x1,x2

if __name__ == '__main__':
    model = classifier()
    model.train()
    # print(model)
    input = torch.randn(4, 1, 128, 400)
    y1,y2 = model(input)
    print(y1.size())
    # print(y1)
    print(y2.size())
    # print(y2)
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
