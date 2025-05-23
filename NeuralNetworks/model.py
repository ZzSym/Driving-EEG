import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class EEG_MLP(nn.Module):
    def __init__(self, input_dim=63*200, hidden_dims=[512, 256, 64], dropout=0.3):
        super(EEG_MLP, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (B, 63, 200)
        x = x.view(x.size(0), -1)
        out = self.network(x)
        return out.squeeze(1) 

class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        
        # 输入: (B, 1, 63, 200)
        
        # Layer 1: 时域卷积
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 64), padding=(0, 32), bias=False)  # 保持时间维度
        self.batchnorm1 = nn.BatchNorm2d(16)

        # Layer 2: 空间卷积（跨通道）
        self.depthwiseConv = nn.Conv2d(16, 32, kernel_size=(63, 1), groups=16, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout2 = nn.Dropout(p=0.25)

        # Layer 3: 深度卷积
        self.separableConv = nn.Conv2d(32, 64, kernel_size=(1, 16), padding=(0, 8), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.pooling3 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout3 = nn.Dropout(p=0.25)

        # FC Layer: 输出一个回归值
        self.fc = nn.Linear(64 * ((200 // 4 // 8)), 1)

    def forward(self, x):
        # x: (B, 63, 200)
        x = x.unsqueeze(1)  # -> (B, 1, 63, 200)

        x = F.elu(self.batchnorm1(self.conv1(x)))       # -> (B, 16, 63, 200)
        x = F.elu(self.batchnorm2(self.depthwiseConv(x)))  # -> (B, 32, 1, 200)
        x = self.pooling2(x)                             # -> (B, 32, 1, 50)
        x = self.dropout2(x)

        x = F.elu(self.batchnorm3(self.separableConv(x)))  # -> (B, 64, 1, 50)
        x = self.pooling3(x)                             # -> (B, 64, 1, 6)
        x = self.dropout3(x)

        x = x.reshape(x.size(0), -1)                     # -> (B, 64*6)
        x = self.fc(x)                                   # -> (B, 1)
        # return torch.tanh(x)  # to [-1, 1]
        return x

class EEGNet_LSTM(nn.Module):
    def __init__(self, lstm_hidden_dim=32, lstm_layers=2, num_classes=1):
        super(EEGNet_LSTM, self).__init__()
        self.T = 200
        self.affine = True

        # EEGNet
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, affine=self.affine)

        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(16, 4, (63, 1))
        self.batchnorm2 = nn.BatchNorm2d(4, affine=self.affine)
        self.pooling2 = nn.MaxPool2d((1, 4))

        self.padding2 = nn.ZeroPad2d((2, 1, 0, 0))
        self.conv3 = nn.Conv2d(4, 4, (1, 8))
        self.batchnorm3 = nn.BatchNorm2d(4, affine=self.affine)
        self.pooling3 = nn.MaxPool2d((1, 8))

        self.dropout = nn.Dropout(0.0)

        # LSTM
        self.lstm_input_dim = 4 
        self.lstm = nn.LSTM(input_size=self.lstm_input_dim,
                            hidden_size=lstm_hidden_dim,
                            num_layers=lstm_layers,
                            batch_first=True)
        
        self.fc = nn.Linear(lstm_hidden_dim, num_classes)

    def forward(self, x):
        # -> (B, 63, 200)
        x = x.unsqueeze(1)  # (B, 1, 63, 200)

        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = self.dropout(x)

        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.pooling2(x)

        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = self.dropout(x)
        x = self.pooling3(x)  # (B, 4, 1, T')

        # reshape for LSTM: (B, T', Features)
        x = x.mean(dim=2)       # robust spatial flatten
        x = x.permute(0, 2, 1)  # -> (B, T', 4)

        # LSTM
        out, _ = self.lstm(x)  # (B, T', H)
        x = out[:, -1, :]      # -> (B, H)

        x = self.fc(x)         # (B, num_classes)
        return x

class EEGNetRegression(nn.Module):
    def __init__(self,
                 num_electrodes=63,
                 chunk_size=200,
                 F1=8,
                 D=2,
                 F2=16,
                 kernel_1=64,
                 kernel_2=16,
                 dropout=0.1):
        super(EEGNetRegression, self).__init__()
        
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_1), padding=(0, kernel_1 // 2), bias=False),
            nn.BatchNorm2d(F1)
        )
        
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (num_electrodes, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )
        
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, kernel_2), padding=(0, kernel_2 // 2), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )

        self._output_dim = self._get_flattened_size(num_electrodes, chunk_size)

        self.regressor = nn.Linear(self._output_dim, 1)

    def _get_flattened_size(self, num_electrodes, chunk_size):
        with torch.no_grad():
            x = torch.zeros(1, 1, num_electrodes, chunk_size)
            x = self.firstconv(x)
            x = self.depthwiseConv(x)
            x = self.separableConv(x)
            return x.shape[1] * x.shape[2] * x.shape[3]

    def forward(self, x):
        x = x.unsqueeze(1)  # 从 (B, C, T) → (B, 1, C, T)
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.size(0), -1)  # Flatten
        out = self.regressor(x)
        return out.squeeze(1)  # 输出为 (B,)

class EEGNetLSTMRegression(nn.Module):
    def __init__(self,
                 num_electrodes=63,
                 chunk_size=200,
                 F1=8,
                 D=2,
                 F2=16,
                 kernel_1=64,
                 kernel_2=16,
                 dropout=0.5,
                 lstm_hidden_dim=64,
                 lstm_layers=2):
        super(EEGNetLSTMRegression, self).__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_1), padding=(0, kernel_1 // 2), bias=False),
            nn.BatchNorm2d(F1)
        )
        
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (num_electrodes, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )
        
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, kernel_2), padding=(0, kernel_2 // 2), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )

        # 得到 Flatten 后的维度，用于构造 LSTM 输入
        with torch.no_grad():
            x = torch.zeros(1, 1, num_electrodes, chunk_size)
            x = self.firstconv(x)
            x = self.depthwiseConv(x)
            x = self.separableConv(x)
            _, C, H, W = x.shape
            self.lstm_input_dim = C * H
            self.seq_len = W

        # LSTM 层（batch_first=True）
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False
        )

        # 最终回归输出层
        self.regressor = nn.Linear(lstm_hidden_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 63, 200)
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)  # (B, F2, 1, W)
        
        x = x.squeeze(2)  # 去掉 H=1 → (B, F2, W)
        x = x.permute(0, 2, 1)  # (B, W, F2) → 作为 LSTM 输入
        # 如果 F2=16，W=时间轴，那么输入是 (B, seq_len=W, input_dim=F2)

        out, _ = self.lstm(x)  # out: (B, W, hidden_dim)
        last_out = out[:, -1, :]  # 取最后时间步的输出

        out = self.regressor(last_out)  # (B, 1)
        return out.squeeze(1)  # (B,)
    
class EEGNet_LSTM_2(nn.Module):
    # 增大fc.weight初始值, 激活函数改为ReLU或LeakyReLU, 在LSTM增加LayerNorm
    def __init__(self, lstm_hidden_dim=32, lstm_layers=2, num_classes=1):
        super(EEGNet_LSTM_2, self).__init__()
        self.T = 200
        self.affine = True
 
        # EEGNet
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, affine=self.affine)

        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(16, 4, (63, 1))
        self.batchnorm2 = nn.BatchNorm2d(4, affine=self.affine)
        self.pooling2 = nn.MaxPool2d((1, 4))

        self.padding2 = nn.ZeroPad2d((2, 1, 0, 0))
        self.conv3 = nn.Conv2d(4, 4, (1, 8))
        self.batchnorm3 = nn.BatchNorm2d(4, affine=self.affine)
        self.pooling3 = nn.MaxPool2d((1, 8))

        self.dropout = nn.Dropout(0.0)

        # LSTM
        self.lstm_input_dim = 4 
        self.lstm = nn.LSTM(input_size=self.lstm_input_dim,
                            hidden_size=lstm_hidden_dim,
                            num_layers=lstm_layers,
                            batch_first=True)
        
        # 增加LayerNorm
        self.norm_fc = nn.LayerNorm(lstm_hidden_dim)

        self.fc = nn.Linear(lstm_hidden_dim, num_classes)

        # 增大fc.weight初始值
        nn.init.kaiming_uniform_(self.fc.weight, a=0.01)
        self.fc.weight.data *= 10 

        # 更改为稳定的激活函数（ReLU或LeakyReLU）
        self.activation = nn.ReLU()
        # self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 63, 200)

        x = self.activation(self.conv1(x))
        x = self.batchnorm1(x)
        x = self.dropout(x)

        x = self.padding1(x)
        x = self.activation(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.pooling2(x)

        x = self.padding2(x)
        x = self.activation(self.conv3(x))
        x = self.batchnorm3(x)
        x = self.dropout(x)
        x = self.pooling3(x)  # (B, 4, 1, T')

        x = x.mean(dim=2)       # (B, 4, T')
        x = x.permute(0, 2, 1)  # (B, T', 4)

        out, _ = self.lstm(x)
        x = out[:, -1, :]       # (B, H)

        # 使用增加的LayerNorm
        x = self.norm_fc(x)
        x = self.fc(x)          # (B, num_classes)
        return x

class EEGNet_LSTM_3(nn.Module):
    ''' 减小参数量的版本
    '''
    def __init__(self, 
                 conv1_channels=32, 
                 conv1_kernel=64, 
                 lstm_hidden_dim=64, 
                 lstm_layers=1, 
                 dropout=0.25):
        super(EEGNet_LSTM_2, self).__init__()

        self.conv1 = nn.Conv2d(1, conv1_channels, (1, conv1_kernel), padding=(0, conv1_kernel // 2))
        self.bn1 = nn.BatchNorm2d(conv1_channels)

        self.depthwise_conv = nn.Conv2d(conv1_channels, conv1_channels, (63, 1), groups=conv1_channels)
        self.bn2 = nn.BatchNorm2d(conv1_channels)
        self.pool1 = nn.AvgPool2d((1, 4))

        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(input_size=conv1_channels,
                            hidden_size=lstm_hidden_dim,
                            num_layers=lstm_layers,
                            batch_first=True)

        self.fc = nn.Linear(lstm_hidden_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.depthwise_conv(x)))
        x = self.pool1(x)
        x = self.dropout(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        x = torch.mean(out, dim=1)
        x = self.fc(x)
        return torch.tanh(x)


