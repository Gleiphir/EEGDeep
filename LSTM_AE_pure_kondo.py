import numpy
from torchviz import make_dot
from datetime import datetime
import torch
import torch.fft
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import weight_norm

num_layer = 4
# input_len = 18  # CHANGED
in_channel = 5

en_lstm_out_channel = 32
de_lstm_out_channel = 32
maxpool_factor =4
conv_window = 5

mid_channel = 8
fc_channel = 64
bottleneck_channel = 32

batch_size = 16


AVR = 4200.0
RANGE = 1500.0

assert bottleneck_channel > in_channel

def preprocess(raw_batch:torch.Tensor):
    return (( torch.clamp(raw_batch,min=AVR- RANGE,max=AVR + RANGE) -AVR) / RANGE ) * 0.5 + 0.5

"""
rnn = nn.LSTM(in_channel, out_channel, num_layer)
input = torch.randn(batch_size, input_len, in_channel)
h0 = torch.randn(num_layer, input_len, out_channel)
c0 = torch.randn(num_layer, input_len, out_channel)
output, (hn, cn) = rnn(input, (h0, c0))
"""

# class LSTMEncoder(nn.Module):
#     """
#     Batch x

#     5 * 36 ->  8 * 36-> 1 * 128 -> 1 * 16

#     LSTMLayer -> fulconn ->
#     """
#     def __init__(self):
#         super().__init__()
#         self.LSTMLayer = nn.LSTM(in_channel, en_lstm_out_channel, num_layer,dropout=0.2)
#         self.actv1  = nn.ReLU()
#         self.dense = weight_norm(nn.Linear( en_lstm_out_channel * input_len  , bottleneck_channel))
#         self.actv2 = nn.Sigmoid()



#     def forward(self,x):
#         output, (hn, cn) = self.LSTMLayer(x)
#         p = self.actv1(output)
#         p = self.dense(p.view(-1,1,en_lstm_out_channel * input_len))
#         p = self.actv2(p)
#         #print(p.shape)
#         #print("done")
#         return p

# class LSTMDecoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.LSTMLayer1 = nn.LSTM(bottleneck_channel, bottleneck_channel, 2, dropout=0.2)
#         self.LSTMLayer2 = nn.LSTM(bottleneck_channel, de_lstm_out_channel, 2,dropout=0.2)
#         self.actv1 = nn.ReLU()
#         self.dense = weight_norm(nn.Linear(de_lstm_out_channel , in_channel))
#         self.actv2 = nn.Sigmoid()

#     def forward(self,x):
#         output, (hn, cn) = self.LSTMLayer1(x)
#         output = self.actv1(output)
#         output, (hn, cn) = self.LSTMLayer2(output)
#         p = self.actv1(output)
#         p = self.dense(p)
#         p = self.actv2(p)
#         # print(p.shape)
#         # print("done")
#         return p


# CHANGED
from torch.nn import functional as F

DIM=64
input_len = 32


class LSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(5  , DIM, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(DIM, DIM, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(DIM, DIM, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(4*DIM, bottleneck_channel)
        self.ln1 = nn.GroupNorm(1, DIM)
        self.ln2 = nn.GroupNorm(1, DIM)
        self.ln3 = nn.GroupNorm(1, DIM)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.ln1(self.conv1(x)))
        x = F.relu(self.ln2(self.conv2(x)))
        x = F.relu(self.ln3(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class LSTMDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(bottleneck_channel, 4*DIM)
        self.conv1 = nn.ConvTranspose1d(DIM, DIM, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose1d(DIM, DIM, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose1d(DIM, 5  , kernel_size=4, stride=2, padding=1)
        self.ln1 = nn.GroupNorm(1, DIM)
        self.ln2 = nn.GroupNorm(1, DIM)
        self.ln3 = nn.GroupNorm(1, DIM)

    def forward(self, x):  # BxDIM
        x = self.fc1(x)
        x = x.view(-1, DIM, 4)               # BxDIMx4
        x = F.relu(self.ln1(self.conv1(x)))  # BxDIMx8
        x = F.relu(self.ln2(self.conv2(x)))  # BxDIMx16
        x = self.conv3(x)  # BxDIMx32
        return x.transpose(1,2)


class ENDE(nn.Module):
    def __init__(self):
        super().__init__()
        self.en = LSTMEncoder().cuda()
        self.de = LSTMDecoder().cuda()

    def forward(self,x):
        inter = self.en(x)
        # interSeq = inter.expand([-1,input_len,-1])  # CHANGED
        #print(interSeq.requires_grad)
        #print(interSeq.shape)
        # regen = self.de(interSeq)
        regen = self.de(inter)
        return regen,inter


class ENDE_Trainsys:
    def __init__(self):
        self.ende = ENDE()

        #self.optEnc = optim.Adam(self.ende.en.parameters(),lr=2e-5)
        #self.optDec = optim.Adam(self.ende.de.parameters(), lr=2e-5)
        # self.opt = optim.Adam(self.ende.parameters(), lr=2e-4)
        self.opt = optim.Adam(self.ende.parameters())  # CHANGED
        # self.loss = nn.BCELoss()
        self.loss = nn.MSELoss()  # CHANGED

    def train_epoch(self,data_batch:torch.Tensor):
        norm_batch = data_batch.cuda()
        #self.optEnc.zero_grad()
        #self.optDec.zero_grad()
        self.opt.zero_grad()
        regen, encoded = self.ende(norm_batch)

        #belt_batch = belt_filter(norm_batch, 0.5, 40)
        # print("regen")
        # print(regen.shape)
        loss = self.loss(regen,norm_batch.detach())

        loss.backward()

        #self.optEnc.step()
        #self.optDec.step()
        self.opt.step()

        return loss.item()

    def test(self,test_batch:torch.Tensor):
        data_batch = test_batch.cuda()
        regen, encoded = self.ende(data_batch)
        loss = self.loss(regen,data_batch.detach())
        return loss.item(), encoded, regen

    def Encode(self,feed_batch:torch.Tensor):
        feed = torch.Tensor(feed_batch).cuda()
        regen, encoded = self.ende(feed)
        return encoded

    def save(self):
        t = datetime.now()
        fileTimeStamp = t.strftime("%Y%m%d-%H%M%S")
        print(fileTimeStamp)
        torch.save(self.ende.en, fileTimeStamp + "Enc.pth")
        torch.save(self.ende.de, fileTimeStamp + "Dec.pth")
        print("saved")

    def load(self,enc:str,dec:str):
        self.ende.en = torch.load(enc)
        self.ende.de = torch.load(dec)

if __name__ =="__main__":
    s = ENDE_Trainsys()

    input_v = torch.rand(1, input_len, 5)
    print(s.Encode(input_v))
    fake =  torch.rand(batch_size,input_len,5)
    print(fake.min(),fake.max())
    fake.requires_grad = True
    s.train_epoch(fake)
    print(s.Encode(input_v))
