import numpy
from torchviz import make_dot
from datetime import datetime
import torch
import torch.fft
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import weight_norm


num_layer_en = 1
num_layer_de = 4
input_len = 36
in_channel = 5

lstm_out_channel = 8
maxpool_factor =4
conv_window = 5

mid_channel = 8
fc_channel = 64
bottleneck_channel = 16
batch_size = 16

bottleneck_size = 32

AVR = 4200.0
RANGE = 1500.0

def preprocess(t:torch.Tensor):
    return (t - AVR ) / RANGE


def belt_filter(sig:torch.Tensor,Low:float,High:float):
    batch,length,channel = sig.shape[0],sig.shape[1],sig.shape[-1]
    #print(batch,channel)
    batch_axis =  []
    for i in range(batch):
        channel_axis = []
        for j in range(channel):
            FFT_res = torch.fft.fft(sig[i, :, j])
            belt = FFT_res[ torch.logical_and(torch.abs(FFT_res) >= Low, torch.abs(FFT_res) <= High)]
            channel_axis.append(torch.fft.ifft(belt,n=length))
        #print([t.shape for t in channel_axis])
        batch_axis.append(torch.stack(channel_axis,dim=1))
    #print([ t.shape for t in batch_axis])
    return torch.stack(batch_axis,dim=0).real
"""
rnn = nn.LSTM(in_channel, out_channel, num_layer)
input = torch.randn(batch_size, input_len, in_channel)
h0 = torch.randn(num_layer, input_len, out_channel)
c0 = torch.randn(num_layer, input_len, out_channel)
output, (hn, cn) = rnn(input, (h0, c0))
"""

class LSTMEncoder(nn.Module):
    """
    Batch x

    5 * 36 ->  8 * 36-> 1 * 128 -> 1 * 16

    LSTMLayer -> fulconn ->
    """
    def __init__(self):
        super().__init__()
        self.LSTMLayer = nn.LSTM(in_channel, mid_channel, num_layer_en)
        self.actv1 = nn.ReLU()
        self.conv1 = weight_norm(nn.Conv1d(lstm_out_channel,lstm_out_channel,(conv_window,)))
        self.pool1 = nn.MaxPool1d(maxpool_factor)
        # batch x ((input_len - conv_window +1 ) / maxpool_factor)
        self.L1 = weight_norm(nn.Linear( lstm_out_channel * ((input_len - conv_window +1 ) // maxpool_factor) , bottleneck_channel))
        # 2 * 36 = 72 -> 64
        self.actv2 = nn.Sigmoid()



    def forward(self,x):
        output, (hn, cn) = self.LSTMLayer(x)
        #print(output.shape)
        p = self.actv1(output.transpose(1,2))
        p = self.conv1(p)
        #print(p.shape)
        p = self.pool1(p)
        #print(p.shape)
        p = self.actv1(p).view(-1,1,lstm_out_channel * ((input_len - conv_window +1 ) // maxpool_factor) )
        #print(p.shape)
        p = self.L1(p)
        #print(p.shape)
        p = self.actv2(p)
        #print(p.shape)
        #print("done")
        return p

class LSTMDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        #self.fulconn = nn.Linear()
        self.LSTMLayer = nn.LSTM(bottleneck_channel, in_channel, num_layer_de)

    def forward(self,x):
        p = x
        output, (hn, cn) = self.LSTMLayer(p)
        return output

class ENDE(nn.Module):
    def __init__(self):
        super().__init__()
        self.en = LSTMEncoder().cuda()
        self.de = LSTMDecoder().cuda()

    def forward(self,x):
        inter = self.en(x)
        interSeq = inter.expand([-1,input_len,-1])
        #print(interSeq)
        regen = self.de(interSeq)
        return regen,inter


class ENDE_Trainsys:
    def __init__(self):
        self.ende = ENDE()

        self.optEnc = optim.Adam(self.ende.en.parameters(),lr=1e-5)
        self.optDec = optim.Adam(self.ende.de.parameters(), lr=1e-5)

        self.loss = nn.BCELoss()

    def train_epoch(self,data_batch:torch.Tensor):
        norm_batch = data_batch.cuda()
        self.optEnc.zero_grad()
        self.optDec.zero_grad()
        regen, encoded = self.ende(norm_batch)

        belt_batch = belt_filter(norm_batch,0.5,40)
        #print("regen")
        #print(regen.shape)
        loss = self.loss(belt_batch,regen).mean()

        loss.backward()

        self.optEnc.step()
        self.optDec.step()

        return loss.item()

    def test(self,test_batch:torch.Tensor):
        data_batch = preprocess(test_batch.cuda())
        regen, encoded = self.ende(data_batch)
        loss = self.loss(data_batch, regen)
        return loss.item(), encoded, regen

    def Encode(self,feed_batch:torch.Tensor):
        feed = preprocess(torch.Tensor(feed_batch).cuda())
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
