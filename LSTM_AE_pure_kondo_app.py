import random

import numpy

from LSTM_AE_pure_conv import ENDE_Trainsys,batch_size,input_len,AVR,RANGE
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import csv
from matplotlib import pyplot as plt


def preprocess(raw_batch:torch.Tensor):
    return (( torch.clamp(raw_batch,min=AVR- RANGE,max=AVR + RANGE) -AVR) / RANGE ) * 0.5 + 0.5

def DiffProcess(raw_batch:torch.Tensor):
    return raw_batch / RANGE + 0.5

# class CustomCSVDset(Dataset):
#     def __init__(self, filename):
#         fp = open(filename)
#         self.data = []
#         reader = csv.reader(fp)
#         next(reader)
#         next(reader)
#         for row in reader:
#             self.data.append([ float(f) for f in row[3:8]])
#         print("{} : data length:{}".format(filename,len(self.data)))
#         fp.close()

#     def __len__(self):
#         return len(self.data) - input_len

#     def __getitem__(self, index):
#         return preprocess(torch.Tensor(self.data[index:index + input_len]))

# CHANGED
import numpy as np

class CustomCSVDset(Dataset):
    def __init__(self, filename):
        fp = open(filename)
        self.data = []
        reader = csv.reader(fp)
        next(reader)
        next(reader)
        for row in reader:
            self.data.append([ float(f) for f in row[3:8]])
        self.data = np.array(self.data, dtype=np.float32)
        print("{} : data shape:{}".format(filename,self.data.shape))

        #for i in range(5):
            #self.data[:,i] = (self.data[:,i] - self.data[:,i].mean()) / self.data[:,i].std()

        fp.close()

    def __len__(self):
        return len(self.data) - input_len

    def __getitem__(self, index):
        return preprocess(torch.Tensor(self.data[index:index + input_len]))
        #return self.data[index:index + input_len]


class CustomCSVDiffDset(Dataset):
    def __init__(self, filename):
        fp = open(filename)
        raw_data = []
        reader = csv.reader(fp)
        next(reader)
        next(reader)
        for row in reader:
            raw_data.append([ float(f) for f in row[3:8]])
        print("{} : data length:{}".format(filename,len(raw_data)))
        fp.close()
        npdata = numpy.array(raw_data)
        self.data = npdata[1:] - npdata[:-1]


    def __len__(self):
        return len(self.data) - input_len

    def __getitem__(self, index):
        return DiffProcess(torch.Tensor(self.data[index:index + input_len]))

#training_data = CustomCSVDiffDset("0826insight.csv")



"""
s = ENDE_Trainsys()
fake =  torch.rand(batch_size,input_len,5)
print(fake.max())
fake.requires_grad = True
s.train_epoch(fake)
print(s.Encode(torch.rand(1,input_len,5)))
"""
num_epoch = 200


if __name__ == '__main__':
    plt.ion()
    training_data = CustomCSVDset("0826insight.csv")
    print(len(training_data))
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
    lossval = 0.0
    loss_seq = []
    trainSys = ENDE_Trainsys()
    for epoch in range(num_epoch):
        for idx, batch in enumerate(train_dataloader):
            norm_batch = batch
            #print(norm_batch.max(),norm_batch.min())
            loss_seq.append( trainSys.train_epoch(norm_batch) )
            if idx %500 == 0:
                loss, enc, fake = trainSys.test(norm_batch)
                #print(norm_batch.size(),fake.size())
                print("LOSS: {:.6f}, ENC: {:.6f} ~ {:.6f}, R{:.6f}~{:.6f} F{:.6f} ~{:.6f}".format(loss,enc.min().item(),enc.max().item(),norm_batch.min().item(),norm_batch.max().item(),fake.min().item(),fake.max().item()))
                #plt.cla()
                #plt.ylim([0.0, 1.0])
                #plt.plot(norm_batch.cpu().detach().numpy()[0, :, 0],color='black')
                #plt.plot(fake.cpu().detach().numpy()[0, :, 0],color='orange')
                #plt.draw()
                #plt.pause(0.2)
        if epoch %50 == 1:
            trainSys.save()

        print("#ep{}".format(epoch))


    randx = random.randint(0,len(train_dataloader))

    plt.plot(loss_seq)
    plt.show()
