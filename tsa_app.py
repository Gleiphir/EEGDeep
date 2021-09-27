import random

from LSTM_AE import ENDE_Trainsys,batch_size,input_len,AVR,RANGE
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import csv
from matplotlib import pyplot as plt


def preprocess(raw_batch:torch.Tensor):
    return (( torch.clamp(raw_batch,min=AVR- RANGE,max=AVR + RANGE) -AVR) / RANGE ) * 0.5 + 0.5

class CustomCSVDset(Dataset):
    def __init__(self, filename):
        fp = open(filename)
        self.data = []
        reader = csv.reader(fp)
        next(reader)
        next(reader)
        for row in reader:
            self.data.append([ float(f) for f in row[3:8]])
        print("{} : data length:{}".format(filename,len(self.data)))
        fp.close()

    def __len__(self):
        return len(self.data) - input_len

    def __getitem__(self, index):
        return torch.Tensor(self.data[index:index + input_len])





"""
s = ENDE_Trainsys()
fake =  torch.rand(batch_size,input_len,5)
print(fake.max())
fake.requires_grad = True
s.train_epoch(fake)
print(s.Encode(torch.rand(1,input_len,5)))
"""
num_epoch = 50


if __name__ == '__main__':
    training_data = CustomCSVDset("0826insight.csv")
    print(len(training_data))
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0)
    lossval = 0.0
    loss_seq = []
    trainSys = ENDE_Trainsys()
    for epoch in range(num_epoch):
        for idx, batch in enumerate(train_dataloader):
            #print(idx,end=" ")
            loss_seq.append( trainSys.train_epoch(batch) )
            if idx %200 == 0:
                loss, enc, fake = trainSys.test(batch)
                print("ENC: {} ~ {}, FAKE : {} ~{}".format(enc.min().item(),enc.max().item(),fake.min().item(),fake.max().item()))
                #print()
        if epoch %50 == 1:
            trainSys.save()

        print("#ep{}".format(epoch))


    randx = random.randint(0,len(train_dataloader))

    plt.plot(loss_seq)
    plt.show()
