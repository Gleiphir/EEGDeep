import random

from LSTM_AE_pure_conv import ENDE_Trainsys,batch_size,input_len,AVR,RANGE
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import csv
from matplotlib import pyplot as plt


def preprocess(raw_batch:torch.Tensor):
    return (( torch.clamp(raw_batch,min=AVR- RANGE,max=AVR + RANGE) -AVR) / RANGE ) * 0.5+ 0.5

class CustomCSVDset(Dataset):
    def __init__(self, filename):
        fp = open(filename)
        self.data = []
        reader = csv.reader(fp)
        next(reader)
        next(reader)
        for row in reader:
            self.data.append([ float(f) for f in row[3:8]])
        fp.close()

    def __len__(self):
        return len(self.data) - input_len

    def __getitem__(self, idx):
        return torch.Tensor(self.data[idx:idx+input_len])

training_data = CustomCSVDset("0826insight_trimmed.csv")

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True,num_workers=2)



"""
s = ENDE_Trainsys()
fake =  torch.rand(batch_size,input_len,5)
print(fake.max())
fake.requires_grad = True
s.train_epoch(fake)
print(s.Encode(torch.rand(1,input_len,5)))
"""
num_epoch = 1500


if __name__ == '__main__':
    lossval = 0.0
    loss_seq = []
    testSys = ENDE_Trainsys()
    testSys.load("20210831-135656Enc.pth","20210831-135656Dec.pth")
    randx = random.randint(0, len(train_dataloader))
    for idx, batch in enumerate(train_dataloader):
        if idx < randx: continue
        normed_batch = preprocess(batch)
        loss, encoded, regen = testSys.test(normed_batch)
        encoded = encoded.cpu().detach().numpy()
        regen = regen.cpu().detach().numpy()
        print(regen)
        print(encoded.shape)
        randx = random.randint(0, batch_size)
        for i in range(5):
            #plt.plot(normed_batch[randx,:,i],'red')
            plt.plot(regen[randx,:,i],'blue')
        plt.show()
        input()


    #plt.plot(loss_seq)
