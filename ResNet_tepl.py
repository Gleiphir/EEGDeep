import torch
import torch.nn as nn
from collections import OrderedDict




def nameBy(m:nn.Module):
    return type(m).__name__

class ResBlockCNN(nn.Module):
    def __init__(self, *layers, upsample_factor=1,upmethod ='bilinear',downsample_layer=None,act = None):
        super().__init__()
        od = OrderedDict()
        if upsample_factor != 1:
            us = nn.Upsample(upsample_factor,mode=upmethod)
            od["Upsample-{}".format(nameBy(us))] = us

        for i,l in enumerate(layers):
            od["{}#{}".format(i+1,nameBy(l))] = l

        if downsample_layer:
            od["Downsample-{}".format(nameBy(downsample_layer))] = downsample_layer

        self.od =od
        print(od)

        self.seq = nn.Sequential(od)

        if act ==None:
            self.act = act
        else:
            self.act = nn.ReLU()

    def forward(self,x):
        p = x
        return self.act(x + self.seq(p))



if __name__ == '__main__':
    v = torch.Tensor(16,3,5,13)
    resb = ResBlockCNN( nn.Conv2d(3, 5, 3) )
    print(resb(v))