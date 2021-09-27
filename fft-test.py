import torch
import torch.fft
import cupy as np
from matplotlib import pyplot as plt

sig =  torch.rand(117,32,5)
sig_cp = np.random.rand(128,5)
FFT = np.fft.fft(sig_cp,axis=1)

def belt_filter(sig:torch.Tensor,Low:float,High:float):
    batch,length,channel = sig.shape[0],sig.shape[1],sig.shape[-1]
    print(batch,channel)
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
    return torch.stack(batch_axis,dim=0)

fig, axs = plt.subplots(4,1)

print(belt_filter(sig,1.0,10.0).shape)

axs[0].plot(sig)
FFTres = torch.fft.fft(sig)

#mask = torch.logical_and( torch.abs(FFTres) < 40.0 , torch.abs(FFTres) > 0.5 )
mask_low = torch.abs(FFTres) <= 10.0 # low Freq
mask_high = torch.abs(FFTres) > 2.0 # High freq

#band = FFTres[mask]

#band = band.real

high = FFTres[mask_high].real
low = FFTres[mask_low].real



#belt = FFTres[ torch.logical_and(mask_low,mask_high)   ].real

#belt = FFTres[ torch.logical_and(torch.abs(FFTres) > 5.0,torch.abs(FFTres) <= 10.0)    ].real

axs[1].plot(torch.fft.ifft(high))
axs[2].plot(torch.fft.ifft(low))
#axs[3].plot(torch.fft.ifft(belt))

plt.show()