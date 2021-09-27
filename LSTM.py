import torch
import torch.nn as nn

num_layer = 2
input_len = 7
in_channel = 10
out_channel = 20
batch_size = 5

#5 7 10 -> 5 7 20

rnn = nn.LSTM(in_channel, out_channel, num_layer)
input = torch.randn(batch_size, input_len, in_channel)
h0 = torch.randn(num_layer, input_len, out_channel)
c0 = torch.randn(num_layer, input_len, out_channel)
output, (hn, cn) = rnn(input, (h0, c0))

print(output.size(),hn.size(),cn.size())