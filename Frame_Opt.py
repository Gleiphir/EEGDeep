import timeit
import cupy as np
from LSTM_AE import input_len,bottleneck_channel
from polygon_cu import create_polygon,draw_trangle

WIDTH = 1000
HEIGHT = 1000

LOW_BUNDER = 0.2
RANGE_FACTOR = 1.0 - LOW_BUNDER

out  = np.random.rand(bottleneck_channel) * RANGE_FACTOR + LOW_BUNDER
print(out)
center = np.array([WIDTH // 2, HEIGHT // 2 ])

Radius = 450.0
agl =  np.linspace(2 * np.pi, 0.0, num= bottleneck_channel + 1)[0:bottleneck_channel] + 0.03 #clockwise

#inner = Radius * 0.2 * (np.linspace(2 * np.pi, 0.0, num= bottleneck_channel + 1)[0:bottleneck_channel] + 0.03 + np.pi / bottleneck_channel )

offsetx = np.cos(agl) * Radius
offsety = np.sin(agl) * Radius


#print(pos)
#data = create_polygon([1920,1080],pos)




def renderStar(input_seq):

    masks =[]
    pos = np.stack([offsetx * input_seq, offsety * input_seq]).T + center

    for i in range(bottleneck_channel):
        masks.append( draw_trangle([WIDTH,HEIGHT],pos[i-1],pos[i],center,color= (i%4)*25 + 150)   )

    #print(np.max(data))
    #data = np.any(np.array(masks),axis=0)
    data = np.max(np.array(masks),axis=0)
    return data.T.get()



from matplotlib import pyplot as plt


plt.ion()
imIO = plt.imshow(renderStar(np.random.rand(bottleneck_channel) * 0.8 + 0.2))

while True:
    imIO.set_data(renderStar(np.random.rand(bottleneck_channel) * 0.8 + 0.2))
    plt.pause(0.25)

