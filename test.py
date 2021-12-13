from train import build_network
import torch
from torchsummary import summary



net , _= build_network(None, 'resnet18')



# summary(net, (3, 224, 224))

img = torch.rand((1,3,224,224))

net(img)


