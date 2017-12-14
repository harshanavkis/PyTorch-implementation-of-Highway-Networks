import torch
from torch.autograd import Variable
import torch.nn as nn
import HighwayNet as HN

#imports to load datasets
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim

#user specified data
s = input('Enter the number of activation units in each layer(Plain FC net):')
s = s.split(" ")
dimArr = [int(i) for i in s]

s = input('Enter the number of units in each Highway FC layer(Enter a single number):')
sizeHfc = int(s)

numLayers = len(dimArr) #making number of layers in both nets equal
activationFunc = input('Enter the activation function(Refer to activations.py for more details):')

#load mnist data
transform = transforms.Compose([transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
trainset = torchvision.datasets.MNIST(root = "../data",train=True, 
												download=True, transform=transform)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=4,
												shuffle=True, num_workers=2)
testset = transforms.datasets.MNIST(root="../data", train=False, 
												download=True, transform=transform)
testLoader = torch.utils.data.DataLoader(testset, batch_size=4,
												shuffle=False, num_workers=2) 


#create the FC and HFC nets
fcNet = HN.FcNet(dimArr, numLayers, activationFunc)
HfcNet = HN.HighwayFcNet(sizeHfc, numLayers, activationFunc)

#loss func and optimizer
criterion = nn.CrossEntropyLoss()
optimizerFc = optim.SGD(fcNet.parameters())