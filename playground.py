import torch
from torch.autograd import Variable
import torch.nn as nn
import HighwayNet as HN
import models

#imports to load datasets
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim

#plot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


if __name__== '__main__':
	download = False
	#load mnist data
	batch_size = 1024
	transform = transforms.Compose([transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
	trainset = torchvision.datasets.MNIST(root = "../data",train=True, 
													download=download, transform=transform)
	trainLoader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
													shuffle=True, num_workers=1)
	testset = torchvision.datasets.MNIST(root="../data", train=False, 
													download=download, transform=transform)
	testLoader = torch.utils.data.DataLoader(testset, batch_size=4,
													shuffle=False, num_workers=1) 

	#user specified data
	s = input('Enter the number of activation units in each layer(Plain FC net):')
	s = s.split(" ")
	hiddenDimArr = [int(i) for i in s]

	s = input('Enter the number of units in each Highway FC layer(Enter a single number):')
	sizeHfc = int(s)

	numLayers = len(hiddenDimArr) #making number of hidden layers in both nets equal
	#activationFunc = input('Enter the activation function(Refer to activations.py for more details):')


	output_size = 10

	#create the FC and HFC nets
	fcNet = models.FcModel(784,10,numLayers,hiddenDimArr)
	HfcNet = models.HighwayFcModel(784,sizeHfc,10,numLayers,bias=-3.0)

	#FC optimizers
	fcCriterion = nn.CrossEntropyLoss()
	optimizerFc = optim.SGD(fcNet.parameters(),lr=0.01,momentum=0.9)

	#Highway Net optimizers
	HfcNetCriterion = nn.CrossEntropyLoss()
	optimizerHfc = optim.SGD(HfcNet.parameters(),lr=0.01,momentum=0.9)

	#save data to file
	a = np.asarray([])

	#training
	lossFc = None
	lossHfc = None
	for epoch in range(500):
		for bid,(x,target) in enumerate(trainLoader):
			optimizerFc.zero_grad()
			optimizerHfc.zero_grad()
			x = x.view(-1,28*28)
			x,target = Variable(x), Variable(target)
			fcOut = fcNet.forward(x)
			HfcOut = HfcNet.forward(x)
			lossFc = fcCriterion(fcOut,target)
			lossHfc = HfcNetCriterion(HfcOut,target)
			lossFc.backward()
			lossHfc.backward()
			optimizerFc.step()
			optimizerHfc.step()
			if bid>0 and bid%10 == 0: #print loss every 1000 mini-batches
				print('-->epoch:{},batch index:{},FCloss:{:.3f},HFCloss:{:.3f}'.format(epoch+1,bid,lossFc.data[0],lossHfc.data[0]))
		if lossFc is not None and lossHfc is not None:
			np.append(a,[epoch,lossFc.data[0]/batch_size,lossHfc.data[0]/batch_size])

	#save the data to a csv file
	np.savetxt("{}Depth{}batch_size.csv".format(numLayers+1,batch_size),a,delimiter=',')