import HighwayNet
import torch
import torch.nn as nn


class FcModel(nn.Module):
	def __init__(self, input_size,output_size, numLayers, hiddenDimArr, activation = 'ReLU'):
		super(FcModel,self).__init__()
		self.linears = nn.ModuleList([HighwayNet.FcNet(input_size,hiddenDimArr[0],activation)])
		self.linears.extend([HighwayNet.FcNet(dimArr[i-1],dimArr[i],activation) for i in range(1,numLayers-1)])
		self.append(nn.Linear(hiddenDimArr[-1],output_size))

	def forward(self,x):
		for l in linears:
			x = l(x)
		x = nn.Softmax(x)
		return x

class HighwayFcModel(nn.Module):
	def __init__(self, input_size, output_size, numLayers, activation='ReLU', gate_activation='Sigmoid', bias = -1.0):
		super(HighwayFcModel,self).__init__()
		self.highways = nn.ModuleList([HighwayNet.HighwayFcNet(input_size,numLayers,activation,gate_activation) for _ in range(numLayers)])
		self.linear = nn.Linear(input_size,output_size)

	def forward(self,x):
		for h in self.highways:
			x = h(x)
		x = nn.Softmax(self.linear(x))
		return x 
