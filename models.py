import HighwayNet
import torch
import torch.nn as nn
import torch.nn.functional as F


class FcModel(nn.Module):
	def __init__(self, input_size,output_size, numLayers, hiddenDimArr, activation = 'ReLU'):
		super(FcModel,self).__init__()
		self.linears = nn.ModuleList([HighwayNet.FcNet(input_size,hiddenDimArr[0],activation)])
		self.linears.extend([HighwayNet.FcNet(hiddenDimArr[i-1],hiddenDimArr[i],activation) for i in range(1,numLayers)])
		self.linears.append(nn.Linear(hiddenDimArr[-1],output_size))

	def forward(self,x):
		for l in self.linears:
			x = l(x)
		x = F.softmax(x)
		return x

class HighwayFcModel(nn.Module):
	def __init__(self, inDims, input_size, output_size, numLayers, activation='ReLU', gate_activation='Sigmoid', bias = -1.0):
		super(HighwayFcModel,self).__init__()
		self.highways = nn.ModuleList([HighwayNet.HighwayFcNet(input_size,numLayers,activation,gate_activation) for _ in range(numLayers)])
		self.linear = nn.Linear(input_size,output_size)
		self.dimChange  = nn.Linear(inDims, input_size)

	def forward(self,x):
		x = F.relu(self.dimChange(x))
		for h in self.highways:
			x = h(x)
		x = F.softmax(self.linear(x))
		return x 
