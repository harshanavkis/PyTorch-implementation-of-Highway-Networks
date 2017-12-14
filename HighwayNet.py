import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class FcNet(nn.Module):
	"""
		A more robust fully connected network
	"""
	def __init__(self,dimArr,numLayers):
		"""
			We create a group of fc layers
		"""
		super(FcNet,self).__init__()
		# self.layers = {}
		self.numLayers = numLayers
		for i in range(numLayers):
			w = "layer" + str(i+1)
			vars(self)[w] = nn.Linear(dimArr[i],dimArr[i+1])

	def forward(self,x):
		"""
			Accept an input variable and produce the output
		"""
		numLayers = self.numLayers
		for i in range(numLayers-1):
			w = "layer"+str(i+1)
			x = F.relu(vars(self)[w](x))
		w = "layer"+str(numLayers)
		return vars(self)[w](x)


class HighwayFcNet(nn.Module):
	"""
		A more robust fully connected network
	"""
	def __init__(self, size, numLayers):
		"""
			We create a group of highway fc layers
		"""
		super(HighwayFcNet,self).__init__()
		self.numLayers = numLayers
		for i in range(numLayers):
			wh = "wh" + str(i+1)
			wt = "wt" + str(i+1)
			# wc = "wc" + str(i+1)
			vars(self)[wh] = nn.Linear(size,size)
			vars(self)[wt] = nn.Linear(size,size)
			# vars(self)[wc] = nn.sub(1.0,vars(self)[t])

	def forward(self,x):
		numLayers = self.numLayers
		for i in range(numLayers-1):
			wh = "wh" + str(i+1)
			wt = "wt" + str(i+1)
			# wc = "wc" + str(i+1)
			h = F.relu(vars(self)[wh](x))
			t = F.sigmoid(vars(self)[wt](x))
			c = 1.0 - t
			x = torch.add(torch.mul(h,t),torch.mul(x,c))
		wh = "wh" + str(numLayers)
		return vars(self)[wh](x)
		return x