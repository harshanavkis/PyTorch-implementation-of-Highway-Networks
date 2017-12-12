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
			We create a dictionary of layers
		"""
		super(FcNet,self).__init__()
		# self.layers = {}
		self.numLayers = numLayers
		for i in range(numLayers):
			w = "layer"+str(i+1)
			vars(self)[w] = torch.nn.Linear(dimArr[i],dimArr[i+1])

	def forward(self,x):
		"""
			Accept an input variable and produce the output
		"""
		numLayers = self.numLayers
		for i in range(numLayers):
			w = "layer"+str(i+1)
			x = F.relu(vars(self)[w](x))
		w = "layer"+str(numLayers-1)
		return vars(self)[w](x)