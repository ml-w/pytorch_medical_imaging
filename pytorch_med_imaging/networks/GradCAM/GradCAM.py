from .NetWrapper import NetWrapper
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class GradCam:
	def __init__(self, model, target_layer_names, threeD=True):
		self.model = model
		self.model.eval()
		self.cuda = next(self.model.parameters()).is_cuda
		if self.cuda:
			self.model = model.cuda()
		self.threeD = threeD

		self.extractor = NetWrapper(self.model, target_layer_names)

	def __call__(self, input, index = None):
		if self.cuda:
			output = self.extractor(input.cuda())
		else:
			output = self.extractor(input)

		decisions = torch.argmax(output, dim=1)
		one_hot = torch.zeros_like(output)
		if self.cuda:
			one_hot = one_hot.cuda()

		# Back-propagation with respect to decisions
		for index, d in enumerate(decisions):
			one_hot[index, d] = 1

		# Decision 1 backpropagation
		# one_hot[:,1] = 1

		one_hot = Variable(one_hot, requires_grad=True)
		one_hot = one_hot * output
		one_hot.sum().backward()
		# output.sum().backward()

		features, grad = self.extractor.get_features_grad()

		# TODO: Make this work for multiGPU, hint: use partial to pass name for hook functions
		# Because multi-gpu setting cause the mods to load multiple times, this is needed.
		# features = [torch.cat(features[f]) for f in features]
		# grad = [torch.cat(grad[f]) for f in grad]

		if self.threeD:
			while input.dim() < 5:
				input = input.unsqueeze(0)
			g = grad[0][0]
			while g.dim() < 5:
				g = g.unsqueeze(0)
			weight = torch.mean(g, dim=(2, 3, 4))
		elif not self.threeD:
			while input.dim() < 4:
				input = input.unsuqeeze(0)
			g = grad[0][0]
			while g.dim() < 4:
				g = g.unsqueeze(0)
			weight = torch.mean(g, dim=(2, 3))
		cam = torch.zeros_like(input)


		ff = features[0][0]
		while ff.dim() < cam.dim():
			ff = ff.unsqueeze(0)
		# Batch dimension
		for b in range(input.shape[0]):
			# Channel dimension
			for c, g in enumerate(weight[b]):
				f = ff[b][c]
				while f.dim() < cam.dim():
					f = f.unsqueeze(0)
				f = F.interpolate(F.relu(f), cam[b].squeeze().shape, mode='trilinear',
								  align_corners=True)
				cam[b] += (f * g).view_as(cam[b])
		cam = F.relu(cam)
		return output.cpu(), decisions.cpu(), cam.detach().cpu()
