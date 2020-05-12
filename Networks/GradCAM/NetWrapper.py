from .FeatureExtractor import FeatureExtractor

class NetWrapper(object):
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, net, target_layers):
		self.net = net
		self.feature_extractor = FeatureExtractor(self.net, target_layers)

	def get_features_grad(self):
		return self.feature_extractor.features, self.feature_extractor.gradients

	def __call__(self, x):
		self.feature_extractor.gradients = []
		self.feature_extractor.features = []
		out = self.feature_extractor(x)
		return out