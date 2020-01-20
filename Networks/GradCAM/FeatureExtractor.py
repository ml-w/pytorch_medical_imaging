import torch
from torch.autograd import Variable

class FeatureExtractor(object):
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []
        self.features = []
        modnames = []
        mods = []
        for name, module in self.model.named_modules():
            if name.find('.') == -1:
                modnames.append(name)
                mods.append(module)

        for name, module in self.model.named_modules():
            if name in self.target_layers:
                module.register_backward_hook(self.save_gradient)
                # module.register_forward_hook(self.save_features)
                nextmod = mods[modnames.index(name) + 1]
                nextmod.register_forward_hook(self.save_features)

    def save_gradient(self, mod, grad_in, grad_out):
        self.gradients.append(grad_out)

    def save_features(self, mod, input, output):
        # if not isinstance(output, tuple):
        #     output = (output)
        self.features.append(input)

    def __call__(self, x):
        return self.model(x)