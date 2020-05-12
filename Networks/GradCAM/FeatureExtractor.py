import torch
from torch.autograd import Variable

class FeatureExtractor(object):
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers, threeD=True):
        self.outer = model
        if isinstance(model, torch.nn.DataParallel):
            self.model = model.module
        else:
            self.model = model
        self.threeD = threeD
        self.target_layers = target_layers
        self.gradients = []
        self.features = []

        modnames = []
        mods = []
        for name, module in self.model.named_modules():
            # if name.find('.') == -1:
            modnames.append(name)
            mods.append(module)

        for name, module in self.model.named_modules():
            if name in self.target_layers:
                module.register_backward_hook(self.save_gradient)
                module.register_forward_hook(self.save_features)
                # nextmod = mods[modnames.index(name) + 1]
                # nextmod.register_forward_hook(self.save_features)

    def save_gradient(self, mod, grad_in, grad_out, name=None):
        # this is called n-times if they are put in multiple GPUs to compute
        # if not mod in self.gradients:
        #     self.gradients= {mod: [*grad_out]}
        # else:
        #     self.gradients[mod].append(*grad_out)
        print("Grad size: {}".format([g.shape for g in grad_out]))
        self.gradients.append(grad_out)

    def save_features(self, mod, input, output, name=None):
        # if not isinstance(output, tuple):
        #     output = (output)
        # if not mod in self.features:
        #     self.features = {mod: [*input]}
        # else:
        #     self.features[mod].append(*input)
        # self.features.append(input)

        print("Output size : {}".format([a.shape for a in output]))
        if len(output) > 1:
            self.features.append([torch.stack([a for a in output])])
        else:
            self.features.append(output)


    def __call__(self, x):
        out = self.outer(x)
        return out