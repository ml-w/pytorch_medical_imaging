import torch
import torch.nn as nn

class BatchNormLayer(nn.Module):
    def __init__(self):
        super(BatchNormLayer, self).__init__()
        self.train(False)

    def forward(self, input):
        means = torch.cat([torch.mean(input[i]) for i in xrange(input.data.size()[0])])
        vars = torch.cat([torch.var(input[i]) for i in xrange(input.data.size()[0])])
        x = input.transpose(0, -1)
        x = (x - means.expand_as(x)) / vars.expand_as(x)
        x = x.transpose(0, -1)
        return x
