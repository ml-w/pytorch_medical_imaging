import unittest
from pytorch_med_imaging.networks import *
from pytorch_med_imaging.networks.specialized import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class Test3DNetworks(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test3DNetworks, self).__init__(*args, **kwargs)

    def setUp(self) -> None:
        self.sample_input = torch.rand(2, 1, 350, 350, 35).cuda()
        self.sample_input[0, ..., 25::].fill_(0)

    def test_rAIdiologist(self):
        net = rAIdiologist(1).cuda()
        with torch.no_grad():
            for i in range(6):
                try:
                    net.set_mode(i)
                    self.assertTrue(net._mode == i)
                    out = net(self.sample_input)
                except:
                    self.fail(f"Mode {i} error.")
