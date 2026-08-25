import unittest
from pytorch_med_imaging.networks import *
import torch
import torch.nn as nn


class Test3DNetworks(unittest.TestCase):
    def setUp(self) -> None:
        num_slice = 30
        num_data = 4
        self.sample_input = torch.rand(num_data, 1, 128, 128, num_slice).cuda()
        self.sample_input_3d = torch.rand(num_data, 1, 128, 128, 128).cuda()
        self.sample_input_3d_size1 = torch.rand(1, 1, 128, 128, 128).cuda()

    def test_UNet_p(self):
        net = UNet_p(1, 2, layers=3).cuda()
        with torch.no_grad():
            x = self.sample_input[0].permute(3, 0, 1, 2)  # (30, 1, 128, 128)
            out = net(x)
            self.assertEqual((30, 2, 128, 128), out.shape)

    def test_VNet(self):
        net = VNet(1, 2).cuda()
        with torch.no_grad():
            out = net(self.sample_input_3d)
            self.assertEqual(self.sample_input_3d.shape[2:], out.shape[2:])
            out = net(self.sample_input_3d_size1)
            self.assertEqual(self.sample_input_3d_size1.shape[2:], out.shape[2:])
            with self.assertRaises(AssertionError):
                net(self.sample_input)
