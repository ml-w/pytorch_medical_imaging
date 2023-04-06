import torch
import torch.nn as nn
import unittest
from torch.masked import masked_tensor, as_masked_tensor
from pytorch_med_imaging.networks.layers.StandardLayers3D import activation_funcs
from pytorch_med_imaging.networks.layers.StandardLayers3D import *
from pytorch_med_imaging.networks.AttentionResidual import *

class Test3dLayers(unittest.TestCase):
    r"""Test the 3D layers implemented in pmi.networks.layers"""
    def setUp(self):
        self.batch_size = 4
        self.in_ch = 8
        self.out_ch = 16
        self.num_slice = 30
        self.input_3d = torch.rand([self.batch_size,
                                    self.in_ch, 128, 128,
                                    self.num_slice]).cuda()
        self.activations = activation_funcs.keys()

    def test_standardlayersforward(self):
        r"""Test the standard layers implemented in pmi.networks.layers.StandardLayers3D"""

        # Test Conv3d
        for k in self.activations:
            self.test_layer = Conv3d(self.in_ch, self.out_ch, activation=k).cuda()
            self._test_forward()
            print(f"{k} passed")
        print(f"=== Conv3d all passed ===")

        # Test DoubleConv3d
        for k in self.activations:
            self.test_layer = DoubleConv3d(self.in_ch, self.out_ch, activation=k).cuda()
            self._test_forward()
            print(f"{k} passed")
        print(f"=== DoubleConv3d all passed ===")

    def _test_masked_forward(self, x, seq_length, axis):
        r"""Test the forward pass of :class:`MaskedSequential3d`"""
        # Compute the output from the masked convolution layers
        output = self.test_layer(x, seq_length=seq_length, axis=axis)

        # Check that the output tensor has the expected shape
        expected_shape = (self.batch_size,
                          self.out_ch, 128, 128,
                          self.num_slice)
        self.assertEqual(output.shape, expected_shape)

        # Check if masked location remained as zero.
        for b in range(self.input_3d.shape[0]):
            self.assertEqual(output[b, ..., seq_length[b] + 1:].sum(), 0)

        # Check backwards
        self.test_layer.zero_grad()
        output[b].sum().backward()
        grads = [m.weight.grad for m in self.test_layer.modules() if hasattr(m, 'weight')]
        grads = [g.sum() for g in grads if g is not None]
        self.assertNotEqual(torch.stack(grads).sum(), 0)

    def _test_forward(self):
        # Compute the output of the DoubleConv3d module
        output = self.test_layer(self.input_3d)

        # Check that the output tensor has the expected shape
        expected_shape = (self.batch_size,
                          self.out_ch, 128, 128,
                          self.num_slice)
        self.assertEqual(output.shape, expected_shape)

class TestAttentionLayers(unittest.TestCase):
    r"""Test the attention layers"""