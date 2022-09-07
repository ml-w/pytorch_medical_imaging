import unittest
from pytorch_model_summary import summary
from pytorch_med_imaging.networks import *
from pytorch_med_imaging.networks.specialized import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class Test3DNetworks(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test3DNetworks, self).__init__(*args, **kwargs)

    def setUp(self) -> None:
        num_slice = 30
        num_data = 2
        self.sample_input = torch.rand(num_data, 1, 128, 128, num_slice).cuda()
        self.sample_input_size1 = torch.rand(1, 1, 128, 128, num_slice).cuda()
        self.sample_input_3d = torch.rand(num_data, 1, 128, 128, 128).cuda()
        self.sample_input_3d_size1 = torch.rand(1, 1, 128, 128, 128).cuda()
        self.sample_input[0, ..., 28::].fill_(0)
        self.sample_seg = torch.zeros_like(self.sample_input).cuda()
        self.sample_seg[0, 0, 50, 50, 10:20].fill_(1)
        self.sample_seg[1, 0, 50, 50, 8:15].fill_(1)
        self.sample_seg_size1 = torch.zeros_like(self.sample_input_size1).cuda()
        self.sample_seg_size1[0, 0, 50, 50, 10:20].fill_(1)
        self.expect_nonzero = torch.zeros([num_data, 1, num_slice], dtype=bool)
        self.expect_nonzero[0, ..., 9:21] = True
        self.expect_nonzero[1, ..., 7:16] = True

    def test_rAIdiologist(self):
        net = rAIdiologist(record=False).cuda()
        with torch.no_grad():
            for i in range(6):
                try:
                    net.set_mode(i)
                    self.assertTrue(net._mode == i)
                    out = net(self.sample_input)
                    self.assertEqual(2, out.dim())
                    print(f"Mode {i} passed.")
                except:
                    self.fail(f"Mode {i} error.")

    def test_rAIdiologist_focal(self):
        _inter_mediate_data = []
        def _get_input_hook(module, input, _):
            r"""This hook cleans the rAIdiologist playback list prior to running a mini-batch"""
            _inter_mediate_data.append(input[0].permute(0, 2, 1))

        net = rAIdiologist(record=False).cuda()
        handler = net.lstm_prelayernorm.register_forward_hook(_get_input_hook)
        with torch.no_grad():
            for mode in (1, 2):
                net.set_mode(mode)
                out = net(self.sample_input, self.sample_seg)
                temp_out = _inter_mediate_data[0]
                bool_index = self.expect_nonzero
                print(out.shape)
                # Assert the temp_out is zero where the segmentation is zero
                self.assertEqual(0, temp_out[~bool_index.expand_as(temp_out)].sum())
                # Assert the temp out is not zero where the segmetnation is not zero
                self.assertNotEqual(0, temp_out[bool_index.expand_as(temp_out)].sum())
                _inter_mediate_data.clear()

                # raise error if there are no segmentation (all zeros)
                zeros = torch.zeros_like(self.sample_input)
                zeros[0] = 1 # One of the batch member wasn't all zero
                with self.assertRaises(ArithmeticError):
                    net(self.sample_input, zeros)
                _inter_mediate_data.clear()
                print(f"Mode {mode} passed")
            for mode in (3, 4, 5):
                net.set_mode(mode)
                out = net(self.sample_input, self.sample_seg)
                for temp_out in _inter_mediate_data:
                    # None of the inputs should contain slices with all zeros
                    self.assertFalse(0 in list(temp_out.sum(dim=[0, 1])))
                _inter_mediate_data.clear()

                zeros = torch.zeros_like(self.sample_input)
                zeros[0] = 1 # One of the batch member wasn't all zero
                with self.assertRaises(ArithmeticError):
                    net(self.sample_input, zeros)
                _inter_mediate_data.clear()

                # check if things are correct when there's only one slice with nonzero
                zeros = torch.zeros_like(self.sample_input)
                zeros[0, ..., 12] = 1
                zeros[1, ..., 13] = 1
                out = net(self.sample_input, zeros)
                print(f"{out.shape}")
                print(f"Mode {mode} passed")

    def test_rAIdiologist_focal_size1(self):
        _inter_mediate_data = []
        def _get_input_hook(module, input, _):
            r"""This hook cleans the rAIdiologist playback list prior to running a mini-batch"""
            _inter_mediate_data.append(input[0].permute(0, 2, 1))

        net = rAIdiologist(record=False).cuda()
        handler = net.lstm_prelayernorm.register_forward_hook(_get_input_hook)
        with torch.no_grad():
            for mode in (1, 2, 3, 4, 5):
                net.set_mode(mode)
                out = net(self.sample_input_size1, self.sample_seg_size1)
                self.assertEqual(2, out.dim())

    def test_rAIdiologist_record(self):
        net = rAIdiologist(record=True).cuda()
        with torch.no_grad():
            for i in range(6):
                try:
                    net.set_mode(i)
                    self.assertTrue(net._mode == i)
                    out = net(self.sample_input)
                    self.assertEqual(2, out.dim())
                    print(f"Mode {i} passed.")
                except:
                    self.fail(f"Mode {i} error.")

    def test_AttentionUNet(self):
        net = AttentionUNet(1, 2).cuda()
        with torch.no_grad():
            # in dim: (B x C x W x H)
            out = net(self.sample_input[0].permute(3, 0, 1, 2))
            self.assertEqual((30, 2, 128, 128), out.shape)

    def test_DenseUNet(self):
        net = DenseUNet2D(1, 2).cuda()
        with torch.no_grad():
            # in dim: (B x C x W x H)
            out = net(self.sample_input[0].permute(3, 0, 1, 2))
            self.assertEqual((30, 2, 128, 128), out.shape)

    def test_VNet(self):
        net = VNet(1, 2).cuda()
        with torch.no_grad():
            out = net(self.sample_input_3d)
            out = net(self.sample_input_3d_size1)