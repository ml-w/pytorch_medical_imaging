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
        self.sample_input = torch.rand(2, 1, 128, 128, 30).cuda()
        self.sample_input[0, ..., 28::].fill_(0)
        self.sample_seg = torch.zeros_like(self.sample_input).cuda()
        self.sample_seg[0, 0, 50, 50, 10:20].fill_(1)
        self.sample_seg[1, 0, 50, 50, 8:15].fill_(1)

    def test_rAIdiologist(self):
        net = rAIdiologist(1).cuda()
        with torch.no_grad():
            for i in range(6):
                try:
                    net.set_mode(i)
                    self.assertTrue(net._mode == i)
                    out = net(self.sample_input)
                    print(f"Mode {i} passed.")
                except:
                    self.fail(f"Mode {i} error.")

    def test_rAIdiologist_focal(self):
        _inter_mediate_data = []
        def _get_input_hook(module, input, _):
            r"""This hook cleans the rAIdiologist playback list prior to running a mini-batch"""
            _inter_mediate_data.append(input[0].permute(0, 2, 1))

        net = rAIdiologist(1).cuda()
        handler = net.lstm_prelayernorm.register_forward_hook(_get_input_hook)
        with torch.no_grad():
            for mode in (1, 2):
                net.set_mode(mode)
                out = net(self.sample_input, self.sample_seg)
                temp_out = _inter_mediate_data[0]
                bool_index = self.sample_seg.sum(dim=[-2, -3]).bool()
                self.assertEqual(0, temp_out[~bool_index.expand_as(temp_out)].sum())
                self.assertNotEqual(0, temp_out[bool_index.expand_as(temp_out)].sum())
                _inter_mediate_data.clear()

                # raise error if all zeros
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
                print(f"Mode {mode} passed")