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
        num_data = 4
        self.sample_input_big = torch.rand(num_data, 1, 512, 512, num_slice).cuda()
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
        net = rAIdiologist(1, record=False).cuda()
        with torch.no_grad():
            for i in range(6):
                try:
                    net.set_mode(i)
                    self.assertTrue(net._mode == i)
                    out = net(self.sample_input)
                    self.assertEqual(2, out.dim(), "Failed during forward.")
                    out = net(self.sample_input_size1)
                    self.assertEqual(2, out.dim(), "Failed for batch-size = 1.")
                    print(f"Mode {i} passed.")
                except Exception as e:
                    self.fail(f"Mode {i} error. Original message {e}")


    def test_rAIdiologist_ch1(self):
        net = rAIdiologist(out_ch=1, record=False).cuda()
        with torch.no_grad():
            for i in range(6):
                try:
                    net.set_mode(i)
                    self.assertTrue(net._mode == i)
                    out = net(self.sample_input)
                    self.assertEqual(2, out.dim())
                    print(f"Mode {i} passed. Out size: {out.shape}")
                except Exception as e:
                    self.fail(f"Mode {i} error. Original message {e}")

    def test_rAIdiologist_recordon(self):
        net = rAIdiologist(out_ch=1, record=True).cuda()
        net.set_mode(5)
        with torch.no_grad():
            try:
                out = net(self.sample_input)
                self.assertNotEqual(0, len(net.get_playback()), "Playback length is zero")
                self.assertEqual(2, out.dim(), "Failed during forward.")
                out = net(self.sample_input_size1)
                self.assertEqual(2, out.dim(), "Failed for batch-size = 1.")
            except Exception as e:
                self.fail(f"Original message: {e}")
            self.assertNotEqual(0, len(net.get_playback()), "Play back is empty!")

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
            self.assertEqual(self.sample_input_3d.shape[2:], out.shape[2:])
            out = net(self.sample_input_3d_size1)
            self.assertEqual(self.sample_input_3d_size1.shape[2:], out.shape[2:])
            with self.assertRaises(AssertionError):
                net(self.sample_input)

    def test_ViT_VNet(self):
        from pytorch_med_imaging.networks.third_party_nets import ViTVNet
        net = ViTVNet(img_size = self.sample_input_big.shape[2:]).cuda()
        with torch.no_grad():
            out = net(self.sample_input_big)

    def test_SlicewiseRAN(self):
        from pytorch_med_imaging.networks.specialized.slicewise_ran import SlicewiseAttentionRAN
        for mode in SlicewiseAttentionRAN._strats_dict:
            net = SlicewiseAttentionRAN(1, 5, reduce_strats=mode).cuda()
            with torch.no_grad():
                out = net(self.sample_input)
                print(f"Mode {mode} passed.")

    def test_RAN25D(self):
        from pytorch_med_imaging.networks.specialized.slicewise_ran import AttentionRAN_25D, AttentionRAN_25D_MSE
        net = AttentionRAN_25D(1, 4).cuda()
        with torch.no_grad():
            out = net(self.sample_input)
            print(out.shape)

        net = AttentionRAN_25D(1, 1).cuda()
        with torch.no_grad():
            out = net(self.sample_input)
            print(out.shape)

        net = AttentionRAN_25D_MSE(1, 1, (1, 4)).cuda()
        with torch.no_grad():
            out = net(self.sample_input)
            print(out.shape)