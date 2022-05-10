from pytorch_med_imaging.loss import *
import unittest

class TestLoss(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLoss, self).__init__(*args, **kwargs)

    def test_ConfBCELoss(self):
        import torch
        import torch.autograd as autograd

        loss_func = ConfidenceBCELoss()
        test_target = torch.DoubleTensor([1, 1, 0, 0, 1]).view(5, -1)
        test_input = torch.DoubleTensor([[10.4, 23.6, 1],
                                         [3.8, 10.2, 1],
                                         [-1.9, 0.9, 1],
                                         [10, -5, 1],
                                         [123, 100, 1]]).view(5, -1)
        test_target = autograd.Variable(test_target, requires_grad=True)
        test_input = autograd.Variable(test_input, requires_grad=True)

        if torch.cuda.is_available():
            loss_func = loss_func.cuda()
            test_target = test_target.cuda()
            test_input = test_input.cuda()

        loss = loss_func(test_input, test_target)
        loss.backward()

