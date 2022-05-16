from pytorch_med_imaging.loss import *
import unittest

class TestLoss(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLoss, self).__init__(*args, **kwargs)

    def test_ConfBCELoss(self):
        import torch
        import torch.autograd as autograd

        loss_func = ConfidenceBCELoss(conf_factor=1)
        test_target = torch.FloatTensor([1, 1, 0, 0, 1]).view(5, -1)
        test_input_1 = torch.FloatTensor([[0.99, 0.9, 1],
                                          [0.90, 0.9, 1],
                                          [0.01, 0.9, 1],
                                          [0.10, 0.9, 1],
                                          [0.99, 0.9, 1]]).view(5, -1) # High confidence correct
        test_input_2 = torch.FloatTensor([[0.1, 0.9, 1],
                                          [0.1, 0.9, 1],
                                          [0.9, 0.9, 1],
                                          [0.9, 0.9, 1],
                                          [0.1, 0.9, 1]]).view(5, -1) # High confidence wrong
        test_input_3 = torch.FloatTensor([[0.1, 0.1, 1],
                                          [0.1, 0.1, 1],
                                          [0.9, 0.1, 1],
                                          [0.9, 0.1, 1],
                                          [0.1, 0.1, 1]]).view(5, -1) # Low confidence wrong
        test_input_4 = torch.FloatTensor([[0.9, 0.1, 1],
                                          [0.9, 0.1, 1],
                                          [0.1, 0.1, 1],
                                          [0.1, 0.1, 1],
                                          [0.9, 0.1, 1]]).view(5, -1) # Low confidence right

        test_inputs = [test_input_1,
                       test_input_2,
                       test_input_3,
                       test_input_4]
        test_target = autograd.Variable(test_target, requires_grad=True)
        for i, t in enumerate(test_inputs):
            test_inputs[i] = autograd.Variable(t, requires_grad=True)

        if torch.cuda.is_available():
            loss_func = loss_func.cuda()
            test_target = test_target.cuda()
            for i, t in enumerate(test_inputs):
                test_inputs[i] = t.cuda()

        losses = [loss_func(t, test_target) for t in test_inputs]
        print(torch.stack(losses))

