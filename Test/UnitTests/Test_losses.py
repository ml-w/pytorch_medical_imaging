from pytorch_med_imaging.loss import *
import unittest

class TestLoss(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLoss, self).__init__(*args, **kwargs)

    def setUp(self) -> None:
        import torch
        import torch.autograd as autograd

        self.test_target = torch.FloatTensor([1, 1, 0, 0, 1]).view(5, -1)
        self.test_input_1 = torch.FloatTensor([[0.99, 0.9, 1],
                                               [0.90, 0.9, 1],
                                               [0.01, 0.9, 1],
                                               [0.10, 0.9, 1],
                                               [0.99, 0.9, 1]]).view(5, -1) # High confidence correct
        self.test_input_2 = torch.FloatTensor([[0.1, 0.9, 1],
                                               [0.1, 0.9, 1],
                                               [0.9, 0.9, 1],
                                               [0.9, 0.9, 1],
                                               [0.1, 0.9, 1]]).view(5, -1) # High confidence wrong
        self.test_input_3 = torch.FloatTensor([[0.1, 0.1, 1],
                                               [0.1, 0.1, 1],
                                               [0.9, 0.1, 1],
                                               [0.9, 0.1, 1],
                                               [0.1, 0.1, 1]]).view(5, -1) # Low confidence wrong
        self.test_input_4 = torch.FloatTensor([[0.9, 0.1, 1],
                                               [0.9, 0.1, 1],
                                               [0.1, 0.1, 1],
                                               [0.1, 0.1, 1],
                                               [0.9, 0.1, 1]]).view(5, -1) # Low confidence right
        self.test_input_5 = torch.FloatTensor([0.9,
                                               0.1,
                                               0.1,
                                               0.9,
                                               0.9]).view(5, -1)

        self.test_inputs = [self.test_input_1,
                            self.test_input_2,
                            self.test_input_3,
                            self.test_input_4,
                            self.test_input_5]
        self.test_target = autograd.Variable(self.test_target, requires_grad=True)
        for i, t in enumerate(self.test_inputs):
            self.test_inputs[i] = autograd.Variable(t, requires_grad=True)

    def test_ConfBCELoss(self):
        loss_func = ConfidenceBCELoss(conf_factor=1)
        if torch.cuda.is_available():
            loss_func = loss_func.cuda()
            self.test_target = self.test_target.cuda()
            for i, t in enumerate(self.test_inputs):
                self.test_inputs[i] = t.cuda()

        for t in self.test_inputs:
            msg = f"Loss function failed when handling input with shape ({t.shape}): \n{t}"
            loss = loss_func(t, self.test_target)
            self.assertIn(loss.dim(), (0, 1), msg=msg + " Loss must be a single number!")
            try:
                loss.backward()
            except Exception as e:
                msg += f"The loss shape was: {loss.shape}"
                self.fail(msg)

