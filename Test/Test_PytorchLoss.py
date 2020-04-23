import torch
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from Loss import *
from Algorithms.Analysis import DICE, perf_measure


def test_Dice_loss():
    n = 5
    c = 2
    h = w = 512

    # Create squares probability input
    input = torch.zeros(size=[n, c, h, w], dtype=torch.float32)
    for i in range(n):
        input[i, 1, i * 45: (i + 1) * 35, i * 35: (i + 1) * 35] = 1
        input[i, 0] = 1.
        # input[i, 2, i * 45: (i + 1) * 45, i * 45: (i + 1) * 45] = 0.5

    target = torch.zeros(size=[n, h, w], dtype=torch.uint8)
    for i in range(n):
        for j in range(c):
            target[i, i * 35: (i + 1) * 35, i * 35: (i + 1) * 35] = j

    input = Variable(input, requires_grad=True)
    target = Variable(target, requires_grad=False)
    loss = SoftDiceLoss(epsilon=0, weight=torch.tensor([1., 1]))
    loss2 = CrossEntropyLoss()
    l = loss(input, target)
    l2 = loss2(input,target.long())
    l2.backward()
    l.backward()

    for i in range(n):
        print(DICE(*perf_measure(input.detach().numpy()[i,1], target.detach().numpy()[i])))






if __name__ == '__main__':
    test_Dice_loss()
