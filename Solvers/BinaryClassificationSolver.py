from .ClassificationSolver import ClassificationSolver

import torch.nn as nn

class BinaryClassificationSolver(ClassificationSolver):
    def __init__(self, *args, **kwargs):
        super(BinaryClassificationSolver, self).__init__(*args, **kwargs)

        in_data = args[0]
        gt_data = args[1]
        net = args[2]


        # Recalculate number of one_hot slots and rebuild the lab
        self._log_print("Rebuilding classification solver to binary classification.")
        numberOfClasses = gt_data[0].size()[1]
        inchan = in_data[0].size()[0]
        self._log_print("Found number of binary classes {}.".format(numberOfClasses))

        net = net(inchan, numberOfClasses)

        self._net = net
