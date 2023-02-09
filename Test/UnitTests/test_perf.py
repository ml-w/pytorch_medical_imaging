import unittest
import numpy as np
import seaborn as sns
from scipy.special import expit
from pytorch_med_imaging.perf.classification_perf import *

class TestPerf(unittest.TestCase):
    def __init__(self, *args ,**kwargs):
        super(TestPerf, self).__init__(*args, **kwargs)

    def setUp(self):
        self.example_data = sns.load_dataset('fmri')
        self.example_prediction = self.example_data['signal']
        self.example_gt = self.example_data['event']
        self.key_mapping = {k: i for i, k in enumerate(set(self.example_gt))}
        self.example_gt = self.example_gt.replace(to_replace=self.key_mapping)

    def test_dca(self):
        thres = np.linspace(0.1, 0.99, 100)
        nbs = plot_DCA(thres,expit(self.example_prediction),self.example_gt)

    def test_binary_performance(self):
        prediction = self.example_prediction > 1
        gt = self.example_gt
        binary_performance(prediction, gt)
        