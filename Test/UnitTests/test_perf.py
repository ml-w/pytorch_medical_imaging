import unittest
import numpy as np
import seaborn as sns
from scipy.special import expit
from typing import *
from pytorch_med_imaging.perf.classification_perf import *
from pytorch_med_imaging.perf.segmentation_perf import *
from mnts.mnts_logger import MNTSLogger

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



class TestPerfSegmentation(unittest.TestCase):
    def setUp(self):
        self._logger = MNTSLogger('.', 'pytest', verbose=True, log_level='DEBUG', keep_file=False)
        # Create mock data for segmentation
        self.mock_data_shape = (5, 1, 64, 64, 10)
        self.mock_gt = np.zeros(self.mock_data_shape).astype('int')
        self.mock_gt[:, :, :, :, 5:10] = 1

        self.mock_data_single_batch = (1, 1, 64, 64, 10)
        self.mock_gt_single_batch = np.zeros(self.mock_data_single_batch)
        self.mock_gt_single_batch[:,:, :, :, 5] = 1

        # Generate mock data with squares of ones in several slices for numpy
        self.mock_np_data = np.zeros(self.mock_data_shape).astype('int')  # Start with all zeros
        for i in range(5):  # Create 5 slices with squares of ones
            for j in range(5):
                start = np.random.randint(0, self.mock_data_shape[2] - 10)  # Random start for square
                self.mock_np_data[i, 0, start:start+10, start:start+10, :] = 1  # Fill square with ones


    def test_eval_numpy(self):
        # Test the EVAL function with numpy arrays
        results = EVAL(self.mock_np_data, self.mock_gt)
        self.assertIsInstance(results, pd.DataFrame)
        self.assertFalse(results.empty)
        self._logger.info('\n' + results.to_string())

    def test_eval_torchtensor(self):
        # Test the EVAL function with numpy arrays
        results = EVAL(torch.Tensor(self.mock_np_data).int(), torch.Tensor(self.mock_gt).int())
        self.assertIsInstance(results, pd.DataFrame)
        self.assertFalse(results.empty)
        self._logger.info('\n' + results.to_string())

    def test_eval_torch(self):
        # Test the EVAL function with torch tensors
        results = EVAL(torch.tensor(self.mock_np_data), torch.tensor(self.mock_gt))
        self.assertIsInstance(results, pd.DataFrame)
        self.assertFalse(results.empty)
        self._logger.info('\n' + results.to_string())

    def test_eval_invalid_shape(self):
        # Test the EVAL function with mismatched shapes
        with self.assertRaises(ArithmeticError):
            EVAL(self.mock_np_data, self.mock_gt[:, :, :, :, :9])

    def test_eval_empty_gt(self):
        # Test the EVAL function with an empty ground truth
        empty_gt = np.zeros_like(self.mock_np_data)
        p = EVAL(self.mock_np_data, empty_gt)
        self._logger.info('\n' + p.to_string())

    def test_metric_functions(self):
        # Test individual metric functions with mock data
        TP, FP, TN, FN = 10, 5, 15, 2

        jac = JAC(TP, FP, TN, FN)
        self.assertAlmostEqual(jac, TP / (TP + FP + FN))

        dice = DICE(TP, FP, TN, FN)
        self.assertAlmostEqual(dice, 2 * TP / (2 * TP + FP + FN))

        vs = VS(TP, FP, TN, FN)
        self.assertAlmostEqual(vs, 1 - abs(FN - FP) / (2 * TP + FP + FN))

        vd = VD(TP, FP, TN, FN)
        self.assertAlmostEqual(vd, 1 - vs)

        sensitivity = Sensitivity(TP, FP, TN, FN)
        self.assertAlmostEqual(sensitivity, TP / (TP + FN))

        specificity = Specificity(TP, FP, TN, FN)
        self.assertAlmostEqual(specificity, TN / (TP + FP))

    def test_surface_metrics_functions(self):
        # These function only works on one image at  time
        ASD = compute_ASD(self.mock_np_data[0].squeeze(), self.mock_gt[0].squeeze())
        HD = compute_HD(self.mock_np_data[0].squeeze(), self.mock_gt[0].squeeze())
        HD95 = compute_HD95(self.mock_np_data[0].squeeze(), self.mock_gt[0].squeeze())
        self._logger.info(f"ASD: {ASD:.2f}, HD: {HD:.2f}, HD95: {HD95:.2f}")

        # Test voxel spacing
        ASD  = compute_ASD(self.mock_np_data[0].squeeze() , self.mock_gt[0].squeeze(), voxel_spacing=(0.5, 0.5, 3))
        HD   = compute_HD(self.mock_np_data[0].squeeze()  , self.mock_gt[0].squeeze(), voxel_spacing=(0.5, 0.5, 3))
        HD95 = compute_HD95(self.mock_np_data[0].squeeze(), self.mock_gt[0].squeeze(), voxel_spacing=(0.5, 0.5, 3))
        self._logger.info(f"ASD: {ASD:.2f}, HD: {HD:.2f}, HD95: {HD95:.2f}")