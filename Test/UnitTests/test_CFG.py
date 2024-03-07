from pytorch_med_imaging.pmi_base_cfg import PMIBaseCFG
import unittest


class TestPMICFG(unittest.TestCase):
    def test_cfg_recode(self):
        base_cfg = PMIBaseCFG(
            fold_code = 'B00',
            fold_path = '/path/{fold_code}/{fold_code}_targets',
            version_code = 'v1.0',
            checkpoint_dir = '/path/{fold_code}/checkpoint_{version_code}.pt'
        )
        self.assertEqual(base_cfg.fold_path, '/path/B00/B00_targets')
        self.assertEqual(base_cfg.checkpoint_dir, '/path/B00/checkpoint_v1.0.pt')

    def test_cfg_recurprotection(self):
        base_cfg = PMIBaseCFG(
            A = '{B}',
            B = '{A}',
            C = '{A}_{B}',
        )
        # There's only one layer
        self.assertEqual('{A}', base_cfg.A)
        self.assertEqual('{A}_{A}', base_cfg.C)