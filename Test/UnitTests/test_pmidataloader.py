import unittest
from unittest.mock import MagicMock, patch
from pytorch_med_imaging.pmi_data_loader.pmi_dataloader_base import (
    PMIDataLoaderBase, PMIDataLoaderBaseCFG, _WORKER_CRASH_SIGNATURES
)
from pytorch_med_imaging.pmi_data_loader import *
from pytorch_med_imaging.pmi_data import DataLabel
from mnts.mnts_logger import MNTSLogger
from pathlib import Path
import torchio as tio
import copy
from pytorch_med_imaging.pmi_data import ImageDataSet


class TestDataLoader(unittest.TestCase):
    def setUp(self) -> None:
        if self.__class__.__name__ == 'TestDataLoader':
            raise unittest.SkipTest("Base class.")
        self._logger = MNTSLogger('.', logger_name=self.__class__.__name__, log_level='debug', keep_file=False)

    def __init__(self, *args, **kwargs):
        super(TestDataLoader, self).__init__(*args, **kwargs)
        self.loader: PMIDataLoaderBase = None

    def test_load_training_data(self):
        loader = self.loader._load_data_set_training()
        self.assertEqual(len(loader), self.expected_training_queue_len)
        for l in loader:
            break
        return loader

    def test_load_inference_data(self):
        loader = self.loader._load_data_set_inference()
        self.assertEqual(len(loader), self.expected_inference_queue_len)
        for l in loader:
            break
        return loader

    def test_add_data(self):
        datalabel = DataLabel.from_xlsx('./sample_data/sample_binaryclass_gt.xlsx')
        datalabel.set_target_column('Class')
        self.loader.append_data("additional_data", datalabel)
        loader = self.loader._load_data_set_training()
        for l in loader:
            self._logger.debug(l)
            self.assertIn('additional_data', l)
            break

class TestImageDataLoader(TestDataLoader):
    def setUp(self):
        super(TestImageDataLoader, self).setUp()
        # Setting class attributes make these values the new defaults when creating the cfg instances
        PMIImageDataLoaderCFG.input_dir    = Path('./sample_data/img/')
        PMIImageDataLoaderCFG.target_dir   = Path('./sample_data/seg/')
        PMIImageDataLoaderCFG.mask_dir     = Path('./sample_data/seg/')
        PMIImageDataLoaderCFG.probmap_dir  = Path('./sample_data/seg/')
        PMIImageDataLoaderCFG.augmentation = Path('./sample_data/config/sample_transform.yaml')
        PMIImageDataLoaderCFG.data_types   = [float, 'uint8']
        PMIImageDataLoaderCFG.id_globber   = "^\w+_\d+"
        PMIImageDataLoaderCFG.id_list     = ['MRI_01', 'MRI_02']
        PMIImageDataLoaderCFG.sampler     = 'weighted'
        PMIImageDataLoaderCFG.sampler_kwargs['patch_size']           = [32, 32, 3]
        PMIImageDataLoaderCFG.tio_queue_kwargs['samples_per_volume'] = 2
        PMIImageDataLoaderCFG.inf_samples_per_vol                    = 5
        self.cfg = PMIImageDataLoaderCFG()

        # expected variables
        self.num_subjects = len(self.cfg.id_list)
        self.expected_training_queue_len = self.num_subjects * self.cfg.tio_queue_kwargs['samples_per_volume']
        self.expected_inference_queue_len = self.num_subjects * self.cfg.inf_samples_per_vol

        # prepare loader
        self.loader = PMIImageDataLoader(self.cfg)
        pass

    def test_load_training_data(self):
        loader = super(TestImageDataLoader, self).test_load_training_data()
        for l in loader:
            self.assertTupleEqual(tuple(self.cfg.sampler_kwargs['patch_size']),
                                  tuple(l.shape[1:]))
            break
        return loader

    def test_load_inference_data(self):
        loader = super(TestImageDataLoader, self).test_load_inference_data()
        for l in loader:
            self.assertTupleEqual(tuple(self.cfg.sampler_kwargs['patch_size']),
                                  tuple(l.shape[1:]))
            break
        return loader

    def test_additional_instance(self):
        new_cfg = self.cfg.__class__(id_list=['MRI_02', 'MRI_03'])
        new_loader = self.loader.__class__(new_cfg)
        self.assertTupleEqual(tuple(new_loader.id_list),
                              ('MRI_02', 'MRI_03'))

    def test_no_sampler(self):
        self.cfg.sampler = None
        loader = self.loader.__class__(self.cfg)
        for l in loader.get_torch_data_loader(2):
            msg = self._logger.debug(f"MB keys: {l.keys()}")
            self.assertTupleEqual(tuple(l['input'][tio.DATA].shape),
                                  (2, 1, 250, 250, 15)) # Size specified in sample_transform.yaml
            break

class TestImageFeaturePairLoader(TestImageDataLoader):
    def setUp(self):
        super(TestImageFeaturePairLoader, self).setUp()
        self.cfg = PMIImageFeaturePairLoaderCFG()
        self.cfg.input_dir   = './sample_data/img/'
        self.cfg.target_dir  = './sample_data/sample_binaryclass_gt.xlsx'
        self.cfg.mask_dir    = './sample_data/seg/'
        self.cfg.probmap_dir = './sample_data/seg/'
        self.cfg.id_globber  = "^\w+_\d+"
        self.cfg.id_list     = ['MRI_01', 'MRI_02']
        self.cfg.sampler     = 'weighted'
        self.cfg.sampler_kwargs['patch_size']           = [32, 32, 3]
        self.cfg.tio_queue_kwargs['samples_per_volume'] = 2
        self.cfg.inf_samples_per_vol                    = 5
        self.cfg.excel_sheetname = 'sample_binaryclass_gt'
        self.cfg.target_column   = 'Class'
        self.cfg.augmentation    = './sample_data/config/sample_transform.yaml'

        # expected variables
        self.num_subjects = len(self.cfg.id_list)
        self.expected_training_queue_len = self.num_subjects * self.cfg.tio_queue_kwargs['samples_per_volume']
        self.expected_inference_queue_len = self.num_subjects * self.cfg.inf_samples_per_vol

        # prepare loader
        self.loader = PMIImageFeaturePairLoader(self.cfg)
        pass

    def test_load_training_data(self):
        loader = super(TestImageFeaturePairLoader, self).test_load_training_data()
        for l in loader:
            msg = f"Expect integer class for ground-truth, got {type(l['gt'])} instead"
            self.assertIsInstance(l['gt'].item(), int, msg)
            self._logger.debug(f"{l}")
            break

    def test_load_inference_data(self):
        loader = super(TestImageFeaturePairLoader, self).test_load_inference_data()
        for l in loader:
            msg = f"Expect integer class for ground-truth, got {type(l['gt'])} instead"
            self.assertIsInstance(l['gt'].item(), int, msg)
            self._logger.debug(f"{l}")
            break


class TestImageFeaturePairLoaderConcat(TestImageFeaturePairLoader):
    def setUp(self):
        super(TestImageFeaturePairLoaderConcat, self).setUp()
        self.cfg.target_dir = "./sample_data/sample_concat_df.xlsx"
        self.cfg.excel_sheetname = None
        self.cfg.target_column = "conclusion"
        self.cfg.data_types = [float, str]

        self.loader = PMIImageFeaturePairLoaderConcat(self.cfg)

    def test_load_training_data(self):
        loader = super(TestImageFeaturePairLoader, self).test_load_training_data()
        for l in loader:
            msg = f"Expect integer class for ground-truth, got {type(l['gt'])} instead"
            self.assertIsInstance(l['gt'], str, msg)
            self._logger.debug(f"{l}")
            break

    def test_load_inference_data(self):
        loader = super(TestImageFeaturePairLoader, self).test_load_inference_data()
        for l in loader:
            msg = f"Expect integer class for ground-truth, got {type(l['gt'])} instead"
            self.assertIsInstance(l['gt'], str, msg)
            self._logger.debug(f"{l}")
            break


class TestPMIImageMCDataLoader(TestImageDataLoader):
    def setUp(self):
        super(TestPMIImageMCDataLoader, self).setUp()
        self.cfg = PMIImageMCDataLoaderCFG( # instance attribute can be defined like this too
            mask_dir    = './sample_data/seg/',
            probmap_dir = './sample_data/seg/',
            id_globber = "^\w+_\d+"
        ) # note that super() already defined parent class attributes
        self.cfg.input_dir      = './sample_data/'
        self.cfg.target_dir     = './sample_data/'
        self.cfg.input_subdirs  = ['img'    , 'img']
        self.cfg.target_subdirs = ['seg'    , 'seg']
        self.cfg.new_attr       = ['img_new', 'seg_new']
        self.cfg.data_types     = [float    , 'uint8']
        self.cfg.id_list        = ['MRI_01' , 'MRI_02']
        self.cfg.sampler        = 'weighted'
        self.cfg.sampler_kwargs['patch_size']           = [32, 32, 3]
        self.cfg.tio_queue_kwargs['samples_per_volume'] = 2
        self.cfg.inf_samples_per_vol                    = 5
        self.loader = PMIImageMCDataLoader(self.cfg)

    def test_load_training_data(self):
        loader = super(TestImageDataLoader, self).test_load_training_data()
        for l in loader:
            _shape = l[self.cfg.new_attr[0]].shape
            self.assertTupleEqual(tuple(self.cfg.sampler_kwargs['patch_size']),
                                  tuple(_shape[1:]))
            self.assertEqual(len(self.cfg.input_subdirs),
                             _shape[0])
            break

    def test_load_inference_data(self):
        loader = super(TestImageDataLoader, self).test_load_inference_data()
        for l in loader:
            _shape = l[self.cfg.new_attr[0]].shape
            self.assertTupleEqual(tuple(self.cfg.sampler_kwargs['patch_size']),
                                  tuple(_shape[1:]))
            self.assertEqual(len(self.cfg.input_subdirs),
                             _shape[0])
            break

    def test_no_sampler(self):
        self.cfg.sampler = None
        loader = self.loader.__class__(self.cfg)
        for l in loader.get_torch_data_loader(2):
            msg = self._logger.debug(f"MB keys: {l.keys()}")
            self.assertTupleEqual(tuple(l['img_new'][tio.DATA].shape),
                                  (2, 2, 250, 250, 15)) # Size specified in sample_transform.yaml
            break


class TestPMITorchioDataLoader(TestDataLoader):
    def setUp(self):
        super().setUp()
        self.cfg = PMITorchioDataLoaderCFG(
            input_data = {
                'input'  : './sample_data/img/',
                'gt'     : './sample_data/seg/',
                'probmap': './sample_data/seg/'
            },
            input_dtypes = {
                'gt': 'uint8',
                'probmap': 'uint8'
            },
            augmentation = Path('./sample_data/config/sample_transform.yaml'),
            id_globber = "^\w+_\d+",
            id_list = ['MRI_01', 'MRI_02'],
            sampler = 'weighted',
            inf_samples_per_vol = 5
        )
        self.cfg.sampler_kwargs['patch_size']           = [32, 32, 3]
        self.cfg.tio_queue_kwargs['samples_per_volume'] = 2
        self._logger.debug(f"{self.cfg = }")

        # expected variables
        self.num_subjects = len(self.cfg.id_list)
        self.expected_training_queue_len = self.num_subjects * self.cfg.tio_queue_kwargs['samples_per_volume']
        self.expected_inference_queue_len = self.num_subjects * self.cfg.inf_samples_per_vol

        # prepare loader
        self.loader = PMITorchioDataLoader(self.cfg)

    def test_get_subject(self):
        data_loader = self.loader.get_torch_data_loader(batch_size=2)
        for mb in data_loader:
            self.assertIn('input', mb)
            self.assertIn('gt', mb)
            self.assertIn('probmap', mb)

    def test_map_to_master(self):
        d = DataLabel.from_xlsx('./sample_data/sample_binaryclass_gt.xlsx')
        d.set_target_column("Class")
        new_list = ['MRI_01', 'MRI_02', 'MRI_04']

        cfg = copy.deepcopy(self.cfg)
        cfg.master_data_key = 'input'
        cfg.id_list = new_list
        cfg.input_data['label'] = d
        loader = PMITorchioDataLoader(cfg)

        for i, v in enumerate(loader._load_data_set_training()):
            self.assertIn(v['uid'], new_list)
            if i == 3:
                break

        for i, v in enumerate(loader._load_data_set_inference()):
            self.assertIn(v['uid'], new_list)
            if i == 3:
                break


# ---------------------------------------------------------------------------
# Minimal concrete stub — no real I/O, used only for NFS resilience tests
# ---------------------------------------------------------------------------
class _StubLoader(PMIDataLoaderBase):
    def _check_input(self):         return True
    def _load_data_set_training(self, exclude_augment=False): return MagicMock()
    def _load_data_set_inference(self): return MagicMock()
    def _prepare_data(self):        return {}


def _stub_loader(**overrides):
    """Return a _StubLoader with _torch_loader pre-set and no real data loading."""
    loader = object.__new__(_StubLoader)
    loader._torch_loader = MagicMock()
    loader.nfs_resilient_max_retries = overrides.get('nfs_resilient_max_retries', 10)
    loader.nfs_resilient_backoff     = overrides.get('nfs_resilient_backoff', 0.0)
    return loader


class TestNFSResilience(unittest.TestCase):
    """Unit tests for NFS worker-crash resilience in PMIDataLoaderBase.__iter__."""

    def test_normal_iteration_yields_all_batches(self):
        loader = _stub_loader()
        batches = ['a', 'b', 'c']
        loader._torch_loader.__iter__ = MagicMock(return_value=iter(batches))
        loader._torch_loader.__len__  = MagicMock(return_value=3)
        self.assertEqual(list(loader), batches)

    def test_len_proxies_to_inner_loader(self):
        loader = _stub_loader()
        loader._torch_loader.__len__ = MagicMock(return_value=42)
        self.assertEqual(len(loader), 42)

    def test_single_crash_is_recovered(self):
        loader = _stub_loader(nfs_resilient_max_retries=3)
        crash  = RuntimeError('unable to open shared memory object: No such file or directory')
        call_count = [0]

        def flaky_iter():
            call_count[0] += 1
            if call_count[0] == 1:
                yield 'batch_0'
                raise crash
            yield 'batch_1'
            yield 'batch_2'

        loader._torch_loader.__iter__ = flaky_iter
        loader._torch_loader.dataset  = None

        with patch('pytorch_med_imaging.pmi_data_loader.pmi_dataloader_base._time.sleep') as mock_sleep, \
             patch('pytorch_med_imaging.pmi_data_loader.pmi_dataloader_base._warnings.warn'):
            result = list(loader)

        mock_sleep.assert_called_once_with(0.0)
        self.assertEqual(result, ['batch_0', 'batch_1', 'batch_2'])

    def test_exceeding_max_retries_reraises(self):
        loader = _stub_loader(nfs_resilient_max_retries=2)
        crash  = RuntimeError('shared memory object: No such file or directory')

        def always_crash():
            raise crash
            yield  # make it a generator

        loader._torch_loader.__iter__ = always_crash
        loader._torch_loader.dataset  = None

        with patch('pytorch_med_imaging.pmi_data_loader.pmi_dataloader_base._time.sleep'), \
             patch('pytorch_med_imaging.pmi_data_loader.pmi_dataloader_base._warnings.warn'):
            with self.assertRaises(RuntimeError) as ctx:
                list(loader)

        self.assertIs(ctx.exception, crash)

    def test_consecutive_counter_resets_on_success(self):
        """A successful batch resets the retry counter so a later crash still gets retries."""
        loader = _stub_loader(nfs_resilient_max_retries=1)
        crash  = RuntimeError('shared memory: No such file or directory')
        attempt = [0]

        def intermittent():
            attempt[0] += 1
            if attempt[0] == 1:
                yield 'ok_1'
                raise crash
            yield 'ok_2'

        loader._torch_loader.__iter__ = intermittent
        loader._torch_loader.dataset  = None

        with patch('pytorch_med_imaging.pmi_data_loader.pmi_dataloader_base._time.sleep'), \
             patch('pytorch_med_imaging.pmi_data_loader.pmi_dataloader_base._warnings.warn'):
            result = list(loader)

        self.assertEqual(result, ['ok_1', 'ok_2'])

    def test_unrelated_runtime_error_is_not_caught(self):
        loader    = _stub_loader(nfs_resilient_max_retries=5)
        unrelated = RuntimeError('CUDA out of memory.')

        def bad_iter():
            raise unrelated
            yield

        loader._torch_loader.__iter__ = bad_iter
        loader._torch_loader.dataset  = None

        with patch('pytorch_med_imaging.pmi_data_loader.pmi_dataloader_base._time.sleep') as mock_sleep:
            with self.assertRaises(RuntimeError) as ctx:
                list(loader)

        self.assertIs(ctx.exception, unrelated)
        mock_sleep.assert_not_called()

    def test_queue_workers_are_restarted_on_crash(self):
        loader = _stub_loader(nfs_resilient_max_retries=3)
        crash  = RuntimeError('shared memory: No such file or directory')

        mock_queue = MagicMock(spec=tio.Queue)
        mock_queue.patches_list       = MagicMock()
        mock_queue._subjects_iterable = 'old_iter'
        iters = [0]

        def flaky():
            iters[0] += 1
            if iters[0] == 1:
                raise crash
            yield 'batch'

        loader._torch_loader.__iter__ = flaky
        loader._torch_loader.dataset  = mock_queue

        with patch('pytorch_med_imaging.pmi_data_loader.pmi_dataloader_base._time.sleep'), \
             patch('pytorch_med_imaging.pmi_data_loader.pmi_dataloader_base._warnings.warn'):
            result = list(loader)

        mock_queue.patches_list.clear.assert_called_once()
        self.assertIsNone(mock_queue._subjects_iterable)
        mock_queue._initialize_subjects_iterable.assert_called_once()
        self.assertEqual(result, ['batch'])

    def test_iter_without_torch_loader_raises(self):
        loader = object.__new__(_StubLoader)
        with self.assertRaises(RuntimeError):
            list(loader)

    def test_all_crash_signatures_trigger_recovery(self):
        for sig in _WORKER_CRASH_SIGNATURES:
            with self.subTest(signature=sig):
                loader = _stub_loader(nfs_resilient_max_retries=1)
                crash  = RuntimeError(f'prefix {sig.upper()} suffix')
                iters  = [0]

                def flaky():
                    iters[0] += 1
                    if iters[0] == 1:
                        raise crash
                    yield 'ok'

                loader._torch_loader.__iter__ = flaky
                loader._torch_loader.dataset  = None

                with patch('pytorch_med_imaging.pmi_data_loader.pmi_dataloader_base._time.sleep'), \
                     patch('pytorch_med_imaging.pmi_data_loader.pmi_dataloader_base._warnings.warn'):
                    result = list(loader)

                self.assertEqual(result, ['ok'])
                iters[0] = 0
