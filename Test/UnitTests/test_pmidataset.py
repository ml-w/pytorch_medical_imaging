import tempfile
import unittest
from pathlib import Path

import torch
import numpy as np
import torchio as tio
from mnts.mnts_logger import MNTSLogger
from pytorch_med_imaging.pmi_data import *


class Test_PMIData(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_PMIData, self).__init__(*args, **kwargs)
        pass

    @classmethod
    def setUpClass(cls) -> None:
        cls._logger = MNTSLogger('.', logger_name=cls.__name__, verbose=True,
                                 keep_file=False, log_level='debug')
        cls._expected_class = torch.Tensor

    @classmethod
    def tearDownClass(cls):
        # del cls._logger
        # pass
        del cls._logger

    def setUp(self):
        if self.__class__.__name__ == 'Test_PMIData':
            raise unittest.SkipTest("Base class")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)
        self._id_globber = "MRI_\d+"
        MNTSLogger('.', verbose=True, log_level='debug')

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_get_multipledata(self):
        data = self.data[1:3]
        for dd in data:
            self.assertIsInstance(dd, self._expected_class)
        return data

    def test_getUniqueIDs(self):
        ids = self.data.get_unique_IDs()
        self.assertTupleEqual(tuple(ids),
                              tuple(f"MRI_0{i+1}" for i in range(len(self.data))))

    def test_getDataByID(self):
        data = self.data.get_data_by_ID("MRI_02")
        if isinstance(self.data[1], torch.Tensor):
            self.assertTrue(torch.allclose(data, self.data[1]))    # Test also the sorting order is the same
        else:
            self._logger.warning(f"{self.__class__.__name__} does not return tensors.")
            self._logger.debug(f"{data = }")

    def test_getDataByIndex(self):
        data = self.data[0]

    def test_sort(self):
        data = self.data.sort_uid()

    def test_iterator(self):
        for d in self.data:
            self.assertIsInstance(d, self._expected_class)

    def test_gettype(self):
        self.data.dtype

    @classmethod
    def get_class_name(cls):
        return cls.__class__.__name__


class Test_ImageDataSet(Test_PMIData):
    def __init__(self, *args, **kwargs):
        super(Test_ImageDataSet, self).__init__(*args, **kwargs)

    def setUp(self):
        from pytorch_med_imaging.pmi_data import ImageDataSet
        super(Test_ImageDataSet, self).setUp()
        self.data_path = Path("./sample_data/img")
        self.seg_path = Path("./sample_data/seg")
        self.data = ImageDataSet(str(self.data_path), verbose=True, id_globber=self._id_globber)
        self.data_segment = ImageDataSet(str(self.seg_path), verbose=True, id_globber=self._id_globber)
        self._expected_class = tio.Image

    def test_matchID(self):
        self._logger.debug(f"{self.data.get_unique_IDs()}")
        self._logger.debug(f"{self.data_segment.get_unique_IDs()}")
        self.assertTrue(self.data.get_unique_IDs() == self.data_segment.get_unique_IDs())

    def test_printmetadata(self):
        self.data.update_metadata_table()
        self._logger.debug(f"{self.data.metadata_table}")

    def test_nonuniqueID(self):
        with self.assertRaises(KeyError):
            ImageDataSet(
                str(self.data_path),
                verbose=True,
                id_globber=self._id_globber,
                raise_id_duplication=True)

    def test_get_properties(self):
        self._logger.debug(f"Size: {self.data.get_size(0) = }")
        self._logger.debug(f"Origin: {self.data.get_origin(0) = }")
        self._logger.debug(f"Direction: {self.data.get_direction(0) = }")
        self._logger.debug(f"Orientation: {self.data.get_verbose_orientation(0) = }")

    def test_get_data(self):
        dat = self.data['MRI_01']
        self.assertIsInstance(dat, self._expected_class)

    def test_get_source_path(self):
        path = self.data.data_source_path.iloc[0]
        self.assertIsInstance(path, Path)

class Test_DataLabel(Test_PMIData):
    def __init__(self, *args, **kwargs):
        super(Test_DataLabel, self).__init__(*args, **kwargs)

    def setUp(self):
        from pytorch_med_imaging.pmi_data import DataLabel
        super(Test_DataLabel, self).setUp()
        self.data_path = Path("./sample_data/sample_class_gt.csv")
        self.data = DataLabel(str(self.data_path))

    def test_set_target_col(self):
        self.data.set_target_column('Class', int)
        dat = self.data[0]
        self.assertIsInstance(dat, self._expected_class)

    def test_set_multi_col(self):
        # Differnet types
        self.data.set_target_column(['Class', 'Class_2'], (int, str))
        dat = self.data[0]
        self._logger.debug(f"{dat = }")
        self.assertIsInstance(dat, np.ndarray) # torch.Tensor does not support multi types
        self.assertTupleEqual(tuple(dat.shape), (1, 2))

        # Same types
        self.data.set_target_column(['Class', 'Class_2'], int)
        dat = self.data[0]
        self.assertIsInstance(dat, self._expected_class)
        self.assertTupleEqual(tuple(dat.shape), (1, 2))



class Test_DataLabelConcat(Test_PMIData):
    def __init__(self, *args, **kwargs):
        super(Test_DataLabelConcat, self).__init__(*args, **kwargs)
        self._expected_class = str

    def setUp(self):
        super(Test_DataLabelConcat, self).setUp()
        from pytorch_med_imaging.pmi_data import DataLabelConcat
        self.data = DataLabelConcat("./sample_data/sample_concat_df.xlsx")
        pass

class Test_ImageDataSetMC(Test_PMIData):
    def __init__(self, *args, **kwargs):
        super(Test_ImageDataSetMC, self).__init__(*args, **kwargs)
        self._expected_class = torch.Tensor

    def setUp(self):
        from pytorch_med_imaging.pmi_data import ImageDataMultiChannel
        super(Test_ImageDataSetMC, self).setUp()
        self.data_path = Path("./sample_data/img")
        self.seg_path = Path("./sample_data/seg")
        self.data = ImageDataMultiChannel(str(self.data_path.parent),
                                          channel_subdirs=[self.data_path.name, self.data_path.name],
                                          verbose=True, id_globber=self._id_globber)
        self.data_segment = ImageDataMultiChannel(str(self.seg_path.parent),
                                         channel_subdirs=[self.seg_path.name, self.seg_path.name],
                                         verbose=True, id_globber=self._id_globber)

    def test_get_data(self):
        data = self.data[0]
        self.assertEqual(data.shape[0], 2)

    def test_get_data_dimension(self):
        data = self.data[1]
        self.assertTupleEqual(tuple(data.shape), (2, 124, 256, 256))

    def test_get_multipledata(self):
        data = super().test_get_multipledata()
        self.assertTupleEqual(tuple(data.shape), (2, 2, 124, 256, 256))