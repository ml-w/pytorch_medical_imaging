import tempfile
import unittest
from pathlib import Path

import torch
from mnts.mnts_logger import MNTSLogger
from pytorch_med_imaging.med_img_dataset import *


class Test_PMIData(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_PMIData, self).__init__(*args, **kwargs)
        pass

    def setUp(self):
        if self.__class__.__name__ == 'Test_PMIData':
            raise unittest.SkipTest("Base class")

        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)

        self._logger = MNTSLogger(self.temp_dir.name + "/log",
                                  logger_name='unittest', verbose=True, keep_file=False, log_level='debug')

        self._idGlobber = "MRI_\d+"
        pass

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_getUniqueIDs(self):
        ids = self.data.get_unique_IDs(self._idGlobber)
        self.assertEqual(ids, [f"MRI_0{i+1}" for i in range(len(self.data))])

    def test_getDataByID(self):
        data = self.data.get_data_by_ID("MRI_02")
        self.assertTrue(torch.allclose(data, self.data[1]))    # Test also the sorting order is the same

    def test_getDataByIndex(self):
        data = self.data[0]

    @classmethod
    def get_class_name(cls):
        return cls.__class__.__name__

class Test_ImageDataSet(Test_PMIData):
    def __init__(self, *args, **kwargs):
        super(Test_ImageDataSet, self).__init__(*args, **kwargs)

    def setUp(self):
        from pytorch_med_imaging.med_img_dataset import ImageDataSet
        super(Test_ImageDataSet, self).setUp()
        self.data_path = Path("./sample_data/img")
        self.seg_path = Path("./sample_data/seg")
        self.data = ImageDataSet(str(self.data_path), verbose=True, idGlobber=self._idGlobber)
        self.data_segment = ImageDataSet(str(self.seg_path), verbose=True, idGlobber=self._idGlobber)

    def test_matchID(self):
        self._logger.debug(f"{self.data.get_unique_IDs()}")
        self._logger.debug(f"{self.data_segment.get_unique_IDs()}")
        self.assertTrue(self.data.get_unique_IDs() == self.data_segment.get_unique_IDs())



class Test_DataLabel(Test_PMIData):
    def __init__(self, *args, **kwargs):
        super(Test_DataLabel, self).__init__(*args, **kwargs)

    def setUp(self):
        from pytorch_med_imaging.med_img_dataset import DataLabel
        super(Test_DataLabel, self).setUp()
        self.data_path = Path("./sample_data/sample_class_gt.csv")
        self.data = DataLabel(str(self.data_path))



