import os
from functools import partial

import numpy as np
import configparser

from MedImgDataset import *
from MedImgDataset.Computation import *


# DataLoaders
def LoadClassificationDataSet(a, debug=False):
    image = lambda input, fsuffix, filelist, dtype: ImageDataSet(input,
                                                      dtype=dtype,
                                                      verbose=True,
                                                      debugmode=debug,
                                                      filesuffix=fsuffix,
                                                      idlist=filelist)
    classes = lambda fname: DataLabel.from_csv(fname)

    if a.train is None:
        imset = image(a.input, a.lsuffix, a.loadbyfilelist, np.float32)
        return imset
    else:
        # Training Mode
        imset = image(a.input, a.lsuffix, a.loadbyfilelist, np.float32)
        gtset = classes(a.train)
        gtset.set_target_column('Benign')
        gtset.map_to_data(imset, target_id_globber="(?i)(NPC|P)?[0-9]{3,5}")
        return imset, gtset






datamap = {'imgclassification': LoadClassificationDataSet,
           'imgclassification_debug': partial(LoadClassificationDataSet, debug=True)
           }
