from .pmi_img_feat_pair_dataloader import PMIImageFeaturePair
from .pmi_image_dataloader import PMIImageDataLoader
from mnts.mnts_logger import MNTSLogger
import traceback as tr
import re

__all__ = ['PMIDataFactory']

class PMIDataFactory(object):
    def __init__(self):
        self._possible_products = {
            'PMIImageDataLoader': PMIImageDataLoader,
            'PMIImageFeaturePair': PMIImageFeaturePair,
        }

        self._logger = MNTSLogger[__class__.__name__]


    def produce_object(self, config, run_mode=None):
        """
        Use this to produce a dataset loader.

        Attributes:
            ('General','forece_train_data')
                If this tag is True in config, this will load the training data regardless of other specifications.

        Args:
            config (configparser.ConfigParser):
                This is the same as the config file you loaded in the main thread.
            run_mode (str):
                'train' or 'inference'.

        Returns:
            product (PMIDataLoaderBase)
        """
        requested_datatype = config['LoaderParams']['PMI_datatype_name']
        if run_mode is None:
            run_mode = config['General'].get('run_mode', 'training')
        # Force loading training data
        force_train_data = config['General'].getboolean('force_train_data', False)
        if force_train_data:
            self._logger.info("Force loading training dataset!")
            run_mode = 'training'
        debug = config['General'].getboolean('debug', False)
        self._logger.log_print_tqdm("Creating object: {}".format(requested_datatype))

        try:
            if re.search("[\W]+", requested_datatype.translate(str.maketrans('', '', "(), "))) is not None:
                raise AttributeError(f"You requested_datatype specified ({requested_datatype}) "
                                     f"contains illegal characters!")
            if requested_datatype not in self._possible_products:
                msg += f"Expect requested_datatype to be one of the followings: " \
                       f"[{'|'.join(self._possible_products.keys())}], " \
                       f"but got {requested_datatype}."
                raise AttributeError(msg)

            product = eval(requested_datatype)(config,
                                               run_mode,
                                               debug=debug,
                                               verbose=True
                                               )
            product.class_name = requested_datatype
            return product
        except Exception as e:
            import sys
            cl, exc, tb = sys.exc_info()
            self._logger.info("Error when creating object {}".format(requested_datatype))
            self._logger.info("Possible targets are {}.".format(",".join(self._possible_products.keys())))
            raise e
