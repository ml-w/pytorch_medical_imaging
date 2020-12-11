from .PMIImagePatchesLoader import PMIImagePatchesLoader
from .PMIImageFeaturePair import PMIImageFeaturePair
from .PMIImageDataLoader import PMIImageDataLoader
from .PMIImageMCFeaturePair import PMIImageMCFeaturePair
from pytorch_med_imaging.logger import Logger
import traceback as tr

__all__ = ['PMIDataFactory']

class PMIDataFactory(object):
    def __init__(self):
        self._possible_products = {
            'PMIImageDataLoader': PMIImageDataLoader,
            'PMIImagePatchesLoader': PMIImagePatchesLoader,
            'PMIImageFeaturePair': PMIImageFeaturePair,
            'PMIImageMCFeaturePair': PMIImageMCFeaturePair
        }

        self._logger = Logger[__class__.__name__]


    def produce_object(self, config):
        """
        Use this to produce a dataset loader.

        Args:
            config (configparser.ConfigParser):
                This is the same as the config file you loaded in the main thread.

        Returns:
            product (PMIDataLoaderBase)
        """
        requested_datatype = config['LoaderParams']['PMI_datatype_name']
        run_mode = config['General'].get('run_mode', 'training')
        debug = config['General'].getboolean('debug', False)
        self._logger.log_print_tqdm("Creating object: {}".format(requested_datatype))

        try:
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
            Logger.Log_Print("Error when creating object {}".format(requested_datatype), 40)
            Logger.Log_Print("Original error is {}.".format(tr.print_tb(tb)))
            Logger.Log_Print("Possible targets are {}.".format(",".join(self._possible_products.keys())))
            raise e