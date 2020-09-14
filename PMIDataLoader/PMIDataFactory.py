from .PMIImagePatchesLoader import PMIImagePatchesLoader
from .PMIImageFeaturePair import PMIImageFeaturePair
from .PMIImageDataLoader import PMIImageDataLoader
from .PMIImageMCFeaturePair import PMIImageMCFeaturePair
from logger import Logger

class PMIDataFactory(object):
    def __init__(self):
        self._possible_products = {
            'PMIImageDataLoader': PMIImageDataLoader,
            'PMIImagePatchesLoader': PMIImagePatchesLoader,
            'PMIImageFeaturePair': PMIImageFeaturePair,
            'PMIImageMCFeaturePair': PMIImageMCFeaturePair
        }


    def produce_object(self, config):
        requested_datatype = config['LoaderParams']['PMI_datatype_name']
        run_mode = config['General'].get('run_mode', 'training')
        debug = config['General'].getboolean('debug', False)
        logger = Logger.get_global_logger()
        logger.log_print_tqdm("Creating object: {}".format(requested_datatype))

        try:
            product = eval(requested_datatype)(config,
                                               run_mode,
                                               debug=debug,
                                               verbose=True,
                                               logger=logger)
            return product
        except Exception as e:
            Logger.Log_Print("Error when creating object {}".format(requested_datatype), 40)
            Logger.Log_Print("Original error is {}.".format(e))
            Logger.Log_Print("Possible targets are {}.".format(",".join(self._possible_products.keys())))
            raise e
