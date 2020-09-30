import os
from abc import abstractmethod
from pytorch_med_imaging.logger import Logger

class InferencerBase(object):
    def __init__(self, inferencer_configs):
        super(InferencerBase, self).__init__()

        # required
        self._in_dataset        = inferencer_configs['indataset']
        self._net               = inferencer_configs['net'] # note that this should be callable
        self._net_state_dict    = inferencer_configs['netstatedict']
        self._batchsize         = inferencer_configs['batchsize']
        self._iscuda            = inferencer_configs['iscuda']
        self._outdir            = inferencer_configs['outdir']
        assert os.path.isfile(self._net_state_dict), "Cannot open network checkpoint!"

        # optional
        self._logger = inferencer_configs['logger'] if 'logger' in inferencer_configs else None
        assert isinstance(self._logger, Logger) or self._logger is None, "Incorrect logger."

        if 'target_data' in inferencer_configs:
            self._target_dataset = inferencer_configs['target_data']
            self._TARGET_DATASET_EXIST_FLAG = True
        else:
            self._TARGET_DATASET_EXIST_FLAG = False

        self._input_check()
        self._create_net()
        self._create_dataloader()


    def get_net(self):
        return self._net

    @abstractmethod
    def _input_check(self):
        raise NotImplementedError

    @abstractmethod
    def _create_net(self):
        raise NotImplementedError

    @abstractmethod
    def _create_dataloader(self):
        raise NotImplementedError

    @abstractmethod
    def display_summary(self):
        raise NotImplementedError