import logging
import traceback
import os, sys
from tqdm import *

__all__ = ['Logger']

class Logger(object):
    def __init__(self, log_dir):
        super(Logger, self).__init__()
        self._log_dir = log_dir

        # Check and create directory for log
        os.makedirs(os.path.dirname(log_dir), exist_ok=True)


        self._logger = logging.getLogger(__name__)
        logging.basicConfig(format="[%(asctime)-12s-%(levelname)s] %(message)s", filename=log_dir, level=logging.DEBUG)
        sys.excepthook= self.exception_hook


    def log_print(self, msg, level=logging.INFO):
        self._logger.log(level, msg)
        print(msg)

    def log_print_tqdm(self, msg, level=logging.INFO):
        self._logger.log(level, msg)
        tqdm.write(msg)

    def exception_hook(self, *args):
        self._logger.error('Uncaught exception:', exc_info=args)
        traceback.print_tb(args[0])
