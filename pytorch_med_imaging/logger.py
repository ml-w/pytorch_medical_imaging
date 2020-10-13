import logging
import traceback
import os, sys
from tqdm import *

__all__ = ['Logger']

class Logger(object):
    global_logger = None
    all_loggers = {}
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    CRITICAL = logging.CRITICAL
    FATAL = logging.FATAL
    ERROR = logging.ERROR

    def __init__(self, log_dir, logger_name=__name__, verbose=False):
        """
        This is the logger. This is typically passed to all modules for logging. Use class method Logger['str'] to get a
        logger named 'str'.

        Args:
            log_dir (str):
                Filename of the log file.
            verbose (boolean, Optional):
                If True, messages will be printed to stdout alongside being logged. Default to False.

        Returns:
            :class:`Logger` object
        """

        super(Logger, self).__init__()
        self._log_dir = log_dir
        self._verbose = verbose

        # Check and create directory for log
        os.makedirs(os.path.dirname(log_dir), exist_ok=True)


        self._logger = logging.getLogger(logger_name)
        formatter = logging.Formatter("[%(asctime)-12s-%(levelname)s] (%(name)s) %(message)s")
        handler = logging.FileHandler(log_dir)
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(level=logging.DEBUG)

        self.info("Loging to file at: {}".format(os.path.abspath(log_dir)))
        sys.excepthook= self.exception_hook

        # First logger created is the global logger.
        if Logger.global_logger is None:
            Logger.global_logger = self
            Logger.all_loggers[logger_name] = self

    def log_traceback(self):
        self.exception()

    def log_print(self, msg, level=logging.INFO):
        self._logger.log(level, msg)
        if self._verbose:
            print(msg)

    def log_print_tqdm(self, msg, level=logging.INFO):
        self._logger.log(level, msg)
        if self._verbose:
            tqdm.write(msg)

    def info(self, msg):
        self.log_print_tqdm(msg, level=logging.INFO)

    def debug(self, msg):
        self.log_print_tqdm(msg, level=logging.DEBUG)

    def warning(self, msg):
        self.log_print_tqdm(msg, level=logging.WARNING)

    def error(self, msg):
        self.log_print_tqdm(msg, level=logging.ERROR)

    def critical(self, msg):
        self.log_print_tqdm(msg, level=logging.critical())

    def exception(self, msg=""):
        self._logger.exception(msg)

    def exception_hook(self, *args):
        self._logger.error('Uncaught exception:', exc_info=args)
        traceback.print_tb(args[0])

    def __class_getitem__(cls, item):
        if cls.global_logger is None:
            raise AttributeError("Global logger was not created.")
        elif not item in cls.all_loggers:
            cls.global_logger.log_print("Requesting logger [{}] not exist, creating...".format(
                str(item)
            ))
            cls.all_loggers[item] = Logger(cls.global_logger._log_dir,
                                          logger_name=str(item),
                                          verbose=cls.global_logger._verbose)
            return cls.all_loggers[item]
        else:
            return cls.all_loggers[item]


    @staticmethod
    def Log_Print(msg, level=logging.INFO):
        Logger.global_logger.log_print(msg, level)

    @staticmethod
    def Log_Print_tqdm(msg, level=logging.INFO):
        Logger.global_logger.log_print_tqdm(msg, level)


    @staticmethod
    def get_global_logger():
        if not Logger.global_logger is None:
            # Attempts to create a global logger
            Logger.global_logger = Logger('./default_logdir.log', logger_name='default_logname')
            return Logger.global_logger
        else:
            raise AttributeError("Global logger was not created.")