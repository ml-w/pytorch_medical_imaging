import logging
import traceback
import os, sys, traceback
import hashlib
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

    def __init__(self, log_dir, logger_name=__name__, verbose=False, log_level='debug', keep_file=True):
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
        self._warning_hash = {}
        self._keepfile = keep_file

        log_levels={
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR
        }
        assert log_level in log_levels, "Expected argument log_level in one of {}, got {} instead.".format(
            list(log_levels.keys()), log_level
        )

        # Check and create directory for log
        try:
            os.makedirs(os.path.dirname(log_dir), exist_ok=True)
        except:
            pass
        self._log_dir = os.path.abspath(log_dir)


        self._logger = logging.getLogger(logger_name)
        formatter = logging.Formatter("[%(asctime)-12s-%(levelname)s] (%(name)s) %(message)s")
        handler = logging.FileHandler(log_dir)
        handler.setFormatter(formatter)

        stream_handler = TqdmLoggingHandler(verbose=verbose)
        stream_handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.addHandler(stream_handler)
        self._logger.setLevel(level=log_levels[log_level])

        self.info("Loging to file at: {}".format(os.path.abspath(log_dir)))

        # First logger created is the global logger.
        if Logger.global_logger is None:
            Logger.global_logger = self
            Logger.all_loggers[logger_name] = self
            sys.excepthook= self.exception_hook
            self.info("Exception hooked to this logger.")


    def log_traceback(self):
        self.exception()

    def log_print(self, msg, level=logging.INFO):
        self._logger.log(level, msg)
        # if self._verbose:
        #     print(msg)

    def log_print_tqdm(self, msg, level=logging.INFO):
        self._logger.log(level, msg)
        # if self._verbose:
        #     tqdm.write(msg)

    def info(self, msg):
        self.log_print_tqdm(msg, level=logging.INFO)

    def debug(self, msg):
        self.log_print_tqdm(msg, level=logging.DEBUG)

    def warning(self, msg: str, no_repeat=False):
        if no_repeat:
            h = hashlib.md5(msg.encode('utf-8')).hexdigest()
            if not h in self._warning_hash:
                self.log_print_tqdm(msg, level=logging.WARNING)
                self.log_print_tqdm("Warning message won't be shown again in this run",
                                    level=logging.WARNING)
                self._warning_hash[h] = 1
        else:
            self.log_print_tqdm(msg, level=logging.WARNING)

    def error(self, msg):
        self.log_print_tqdm(msg, level=logging.ERROR)

    def critical(self, msg):
        self.log_print_tqdm(msg, level=logging.CRITICAL)

    def exception(self, msg=""):
        self._logger.exception(msg)


    def exception_hook(self, *args):
        self.error('Uncaught exception:', exc_info=args)
        self.exception(args[-1])

    def __class_getitem__(cls, item):
        if cls.global_logger is None:
            cls.global_logger = Logger('./default.log', logger_name='default', verbose=True)
            return Logger[item]

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


    def __del__(self):
        if not self._keepfile & self != Logger.get_global_logger():
            os.remove(self._log_dir)


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET, verbose=False):
        super().__init__(level)
        self.verbose = verbose

    def emit(self, record):
        try:
            msg = self.format(record)
            if self.verbose:
                tqdm.write(msg)
                self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

