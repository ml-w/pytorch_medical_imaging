# System
import argparse
import configparser
import datetime
import os
import gc
from pathlib import Path

# Propietary
from mnts.mnts_logger import MNTSLogger
from .pmi_controller import PMIController


# This package

def console_entry(raw_args=None):
    parser = argparse.ArgumentParser(description="Training reconstruction from less projections.")
    parser.add_argument("--config", metavar='config', action='store', required=True,
                        help="Config .ini file.", type=str)
    parser.add_argument("-t", "--train", dest='train', action='store_true', default=False,
                        help="Set this to force training mode. (Implementing)")
    parser.add_argument("-i", "--inference", dest="inference", action='store_true', default=False,
                        help="Set this to force inference mode. If used with -t option, will still go into inference. (Implementing")
    parser.add_argument("-b", "--batch-size", dest='batch_size', type=int, default=None,
                        help="Set this to override batch-size setting in loaded config.")
    parser.add_argument("-e", "--epoch", dest="epoch", type=int, default=None,
                        help="Set this to override number of epoch when loading config.")
    parser.add_argument("-l", "--lr", dest='lr', type=float, default=None,
                        help="Set this to override learning rate.")
    parser.add_argument('--network', type=str, default="",
                        help="Convenient port to update (Network,network_type)")
    parser.add_argument("--all-checkpoints", dest='inference_all_checkpoints', action='store_true',
                        help="Set this to inference all checkpoints.")
    parser.add_argument("--log-level", dest='log_level', type=str, choices=('debug', 'info', 'warning','error'),
                        default='info', help="Set log-level of the logger.")
    parser.add_argument("--keep-log", action='store_true',
                        help="If specified, save the log file to the `log_dir` specified in the config.")
    parser.add_argument('--debug', dest='debug', action='store_true', default=None,
                        help="Set this to initiate the config with debug setting.")
    parser.add_argument('--debug-validation', action='store_true',
                        help="Set this to true to run validation direction. This also sets --debug to true.")
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help="Print message to stdout.")
    parser.add_argument('--fold-code', action='store', default=None,
                        help="Convenient port to update (General,fold_code).")
    parser.add_argument('--override', dest='override', action='store', type=str, default='',
                        help="Use syntax '(section1,key1)=value1;(section2,key2)=value' to override any"
                             "settings specified in the config file. Note that no space is allowed.")

    a = parser.parse_args(raw_args)

    assert os.path.isfile(a.config), f"Cannot find config file {a.config}! Curdir: {os.listdir('.')}"

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(a.config)

    # Override config settings, move this into override?
    pre_log_message = []

    # Parameters check
    log_dir = config['General'].get('log_dir', './Backup/Log/')
    keep_log = config['General'].getboolean('keep_log', False)
    if not Path(log_dir).parent.is_dir():
        Path(log_dir).parent.mkdir(parents=True, exist_ok=True)
        pass
    if os.path.isdir(log_dir):
        print(f"Log file not designated, creating under {log_dir}")
        log_dir = os.path.join(log_dir, "%s_%s.log"%(config['General'].get('run_mode', 'training'),
                                                     datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    print(f"Log designated to {log_dir}")
    print(f"Fullpath: {os.path.abspath(log_dir)}")
    with MNTSLogger(log_dir, logger_name='pmi-main', verbose=a.verbose, keep_file=keep_log,
                    log_level='debug' if a.debug else 'debug') as logger:
        logger.info("Global logger: {}".format(logger))

        for msg in pre_log_message:
            logger.info(msg)

        logger.info(">" * 40 + " Start Main " + "<" * 40)
        try:
            main = PMIController(config, a)
            main.run()
        except Exception as e:
            logger.error("Uncaught exception!")
            logger.exception(e)
            raise BrokenPipeError("Unexpected error in main().")
        finally:
            gc.collect() # Sometimes the CUDA memory is occupied
        logger.info("=" * 40 + " Done " + "="* 40)

if __name__ == '__main__':
    console_entry()