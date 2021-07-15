from ..med_img_dataset import ImageDataSet
from ..logger import Logger
import sys
import configparser
import argparse
import os

__all__ = ['match_dimension']

def match_dimension(a):
    if a.ids is not None:
        if os.path.isfile(a.ids):
            ids = [r.rstrip() for r in open(a.ids, 'r').readlines()]
        elif a.ids.find(',') >= 0:
            ids = a.ids.split(',')
        else:
            ids = [a.ids]
    else:
        ids = None
    log = a.log

    if not ids is None:
        imsetA = ImageDataSet(a.dirA, verbose=a.verbose, dtype='uint8', debugmode=a.debug, filtermode='idlist',
                              idlist=ids)
    else:
        imsetA = ImageDataSet(a.dirA, verbose=a.verbose, dtype='uint8', debugmode=a.debug)
    ida = imsetA.get_unique_IDs(a.globber)

    imsetB = ImageDataSet(a.dirB, verbose=a.verbose, dtype='uint8', filtermode='idlist', idlist=ida)
    idb = imsetB.get_unique_IDs(a.globber)

    missing = []
    for sa in ida:
        if not sa in idb:
            missing.append(sa)
    log.debug(f"{missing}")

    # check if the two sets contains same number of elements
    if len(imsetA) != len(imsetB):
        log.warning("The images are not paired properly.")

    # check if pairing are correctly done
    if set(ida) != set(idb):
        log.warning("The ids are not paired properly. Using overlapping of the lists.")

    overlapId = list(set(ida) & set(idb))
    a_ids = imsetA.get_unique_IDs(globber=a.globber)
    b_ids = imsetB.get_unique_IDs(globber=a.globber)
    overlap_indices_a = [a_ids.index(i) for i in overlapId]
    overlap_indices_b = [b_ids.index(i) for i in overlapId]


    size_a = [tuple(imsetA.get_size(i)) for i in overlap_indices_a]
    size_b = [tuple(imsetB.get_size(i)) for i in overlap_indices_b]

    miss_match = {}
    for i, (sa, sb) in enumerate(zip(size_a, size_b)):
        if sa != sb:
            miss_match[overlapId[i]] = {'A': sa, 'B': sb}

    if len(miss_match) > 0:
        log.info(f"Miss-match list: {','.join(miss_match.keys())}")
        log.info(f"Size-list: ")
        log.info('\n' + '\n'.join([f"{idx}: {miss_match[idx]['A']} - {miss_match[idx]['B']}" for idx in miss_match]))

        if len(missing) > 0:
            log.warning("Missing in directory B!")
            log.info("Missing ids: \n" + f"{','.join([m for m in missing])}")

        if a.save.endswith('.txt'):
            log.info(f"Writing to: {a.save}")
            with open(a.save, 'w') as f:
                f.write(f"Miss-match list: {','.join(miss_match.keys())}\n")
                f.write(f"Size-list: \n")
                f.writelines([f"{idx}: {miss_match[idx]['A']} - {miss_match[idx]['B']} \n" for idx in miss_match])

                if len(missing) > 0:
                    f.write("\n")
                    f.write("directory B: \n")
                    f.writelines(f"{','.join([m for m in missing])}")

        elif a.save.endswith('.ini'):
            log.info(f"Writing to: {a.save}")
            cf = configparser.ConfigParser()
            cf.add_section('MatchDimension')
            cf['MatchDimension']['Dir_A'] = a.dirA
            cf['MatchDimension']['Dir_B'] = a.dirB
            cf['MatchDimension']['Missmatch list'] = ','.join(miss_match.keys())

            cf.add_section('Missmatch Sizes')
            for m in miss_match:
                cf['Missmatch Sizes'][m] = f"{miss_match[m]['A']},{miss_match[m]['B']}"
            cf.write(a.save)

    else:
        if len(missing) > 0:
            log.warning("Missing in directory B!")
            log.info("Missing ids: \n" + f"{','.join([m for m in missing])}")

        log.info("Check complete. No miss-match found.")

    # If everything goes smoothly, delete the log file
    log.info("Try exiting cleanly")
    log.info("{:=^100}".format(" Finished "))
    del log

def console_entry(*args, **kwargs):
    parser = argparse.ArgumentParser(
        description="Check and match the dimensions if the images in [dirA] and [dirB]")
    parser.add_argument('dirA', metavar='dirA', action='store', type=str,
                        help="Directory A.")
    parser.add_argument('dirB', metavar='dirB', action='store', type=str,
                        help="Directory B.")
    parser.add_argument('-i', '--ids', dest='ids', action='store', type=str, default=None,
                        help="Specify a .txt or input the csv string to only look at some IDs.")
    parser.add_argument('-g', '--globber', dest='globber', action='store', type=str, default=None,
                        help="Globber passed to ImageDataSet for ID globbing.")
    parser.add_argument('-s', '--save', dest='save', action='store', type=str, default='',
                        help="Save results as .ini or .txt file.")
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        help="Print verbose information."),
    parser.add_argument('-l', '--log', dest='log', action='store', default='./utils.log',
                        help="Log path.")
    parser.add_argument('-L', '--saveLog', dest='saveLog', action='store_true',
                        help='If true, program will not delete log file with clean exit.')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help="For debug.")
    a = parser.parse_args(*args, **kwargs)

    for dir in [a.dirA, a.dirB]:
        assert os.path.isdir(dir)

    logpath = a.log
    log = Logger(a.log, logger_name='scripts.match_dimension', verbose=a.verbose, keep_file=a.saveLog)
    log.info("{:=^100}".format(" Matching Dimensions "))
    sys.excepthook = log.exception_hook
    a.log = log

    match_dimension(a)

    if not a.saveLog:
        try:
            os.remove(logpath)
        except:
            pass

if __name__ == '__main__':
    console_entry()

