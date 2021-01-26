from pytorch_med_imaging.med_img_dataset import ImageDataSet
from pytorch_med_imaging.logger import Logger
import sys
import configparser
import argparse
import os

def main(args):
    for dir in [args.dirA, args.dirB]:
        assert os.path.isdir(dir)

    log = Logger(a.log, logger_name='utils.match_dimension', verbose=a.verbose)
    sys.excepthook = log.exception_hook

    imsetA = ImageDataSet(a.dirA, verbose=a.verbose, dtype='uint8', debugmode=a.debug)
    ida = imsetA.get_unique_IDs(a.globber)

    imsetB = ImageDataSet(a.dirB, verbose=a.verbose, dtype='uint8', filtermode='idlist', idlist=ida)
    idb = imsetB.get_unique_IDs(a.globber)

    size_a = [tuple(imsetA.get_data_by_ID(i).shape) for i in ida]
    size_b = [tuple(imsetB.get_data_by_ID(i).shape) for i in idb]

    missing = []
    for sa in ida:
        if not sa in idb:
            missing.append(sa)

    miss_match = {}
    for i, (sa, sb) in enumerate(zip(size_a, size_b)):
        if sa != sb:
            miss_match[ida[i]] = {'A': sa, 'B': sb}

    if len(miss_match) > 0:
        log.info(f"Miss-match list: {','.join(miss_match.keys())}\n")
        log.info(f"Size-list: \n")
        log.info([f"{idx}: {miss_match[idx]['A']} - {miss_match[idx]['B']} \n" for idx in miss_match])

        if len(missing) > 0:
            log.info("Missing: \n")
            log.info(f"{','.join([m for m in missing])}")

        if a.save.endswith('.txt'):
            log.info(f"Writing to: {a.save}")
            with open(a.save, 'w') as f:
                f.write(f"Miss-match list: {','.join(miss_match.keys())}\n")
                f.write(f"Size-list: \n")
                f.writelines([f"{idx}: {miss_match[idx]['A']} - {miss_match[idx]['B']} \n" for idx in miss_match])

                if len(missing) > 0:
                    f.write("\n")
                    f.write("Missing: \n")
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
        log.info("Check complete. No miss-match found.")

    # If everything goes smoothly, delete the log file
    log.info("Try exiting cleanly")
    del log
    if not a.saveLog:
        os.remove(a.log)

    pass

if __name__ == '__main__':
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
    a = parser.parse_args()
    main(a)

