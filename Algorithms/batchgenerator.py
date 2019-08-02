from random import shuffle
import pandas as pd
import re, fnmatch

def GenerateFileList(files):
    regex_dict = {'T1W':        "((?=.*T1.*)(?!.*FS.*)(?!.*[cC].*))",
                  'CE-T1W':     "((?=.*T1.*)(?!.*FS.*)(?=.*[cC].*))",
                  'CE-T1W-FS':  "((?=.*T1.*)(?=.*FS.*)(?=.*[cC].*))",
                  'T2W-FS':     "((?=.*T2.*)(?=.*FS.*))"}

    # get unique indexes
    globber = "[0-9]+"
    outlist = []
    for f in files:
        matchobj = re.search(globber, f)
        if not matchobj is None:
            outlist.append(int(f[matchobj.start():matchobj.end()]))
    idlist = list(set(outlist))

    df = pd.DataFrame()
    for i in idlist:
        row = {'None': []}
        fs = fnmatch.filter(files, str(i) + '*')
        for ff in fs:
            FLAG = False
            for k in regex_dict:
                matchobj = re.match(regex_dict[k], ff)
                if not matchobj is None and not row.has_key(k):
                    row[k] = ff
                    FLAG = True
            if not FLAG:
                row['None'].append(ff)

        if len(row['None']) == 0:
            row['None'] = "NULL"
        else:
            row['None'] = '|'.join(row['None'])

        tmp = pd.DataFrame([row.values()], columns=row.keys(), index=[i])
        df = df.append(tmp)
    return df[regex_dict.keys() + ['None']]



def GenerateTestBatch(gt_files, input_files, numOfTestSamples, outdir, prefix="Batch_", exclude_list=None, k_fold=None):
    # assert len(gt_files) == len(input_files)
    # assert len(gt_files) > numOfTestSamples

    indexes = range(len(gt_files))
    shuffle(indexes)
    #
    # check if outdir exist
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        assert os.path.isdir(outdir), "Cannot create directory!"

    # Load list to exclude
    exclude = []
    if not exclude_list is None:
        if isinstance(exclude_list, list) or isinstance(exclude_list, tuple):
            for row in exclude_list:
                assert isinstance(row, str), "Must be list or tuple of string!"
                exclude.append(row.rstrip())
        elif isinstance(exclude_list, file):
            for row in exclude_list.readlines():
                if row[0] == '#':
                    continue
                exclude.append(row.rstrip())
        elif isinstance(exclude_list, str) and os.path.isfile(exclude_list):
            with open(exclude_list, 'r') as f:
                for row in f.readlines():
                    if row[0] == '#':
                        continue
                    exclude.append(row.rstrip())


    if k_fold is None:
        k_fold = 1
    else:
        numOfTestSamples = len(indexes) // k_fold

    for k in xrange(k_fold):
        fold_number = "_%03d_"%k
        # Testing batch files
        # testing_gt = open(outdir + '/' + prefix + fold_number + "Testing_GT.txt", "w")
        testing_input = open(outdir + '/' + prefix + fold_number + "Testing_Input.txt", "w")

        # Training batch files
        # training_gt = open(outdir + '/' + prefix + fold_number +  "Training_GT.txt", 'w')
        training_input = open(outdir + '/' + prefix + fold_number + "Training_Input.txt", 'w')

        training_samples = {'input': [], 'gt':  []}
        testing_samples = {'input': [], 'gt': []}
        for i in xrange(len(indexes)):
            if (i > k * numOfTestSamples and i < (k+1) * numOfTestSamples) or i > k_fold * numOfTestSamples:
                target = testing_samples
            else:
                target = training_samples

            # each case should be named "[ID]_details.nii.gz"
            matchobj = re.match('^[0-9]{3,5}', gt_files[indexes[i]])
            if not matchobj is None:
                case_id = gt_files[indexes[i]][matchobj.start():matchobj.end()]
            else:
                print "Skipping case: %s"%gt_files[indexes[i]]
                continue

            # skip if this case is not to be included
            if case_id in exclude:
                print "Skipping case: %s"%case_id
                continue

            in_tar = fnmatch.filter(input_files, case_id + "*")
            if len(in_tar) < 1:
                print "Cannot find input files for case: %s"%case_id
                continue

            target['input'].extend([case_id])
            target['gt'].append(case_id)

        # testing_gt.writelines([f + '\n' for f in testing_samples['gt']])
        testing_input.writelines([f + '\n' for f in testing_samples['input']])

        # training_gt.writelines([f + '\n' for f in training_samples['gt']])
        training_input.writelines([f + '\n' for f in training_samples['input']])

        [f.close() for f in [testing_input, training_input]]


def check_batches_files(dir):
    pass


if __name__ == '__main__':
    import os, fnmatch
    # GenerateKFoldBatch("./BrainVessel/01.BatchSource", "./BrainVessel/10.K_Fold_Batches", 10)
    GenerateTestBatch(os.listdir('../NPC_Segmentation/21.NPC_Perfect_SegT2/00.First'),
                      os.listdir('../NPC_Segmentation/06.NPC_Perfect'),
                      50,
                      '../NPC_Segmentation/99.Testing',
                      prefix="B05/B05",
                      k_fold=4
                      )

    # print GenerateFileList(os.listdir('../NPC_Segmentation/06.NPC_Perfect')).to_csv('~/FTP/temp/perfect_file_list.csv')

