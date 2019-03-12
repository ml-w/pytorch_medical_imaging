from random import shuffle
import numpy as np



def GenerateTestBatch(gt_files, input_files, numOfTestSamples, outdir, prefix="Batch_", exclude_list=None):
    # assert len(gt_files) == len(input_files)
    # assert len(gt_files) > numOfTestSamples

    indexes = range(len(gt_files))
    shuffle(indexes)

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

    # Testing batch files
    testing_gt = open(outdir + '/' + prefix + "Testing_GT.txt", "w")
    testing_input = open(outdir + '/' + prefix + "Testing_Input.txt", "w")

    # Training batch files
    training_gt = open(outdir + '/' + prefix + "Training_GT.txt", 'w')
    training_input = open(outdir + '/' + prefix + "Training_Input.txt", 'w')

    training_samples = {'input': [], 'gt':  []}
    testing_samples = {'input': [], 'gt': []}
    for i in xrange(len(indexes)):
        if i < numOfTestSamples:
            target = testing_samples
        else:
            target = training_samples

        # each case should be named "[ID]_details.nii.gz"
        case_id = gt_files[indexes[i]].split('_')[0]

        # skip if this case is not to be included
        if case_id in exclude:
            print "Skipping case: %s"%case_id
            continue

        in_tar = fnmatch.filter(input_files, case_id + "*")
        if len(in_tar) < 1:
            print "Cannot find input files for case: %s"%case_id
            continue

        target['input'].extend(in_tar)
        target['gt'].append(gt_files[indexes[i]])

    testing_gt.writelines([f + '\n' for f in testing_samples['gt']])
    testing_input.writelines([f + '\n' for f in testing_samples['input']])

    training_gt.writelines([f + '\n' for f in training_samples['gt']])
    training_input.writelines([f + '\n' for f in training_samples['input']])

    [f.close() for f in [testing_gt, testing_input, training_input, training_gt]]



if __name__ == '__main__':
    import os, fnmatch
    # GenerateKFoldBatch("./BrainVessel/01.BatchSource", "./BrainVessel/10.K_Fold_Batches", 10)
    GenerateTestBatch(os.listdir('../NPC_Segmentation/05.NPC_seg_T2'),
                      os.listdir('../NPC_Segmentation/01.NPC_dx'),
                      50,
                      '../NPC_Segmentation/99.Testing',
                      prefix="B02/B02_",
                      exclude_list='../NPC_Segmentation/99.Testing/B02/exclude.txt'
                      )

