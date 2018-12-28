from random import shuffle
import numpy as np



def GenerateTestBatch(gt_files, input_files, numOfTestSamples, outdir, prefix="Batch_"):
    # assert len(gt_files) == len(input_files)
    # assert len(gt_files) > numOfTestSamples

    indexes = range(len(gt_files))
    shuffle(indexes)

    # check if outdir exist
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        assert os.path.isdir(outdir), "Cannot create director!"

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

        target['input'].extend(fnmatch.filter(input_files, gt_files[indexes[i]].split('_')[0] + "*"))
        target['gt'].append(gt_files[indexes[i]])

    testing_gt.writelines([f + '\n' for f in testing_samples['gt']])
    testing_input.writelines([f + '\n' for f in testing_samples['input']])

    training_gt.writelines([f + '\n' for f in training_samples['gt']])
    training_input.writelines([f + '\n' for f in training_samples['input']])

    [f.close() for f in [testing_gt, testing_input, training_input, training_gt]]



if __name__ == '__main__':
    import os, fnmatch
    # GenerateKFoldBatch("./BrainVessel/01.BatchSource", "./BrainVessel/10.K_Fold_Batches", 10)
    GenerateTestBatch(os.listdir('../NPC_Segmentation/02.NPC_seg'),
                      os.listdir('../NPC_Segmentation/01.NPC_dx'),
                      20,
                      '../NPC_Segmentation/99.Testing',
                      prefix="B01_"
                      )

