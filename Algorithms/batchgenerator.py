from MedImgDataset import ImageDataSet, Projection
from random import shuffle
import fnmatch
import os
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

def GenerateKFoldBatch(GTfiles, targetdir, numOfTestSamples):
    files = sourcedir

    indexes = range(len(images))
    shuffle(indexes)
    indexes = np.array(indexes, dtype=int)
    indexes = np.pad(indexes, [(0, numOfTestSamples - len(images) % numOfTestSamples)], 'constant', constant_values=0)
    indexes = indexes.reshape(len(indexes)/numOfTestSamples, numOfTestSamples)

    # d1 for testing, d2 for training
    if not os.path.isdir(targetdir):
        os.mkdir(targetdir)

    # last batch has more data
    for i in xrange(indexes.shape[0] - 1):
        testlist = 0




if __name__ == '__main__':
    import os, fnmatch
    # GenerateKFoldBatch("./BrainVessel/01.BatchSource", "./BrainVessel/10.K_Fold_Batches", 10)
    GenerateTestBatch(os.listdir('../DFB_Recon/10.GT_Subbands'),
                      os.listdir('../DFB_Recon/11.SparseView_subbands'),
                      15,
                      '../DFB_Recon/99.Testing/Batch',
                      prefix="B01_"
                      )

