from MedImgDataset import ImageDataSet
from random import shuffle
from shutil import copy2
import os
import numpy as np


def GenerateKFoldBatch(sourcedir, targetdir, numOfTestSamples):
    images = ImageDataSet(sourcedir + "/Image", verbose=True, dtype=np.uint8)
    labels = ImageDataSet(sourcedir + "/Label", verbose=True, dtype=np.uint8)
    smoothed = ImageDataSet(sourcedir + "/Smoothed", verbose=True, dtype=np.uint8)

    indexes = range(len(images))
    shuffle(indexes)
    indexes = np.array(indexes, dtype=int)
    indexes = np.pad(indexes, [(0, numOfTestSamples - len(images) % numOfTestSamples)], 'constant', constant_values=0)
    indexes = indexes.reshape(len(indexes)/numOfTestSamples, numOfTestSamples)

    # d1 for testing, d2 for training
    if not os.path.isdir(targetdir):
        os.mkdir(targetdir)

    # Discard last batch
    row = [images, labels, smoothed]
    for i in xrange(indexes.shape[0] - 1):
        if not os.path.isdir(targetdir + "/%03d"%i):
            os.mkdir(targetdir + "/%03d"%i)
        if not os.path.isdir(targetdir + "/%03d/Testing"%i):
            os.mkdir(targetdir + "/%03d/Testing"%i)
        if not os.path.isdir(targetdir + "/%03d/Training"%i):
            os.mkdir(targetdir + "/%03d/Training"%i)
        for subdir in ["Image", "Label", "Smoothed"]:
            if not os.path.isdir(targetdir + "/%03d/Testing/%s"%(i, subdir)):
                os.mkdir(targetdir + "/%03d/Testing/%s"%(i, subdir))
            if not os.path.isdir(targetdir + "/%03d/Training/%s"%(i, subdir)):
                os.mkdir(targetdir + "/%03d/Training/%s"%(i, subdir))

        for j in xrange(len(images)):
            if j in indexes[i]:
                copy2(row[0].dataSourcePath[j], targetdir + "/%03d/Testing/Image"%i)
                copy2(row[1].dataSourcePath[j], targetdir + "/%03d/Testing/Label"%i)
                copy2(row[2].dataSourcePath[j], targetdir + "/%03d/Testing/Smoothed"%i)
            else:
                copy2(row[0].dataSourcePath[j], targetdir + "/%03d/Training/Image"%i)
                copy2(row[1].dataSourcePath[j], targetdir + "/%03d/Training/Label"%i)
                copy2(row[2].dataSourcePath[j], targetdir + "/%03d/Training/Smoothed"%i)

if __name__ == '__main__':
    GenerateKFoldBatch("./BrainVessel/01.BatchSource", "./BrainVessel/10.K_Fold_Batches", 10)

