from MedImgDataset.ImageData2D import ImageDataSet2D
from MedImgDataset.Landmarks import  Landmarks
from MedImgDataset.ImageFeaturePair import ImageFeaturePair
from torch.utils.data import DataLoader
from random import shuffle
from shutil import copy2
import os
import pandas as pd
import numpy as np


def GenerateKFoldBatch(sourcedir, targetdir, numOfTestSamples, landmarks=True):
    if landmarks:
        L = Landmarks(sourcedir + "/Landmarks.csv")
    images = ImageDataSet2D(sourcedir, verbose=True)

    indexes = range(len(images))
    shuffle(indexes)
    indexes = np.array(indexes, dtype=int)
    indexes = np.pad(indexes, [(0, numOfTestSamples - len(images) % numOfTestSamples)], 'constant', constant_values=0)
    indexes = indexes.reshape(len(indexes)/numOfTestSamples, numOfTestSamples)

    # d1 for testing, d2 for training
    if not os.path.isdir(targetdir):
        os.mkdir(targetdir)

    # Discard last batch
    for i in xrange(indexes.shape[0] - 1):
        if landmarks:
            d1, d2 = [pd.DataFrame(columns=L.d.columns) for k in xrange(2)]

        if not os.path.isdir(targetdir + "/%03d"%i):
            os.mkdir(targetdir + "/%03d"%i)
        if not os.path.isdir(targetdir + "/%03d/Testing"%i):
            os.mkdir(targetdir + "/%03d/Testing"%i)
        if not os.path.isdir(targetdir + "/%03d/Training"%i):
            os.mkdir(targetdir + "/%03d/Training"%i)

        for j in xrange(len(images)):
            if j in indexes[i]:
                copy2(images.dataSourcePath[j], targetdir + "/%03d/Testing"%i)
                if landmarks:
                    d1 = d1.append(L.d.loc[j])
            else:
                copy2(images.dataSourcePath[j], targetdir + "/%03d/Training"%i)
                if landmarks:
                    d2 = d2.append(L.d.loc[j])

        if landmarks:
            d1.to_csv(targetdir + "/%03d/Testing/Landmarks.csv"%i, index=False)
            d2.to_csv(targetdir + "/%03d/Training/Landmarks.csv"%i, index=False)




if __name__ == '__main__':
    # GenerateKFoldBatch("../TOCI/54.BatchSource_Thumb", "../TOCI/55.K_Fold_Thumb", 100, False)
    # GenerateKFoldBatch("../TOCI/70.BatchSource_Thumb_NoAUG", "../TOCI/71.K_Fold_Thumb_NoAug", 100, False)
    pass