import numpy as np
import SimpleITK as sitk
import pandas as pd
import os
import numpy as np
from utils import get_unique_IDs, get_fnames_by_IDs
from tqdm import *

def crop_image(im, center, size):
    in_imsize = im.GetSize()
    lower_bound = [int(c - s//2) for c, s in zip(center, size)]
    upper_bound = [int(ori_s - c - np.ceil(s/2.)) for c, s, ori_s in zip(center, size, in_imsize)]

    print(lower_bound, upper_bound, center)
    cropper = sitk.CropImageFilter()
    cropper.SetLowerBoundaryCropSize(lower_bound)
    cropper.SetUpperBoundaryCropSize(upper_bound)

    outim = cropper.Execute(im)
    return outim


if __name__ == '__main__':
    SRC_DIR = '../NPC_Segmentation/42.Benign_Malignant_Upright/nyul_normed'
    OUT_DIR = '../NPC_Segmentation/44.Benign_Malignant_Cropped_Largest.2/'
    cropsize = [444, 444, 20]
    os.makedirs(OUT_DIR, exist_ok=True)
    min_x, min_y = np.array(cropsize[:2]) / 2
    max_x, max_y = 512 - min_x, 512 - min_y


    df = pd.read_csv('../NPC_Segmentation/42.Benign_Malignant_Upright/Z.DataSheet/center_of_nasopharynx.csv', index_col=0)
    dataID = df.index.to_list()

    fnames = get_fnames_by_IDs(os.listdir(SRC_DIR),
                               dataID)
    for d in tqdm(dataID):
        fn = fnames[d][0]
        if fn is None:
            print(d, "WRONG!")
            continue
        row = df.loc[d]
        center = [np.clip(row['Coord_X'], min_x, max_x),
                  np.clip(row['Coord_Y'], min_y, max_y), row['Slice Number']]


        im = sitk.ReadImage(os.path.join(SRC_DIR, fn))
        cropped = crop_image(im, center, cropsize)

        sitk.WriteImage(cropped, os.path.join(OUT_DIR, fn))



