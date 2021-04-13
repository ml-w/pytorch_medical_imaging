import re
import os
from pytorch_med_imaging.Algorithms import batch_nii2dicom


def main():
    idlist = '../NPC_Segmentation/97.DatasetList/Scoring.txt'
    ids = [r.strip() for r in open(idlist, 'r').readlines()]

    out_dir = '../NPC_Segmentation/00.RAW/For_Scoring'
    os.makedirs(out_dir, exist_ok=True)

    in_dir2 = '../NPC_Segmentation/0A.NIFTI_ALL/Benign'
    in_dir = '../NPC_Segmentation/0A.NIFTI_ALL/Malignant'

    targets = []
    for base, root, files in os.walk(in_dir):
        if not len(files) == 0:
            # print(files)
            l_ids = [re.search("([a-zA-Z0-9]{3,6})", os.path.basename(f)).group()
                     for f in files]
            l_full_dirs = [os.path.join(base, f) for f in files]
            for k, v in zip(l_ids, l_full_dirs):
                if k in ids:
                    targets.append(v)

    for base, root, files in os.walk(in_dir2):
        if not len(files) == 0:
            # print(files)
            l_ids = [re.search("([a-zA-Z0-9]{3,6})", os.path.basename(f)).group()
                     for f in files]
            l_full_dirs = [os.path.join(base, f) for f in files]
            for k, v in zip(l_ids, l_full_dirs):
                if k in ids:
                    targets.append(v)

    batch_nii2dicom(targets, out_dir, blind_out=True)

if __name__ == '__main__':
    main()