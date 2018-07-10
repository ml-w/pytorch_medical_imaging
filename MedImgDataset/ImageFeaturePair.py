from torch.utils.data import Dataset
import torch
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

class ImageFeaturePair(Dataset):
    """
    Data set wrapping like Tensor Dataset, except this also accept a mask.
    """

    def __init__(self, image_Dataset, landmarkDataset, resize_to_square=-1):
        assert len(image_Dataset) == len(landmarkDataset)

        self.image_dataset = image_Dataset
        self.landmarks_dataset = landmarkDataset
        self._resize = resize_to_square

    def __getitem__(self, index):
        if self._resize < 0:
            return self.image_dataset[index], self.landmarks_dataset[index]
        else:
            return self.ResizeToSquare(self.image_dataset[index], self.landmarks_dataset[index])

    def __len__(self):
        return len(self.image_dataset)

    def __str__(self):
        return str(self.image_dataset) + str(self.landmarks_dataset)

    def ResizeToSquare(self, imagedata, landmarks):
        # Get data
        im = imagedata.numpy()
        pts = imagedata.numpy()
        pts = [ia.Keypoint(x=p[1], y=p[0]) for p in pts]
        pts = [ia.KeypointsOnImage(p, shape=tuple(im.shape)) for p in pts]

        # seq = iaa.Sequential([iaa.Pad(px=(pu, pr, pd, pl)),
        seq = iaa.Sequential([iaa.Scale({'height':self._resize, "width": "keep-aspect-ratio"})]) # top, right, bottom, left
        seq_det = seq.to_deterministic()
        im_aug = seq_det.augment_image(im)
        keypoints_aug = seq_det.augment_keypoints(pts)
        keypoints_aug = np.stack([[[p.y, p.x] for p in keypoints_aug[i].keypoints] for i in xrange(len(keypoints_aug))])
        seq_det.augment_keypoints()
        return torch.from_numpy(im_aug), torch.from_numpy(keypoints_aug)

