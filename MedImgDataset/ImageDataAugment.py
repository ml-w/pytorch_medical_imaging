from .ImageData import ImageDataSet
from torchvision import transforms as tr
from imgaug import augmenters as iaa

# This prevents DataLoader hangs up
# import cv2
# cv2.setNumThreads(0)

import imgaug as ia
import torch
import numpy as np
import gc

class ImageDataSetAugment(ImageDataSet):
    def __init__(self, *args, **kwargs):
        # TODO: Allow reading augmentation parameters
        self._augment_factor = kwargs.pop('aug_factor') if 'aug_factor' in kwargs else 5
        self._is_segment = kwargs.pop('is_seg') if 'is_seg' in kwargs else False
        self._references_dataset = kwargs.pop('reference_dataset') if 'reference_dataset' in kwargs \
            else None
        self._update_each_epoch = kwargs.pop('renew') if 'renew' in kwargs else False
        self._augmentator = kwargs.pop('augmentator') if 'augmentator' in kwargs else None
        super(ImageDataSetAugment, self).__init__(*args, **kwargs)
        assert self._byslices >= 0, "Currently only support slices augmentation."

        # Change the length of the dataset
        self._base_length = self.length
        self.length = self.length * (self._augment_factor + 1)
        self._nb_of_classes = len(np.unique(self.data.numpy()))
        for i in range(self._augment_factor):
            self._itemindexes = np.concatenate([self._itemindexes, self._itemindexes[1:] + self._itemindexes[-1]])
        self._is_referenced = False
        self._is_referencing = False
        self._referencees = []
        self._call_count = 0

        # Build augmentator
        if self._augmentator is None:
            self._augmentator = iaa.Sequential(
                [iaa.Affine(rotate=(-10, 10), scale=(0.9, 1.1)),
                 iaa.WithChannels(channels=[0], # TODO: this is temp solution to LBP channel
                                  children=iaa.AdditiveGaussianNoise(scale=(0,5), per_channel=False)),
                 iaa.WithChannels(channels=[0],
                                  children=iaa.LinearContrast(alpha=(0.5, 1.5), per_channel=False)),
                 ],
                random_order=False
            )
        self._update_augmentators()


        # Augment dtype
        self._augdtype = None
        if self.dtype == float or self.dtype == 'float' or np.issubdtype(self.dtype, np.float32):
            self._augdtype = 'float32'
        elif self.dtype == int or self.dtype == 'int' or np.issubdtype(self.dtype, np.integer):
            self._augdtype = 'int'


        if not self._references_dataset is None:
            self.set_reference_augment_dataset(self._references_dataset)

    def _update_augmentators(self):
        # Don't touch anything if referencing others.
        if self._is_referencing:
            return

        self._augmentators = self._augmentator.to_deterministic(n=1)   # reset augmentor
        self._augmentators = self._augmentator.to_deterministic(n=self._base_length*self._augment_factor)

        for _referencee in self._referencees:
            _referencee._augmentators = self._augmentators


    def set_reference_augment_dataset(self, dataset):
        assert isinstance(dataset, ImageDataSetAugment)
        assert not dataset in self._referencees,"Assigned dataset is already referenced."
        assert len(self) == len(dataset), "Datasets have different length! [%i, %i]"%(len(self), len(dataset))

        try:
            self._is_referenced=False
            self._augmentators = dataset._augmentators
            self._augmentator = dataset._augmentators
            self._augment_factor = dataset._augment_factor
            self._is_referencing=True
            dataset._is_referenced=True
            dataset._referencees.append(self)
        except:
            print("Cannot reference target!")
            self._is_referencing=False


    def size(self, int=None):
        if int is None:
            return [self.__len__()] + list(self.data[0].shape)
        else:
            return super(ImageDataSetAugment, self).size(int)

    def batch_done_callback(self):
        if self._update_each_epoch:
            self._update_augmentators()
        self._call_count = 0


    def get_unique_IDs(self, globber=None):
        return super(ImageDataSetAugment, self).get_unique_IDs(globber) * self._augment_factor

    def __getitem__(self, item):
        self._call_count += 1

        # if item is within original length, return the original image
        if item < self._base_length:
            out = super(ImageDataSetAugment, self).__getitem__(item)
        else:
            # else return the augment image
            slice_index = item % self._base_length
            baseim = super(ImageDataSetAugment, self).__getitem__(slice_index)

            # because imgaug convention is (B, H, W, C), we have to permute it before actual augmentation
            if baseim.squeeze().ndimension() == 3:
                baseim = baseim.permute(1, 2, 0)

            if self._is_segment:
                # segmentation maps requires speacial treatments
                segim = ia.SegmentationMapOnImage(baseim.squeeze().numpy().astype(self._augdtype),
                                                  baseim.squeeze().shape,
                                                  self._nb_of_classes)
                augmented = self._augmentators[item - self._base_length].augment_segmentation_maps([segim])[0]
                augmented = augmented.get_arr_int()
            else:
                augmented = self._augmentators[item - self._base_length].augment_image(
                    baseim.squeeze().numpy().astype(self._augdtype))

            # convert back into pytorch tensor convention (B, C, H, W)
            if baseim.squeeze().ndimension() == 3:
                baseim = baseim.permute(2, 0, 1)
                augmented = augmented.transpose(2, 0, 1)
            out = torch.from_numpy(augmented).view_as(baseim).to(baseim.dtype)
            del augmented, baseim
            # gc.collect()

        if self._call_count >= self.__len__():
            if self._update_each_epoch:
                self.batch_done_callback()

        return out
