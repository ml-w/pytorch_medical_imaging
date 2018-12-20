from MedImgDataset.Computation import ImagePatchLocTex, ImageDataSet
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

import numpy as np

if __name__ == '__main__':
    imset = ImageDataSet('../NPC_Segmentation/01.NPC_dx', verbose=True, debugmode=True,
                         loadBySlices=0)

    patch = ImagePatchLocTex(imset, 128, 128)

    lbp = patch[0:50][0].numpy()
    lbp = lbp.reshape(lbp.shape[0], -1)
    print lbp.shape
    # img = patch[0:50][0]

    # ims1 = make_grid(lbp.unsqueeze(1), nrow=5, normalize=False)
    # ims2 = make_grid(img.unsqueeze(1), nrow=5, normalize=True)
    #
    # plt.imshow(ims1[0])
    # plt.show()
    hist = np.array([np.histogram(lbp[i][lbp[i] != 0], bins=100, range=[0, 255.]) for i in xrange(lbp.shape[0])])
    for i in range(len(hist)):
        plt.plot(hist[i][0])
    plt.show()
