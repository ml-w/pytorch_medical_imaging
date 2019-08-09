from sklearn.mixture import GaussianMixture
from MedImgDataset import ImageDataSet
from functools import partial
import numpy as np
import os


def get_gaussian_mixture(image, n_components=3):
    gmm = GaussianMixture(n_components=n_components, verbose=True, verbose_interval=1)
    gmm.fit(image.flatten().reshape(-1, 1))

    params = np.array(list(zip(gmm.weights_, gmm.means_, gmm.covariances_)))
    return params

def gaussian_mixture_model(params):
    callable = lambda x: np.sum(np.stack([a*np.exp(-(x-m)**2/2./c) for a, m, c in params], 0), axis=0)
    return callable

def main():
    import multiprocessing as mpi

    pool = mpi.Pool(mpi.cpu_count())

    imset = ImageDataSet("../NPC_Segmentation/01.NPC_dx",
                         filelist='../NPC_Segmentation/99.Testing/B02/B02_Training_Input.txt',
                         filesuffix="*T2*",
                         verbose=True)
    segset = ImageDataSet("../NPC_Segmentation/05.NPC_seg_T2",
                          filelist='../NPC_Segmentation/99.Testing/B02/B02_Training_GT.txt',
                          verbose=True,
                          dtype=np.uint8)

    results_im = pool.map_async(partial(get_gaussian_mixture, n_components=3),
                                [im[im > 25].numpy() for im in imset])
    pool.close()
    pool.join()

    pool = mpi.Pool(mpi.cpu_count())
    results_seg = pool.map_async(partial(get_gaussian_mixture, n_components=3),
                                [im[segset[i]].numpy() for i, im in enumerate(imset)])
    pool.close()
    pool.join()

    f_im = file('../NPC_Segmentation/GaussianMixture_T2.txt', 'w')
    f_seg = file('../NPC_Segmentation/GaussianMixture_T2_seg.txt', 'w')
    for i, row in enumerate(results_im.get()):
        name = os.path.basename(imset.get_data_source(i))
        w = name+':'+ '['+ ','.join(['['+','.join(['%.05f'%v for v in val])+']' for val in row])+']'
        f_im.write(w + '\n')

    for i, row in enumerate(results_seg.get()):
        name = os.path.basename(imset.get_data_source(i))
        w = name+':'+ '['+ ','.join(['['+','.join(['%.05f'%v for v in val])+']' for val in row])+']'
        f_seg.write(w + '\n')


    pass

if __name__ == '__main__':
    main()
