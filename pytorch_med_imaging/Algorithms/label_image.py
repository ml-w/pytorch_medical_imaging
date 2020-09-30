import matplotlib.pyplot as plt

from MedImgDataset import ImageDataSet
from torch.nn.functional import avg_pool2d
from torchvision.utils import make_grid

def mark_image(im):
    nrow = 5
    pool_scale = 2
    im = avg_pool2d(im.unsqueeze(1), pool_scale)
    im_grid = make_grid(im, nrow=nrow, normalize=True, padding=0)

    dim_y, dim_x = im[0].shape[1:]
    print(dim_y, dim_x)

    # get point
    plt.ion()
    plt.imshow(im_grid[0])
    plt.tight_layout()
    plt.show()
    pt_x, pt_y = plt.ginput(1, timeout=0)[0]
    plt.ioff()
    plt.cla()

    print(pt_x, pt_y)

    # compute the slice index and coorinate
    row = pt_y // dim_y
    col = pt_x // dim_x

    slice_index = nrow * row + col
    slice_coordinate = [pt_x - col * dim_x,pt_y - row * dim_y]
    slice_coordinate = [int(s * pool_scale) for s in slice_coordinate]

    #inver the y
    # slice_coordinate[1] = dim_y * pool_scale - slice_coordinate[1]

    return slice_index, slice_coordinate

if __name__ == '__main__':
    import csv
    imset = ImageDataSet('../NPC_Segmentation/42.Benign_Malignant_Upright', verbose=True, debugmode=False)

    f = open('../NPC_Segmentation/42.Benign_Malignant_Upright/center_of_nasopharynx.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['Data ID', 'Slice Number', 'Coord_X', 'Coord_Y'])
    for i, im in enumerate(imset):
        tup = mark_image(im)
        data_id = imset.get_unique_IDs("(NPC|P)?[0-9]{3,5}")[i]
        s_num, (coord_x, coord_y) = tup
        writer.writerow([data_id, s_num, coord_x, coord_y])
        print([i, data_id, s_num, coord_x, coord_y])

