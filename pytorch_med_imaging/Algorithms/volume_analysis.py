import pandas as pd
import torch
from tqdm import *
from MedImgDataset import ImageDataSet

def create_volume_column(imdata, column_name='Volume'):
    assert isinstance(imdata, ImageDataSet)

    out_df = pd.DataFrame()
    unique_ids = imdata.get_unique_IDs(globber='[0-9]+')
    for i, im in enumerate(tqdm(imdata)):
        spacing = imdata.get_spacing(i)

        val, count = torch.unique(im, return_counts=True)

        if len(val) < 2:
            continue
        else:
            volume = torch.prod(torch.tensor(spacing)) * count[1]
            volume = volume.item()
            id = unique_ids[i]

            # Creates a panda row
            row = pd.DataFrame([[id, volume]], columns=['Index', column_name])
            out_df = out_df.append(row)
    out_df = out_df.set_index('Index')
    return out_df


if __name__ == '__main__':
    # segset = ImageDataSet('../NPC_Segmentation/98.Output/KFold_All', verbose=True, dtype='uint8',
    #                       debugmode=False)
    # gtset = ImageDataSet('../NPC_Segmentation/15.NPC_seg_T2_secondtime', verbose=True, \
    #                      idlist=segset.get_unique_IDs(), dtype='puint8',
    #                      debugmode=False)
    # varset = ImageDataSet('../NPC_Segmentation/05.NPC_seg_T2_AddCase', verbose=True, \
    #                      idlist=segset.get_unique_IDs(), dtype='uint8',
    #                      debugmode=False)
    #
    # col0 = create_volume_column(segset, 'Predicted Volume')
    # col1 = create_volume_column(gtset, 'Actual Volume')
    # col2 = create_volume_column(varset, 'Variability Volume')
    # df = pd.concat([col0, col1, col2], axis=1)
    #
    # df.to_csv('/home/lwong/FTP/temp/npc_volumes.csv')

    follow_up = ImageDataSet('/home/lwong/FTP/FTP/Images of NP screening_follow up study/Follow up', verbose=True, dtype='uint8')
    original = ImageDataSet('/home/lwong/FTP/FTP/Images of NP screening_follow up study/Original', verbose=True, dtype='uint8')

    col0 = create_volume_column(follow_up, 'Follow Up')
    col1 = create_volume_column(original, 'Original')
    df = pd.concat([col0, col1], axis=1)
    df.to_csv('/home/lwong/FTP/followup_vols.csv')