import os
from MedImgDataset import ImageDataSet

print(os.listdir('../NPC_Segmentation/0A.NIFTI_ALL/Malignant/CE'
                 '-T1W_TRA/'))
dat = ImageDataSet('../NPC_Segmentation/0A.NIFTI_ALL/Malignant/CE'
                 '-T1W_TRA/', readmode='recursive', filtermode='both',
                   idlist=['1001','1002'], regex='(?=.*T1W.*)' ,verbose=True)
dd = dat.get_data_by_ID('NPC001')
print(dat)
print([d.shape for d in dd])

