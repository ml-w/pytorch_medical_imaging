from fpdf import FPDF
from MedImgDataset import ImageDataSet
import numpy as np
import os
from matplotlib.image import imsave

choices = [952, 1251,  995,  868,  798, 988, 1214, 1118, 1236, 932, 1256,
       1011, 1193, 1253, 1058, 1169,  836, 1061, 1243, 1085]


def make_pdf_from_images(imagelist, pdf_fname):
    """"""
    pdf = FPDF('L', 'mm', 'A4')
    pdf.add_page()
    pdf.set_font('Arial', 'B', 50)
    pdf.cell(0, 50, "CE-T1W and T2W-FS", border='B', align='C')
    pdf.set_font('Arial', 'B', 12)
    for im in imagelist:
        pdf.add_page()
        pdf.cell(0, 0, os.path.basename(im))
        pdf.image(im, 0, 15, w=297)
    pdf.output('/home/lwong/FTP/temp/test.pdf', 'F')

def make_images(im_left, im_right, outputdir, vrange):
    assert isinstance(im_left, ImageDataSet) and isinstance(im_right, ImageDataSet)

    for i, row in enumerate(zip(im_left, im_right)):
        t1, t2 = row
        outim = np.concatenate([t1.numpy().squeeze(), t2.numpy().squeeze()], axis=1)
        outname = outputdir + '/' + os.path.basename(im_left.get_data_source(i)).replace('.nii.gz', '_%03d.jpeg'%im_left.get_internal_index(i))
        imsave(outname, outim, vmin=vrange[0], vmax=vrange[1], cmap="Greys_r")

if __name__ == '__main__':
    outfolder = '../NPC_Segmentation/temp/'

    # t1c = ImageDataSet('../NPC_Segmentation/01.NPC_dx', verbose=True, idlist=choices, filesuffix="*T1*C*", loadBySlices=0)
    # t2w = ImageDataSet('../NPC_Segmentation/01.NPC_dx', verbose=True, idlist=choices, filesuffix="*T2*", loadBySlices=0)


    # make_images(t2w, t1c, outfolder, [0,2500])


    files = [outfolder + '/' + f for f in os.listdir(outfolder)]
    files.sort()
    make_pdf_from_images(files, None)