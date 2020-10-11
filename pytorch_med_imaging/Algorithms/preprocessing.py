import SimpleITK as sitk
import os
import numpy as np
import re
import multiprocessing as mpi
from tqdm import *
import random
sitk.ProcessObject_GlobalWarningDisplayOff()
from pytorch_med_imaging.logger import Logger
logger = Logger('./batch_job.log')


def RecursiveListDir(searchDepth, rootdir):
    """
      Recursively lo
    :param searchDepth:
    :param rootdir:
    :return:
    """

    dirs = os.listdir(rootdir)
    nextlayer = []
    for D in dirs:
        if os.path.isdir(rootdir + "/" + D):
            nextlayer.append(rootdir + "/" + D)

    DD = []
    if searchDepth >= 0 and len(nextlayer) != 0:
        for N in nextlayer:
            K = RecursiveListDir(searchDepth - 1, N)
            if not K is None:
                DD.extend(K)

    DD.extend(nextlayer)
    return DD


def SmoothImages(root_dir, out_dir):
    import fnmatch

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    f = os.listdir(root_dir)
    fnmatch.filter(f, "*.nii.gz")

    for fs in f:
        print(fs)
        im = sitk.ReadImage(root_dir + "/" + fs)
        out = sitk.SmoothingRecursiveGaussian(im, 8, True)
        sitk.WriteImage(out, out_dir + "/" + fs)


def dicom2nii(folder, out_dir=None, seq_filters=None):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        assert os.path.isdir(out_dir)

    if not os.path.isdir(folder):
        print("Cannot locate specified folder! ", folder)
        raise IOError("Cannot locate specified folder!")

    print("Handling: ", folder)
    folder = os.path.abspath(folder)
    f = folder.replace('\\', '/')
    # matchobj = re.search('NPC[0-9]+', f)
    matchobj = re.search('(?i)(NPC|P)?[0-9]{3,5}', f)
    # prefix1 = f.split('/')[-2]
    prefix1 = f[matchobj.start():matchobj.end()]


    # Read file
    series = sitk.ImageSeriesReader_GetGDCMSeriesIDs(f)
    for ss in series:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(
            f,
            ss
        ))
        outimage = reader.Execute()

        # Generate out file name
        headerreader = sitk.ImageFileReader()
        headerreader.SetFileName(reader.GetFileNames()[0])
        headerreader.LoadPrivateTagsOn()
        headerreader.ReadImageInformation()
        outname = out_dir + '/%s-%s+%s.nii.gz'%(prefix1,
                                              headerreader.GetMetaData('0008|103e').rstrip().replace(' ','_'),
                                              headerreader.GetMetaData('0020|0011').rstrip()[0])

        if not seq_filters is None:
            if isinstance(seq_filters, list):
                regex = "("
                for i, fil in enumerate(seq_filters):
                    regex += '(.*' + fil + '{1}.*)'
                    if i != len(seq_filters) - 1:
                        regex += '|'
                regex += ')'
                if re.match(regex, headerreader.GetMetaData('0008|103e')) is None:
                    print("skipping ", headerreader.GetMetaData('0008|103e'), "from ", f)
                    continue
            elif isinstance(seq_filters, str):
                if re.match(seq_filters, headerreader.GetMetaData('0008|103e')) is None:
                    print("skipping ", headerreader.GetMetaData('0008|103e'), "from ", f)
                    continue

        # Write image
        print("Writting: ", outname)
        sitk.WriteImage(outimage, outname)
        del reader

def nii2dicom(image, out_dir = None, dicom_tags = None):
    """
    Conver image to dicom. Note that only uint16 is supported for most viewing software.

    Args:
        image (sitk.Image): Input 3D image
        out_dir (str): Output directory to hold the data.
        dicom_tags (dict): Custom DICOM tag items.
    """
    logger.log_print("Worker {}: Recieve job for {}.".format(mpi.current_process().name, out_dir))

    import time
    assert isinstance(image, sitk.Image), "Input is not an image."
    assert dicom_tags is None or isinstance(dicom_tags, dict), "Arguement dicom_tags provided has to be dictionary."

    print(out_dir, dicom_tags)

    print("Processing for: {}".format(out_dir))
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    # Generate a random UID
    uid_root = "1.2.826.0.1.5421469.2.1248"         # 27 charactor root
    uid = uid_root + "." + ''.join([str(np.random.randint(0, 9)) for i in range(12)]) + "." + \
          modification_time + modification_date
    # print(uid, "=------<<<<<")
    sop_uid = "1.2.840.10008.5.1.4.1.1.4"   # MR Image Storage
    # sop_uid = "1.2.840.10008.5.1.4.1.1.4.1" # Enhanced MRI Storage

    sop_instance_uid = sop_uid + "." + modification_time + modification_date


    # Set up some default dicom_tags
    direction = image.GetDirection()
    default_tags = {'0008|0031': modification_time,
                    '0008|0021': modification_date,
                    '0008|0008': "DERIVED\\SECONDARY", # Image type
                    '0020|000d': uid,
                    '0020|000e': uid,
                    '0020|0037': '\\'.join(map(str, (direction[0], direction[3], direction[6],
                                                      direction[1], direction[4], direction[7]))),
                    '0008|0016': sop_uid,
                    }

    # Replace tags
    for k in default_tags:
        if not k in dicom_tags:
            dicom_tags[k] = default_tags[k]

    # if direction is not sepcified in input dicom_tags, use direction and origin of the input image
    for i in range(image.GetDepth()):
        image_slice = image[:,:,i]

        for tag in dicom_tags:
            image_slice.SetMetaData(tag, dicom_tags[tag])

        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time
        image_slice.SetMetaData("0008|0018", sop_uid + "%03d"%i)
        image_slice.SetMetaData("0018|0050", str(np.round(image.GetSpacing()[2], 3)))
        image_slice.SetMetaData("0018|0088", str(np.round(image.GetSpacing()[2], 3)))
        image_slice.SetMetaData("0020|0013", str(i)) # Instance Number
        image_slice.SetMetaData("0020|0032", '\\'.join(
            map(str,image.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
        image_slice.SetMetaData("0028|0030", '\\'.join(map(str, image_slice.GetSpacing())))

        writer.SetFileName(os.path.join(out_dir, "%03d.dcm"%i))
        writer.Execute(image_slice)
        # print(os.path.join(out_dir, "%03d.dcm"%i))

def batch_nii2dicom(filelist, out_dir, workers = 8, blind_out = False):
    sort_dict = {'T2WFS':   "5021",
                 'T2W':     "502",
                 'CE-T1WFS':"4011",
                 'CE-T1W':  "4010",
                 'T1W':     "301",
                 'SURVEY': "00",
                 'NECK': "10"
                 }

    direction = {'_COR': "3",
                 '_TRA': "2",
                 '_SAG': "1",}


    blind_map ={}

    random.shuffle(filelist)

    ps = []
    pool = mpi.Pool(workers)
    for i, f in enumerate(filelist):
        sequence_name = os.path.basename(os.path.dirname(f))
        series_number = ""
        for s in sort_dict:
            if sequence_name.find(s) >= 0:
                series_number += sort_dict[s]
                break
        for d in direction:
            if sequence_name.find(d) >= 0:
                series_number += direction[d]
                break

        id = re.search("([a-zA-Z0-9]{3,6})", os.path.basename(f)).group()

        if blind_out:
            if id not in blind_map:
                blind_map[id] = len(blind_map)
            blind_id = blind_map[id]
            patient_name = "Blind_ID_%03d"%blind_id
        else:
            patient_name = id


        dicomtags = {
            '0020|0011': series_number,
            '0008|103E': sequence_name,
            '0010|0010': patient_name,
            '0008|0060': "MR"
        }

        try:
            image = sitk.ReadImage(f)
            image = sitk.Cast(image, sitk.sitkUInt16)
        except:
            print("Casting failed for: {}".format(f))
            continue

        out_name = os.path.join(out_dir, patient_name, series_number)
        if not os.path.isdir(out_name):
            os.makedirs(out_name)

        try:
            nii2dicom(image, out_name, dicomtags)
        except:
            print("Failed for: {}".format(f))
            continue
        # print(dicomtags)
        # p = pool.apply_async(partial(nii2dicom, out_dir = out_name, dicom_tags = dicomtags), args=image)
        # ps.append(p)

    blind_id_record = open("blind_map.csv", 'w')
    for k in blind_map:
        blind_id_record.write('{},{}\n'.format(k, blind_map[k]))
    #
    # for p in tqdm(ps):
    #     p.wait()
    #
    # pool.close()
    # pool.join()



def batch_dicom2nii(folderlist, out_dir, workers=8, seq_fileters=None):
    import multiprocessing as mpi

    pool = mpi.Pool(workers)
    # pool.map_async(partial(dicom2nii, out_dir=out_dir, seq_fileters=seq_fileters),
    #                folderlist)
    for f in folderlist:
        p = pool.apply_async(dicom2nii, args=[f, out_dir, seq_fileters])
    pool.close()
    pool.join()


def make_mask(inimage, outdir, pos=-1):
    if isinstance(inimage, str):
        # tqdm.write("Reading " + inimage)
        inimage = sitk.ReadImage(inimage)

    gttest = sitk.BinaryThreshold(inimage, upperThreshold=65535, lowerThreshold=200)
    gttest = sitk.BinaryDilate(gttest, [15, 15, 0], sitk.BinaryMorphologicalOpeningImageFilter.Ball)
    gttest = sitk.BinaryErode(gttest, [15, 15, 0], sitk.BinaryMorphologicalOpeningImageFilter.Ball)
    # gttest = sitk.BinaryMorphologicalClosing(gttest, [0, 25, 25], sitk.BinaryMorphologicalOpeningImageFilter.Ball)
    ss = []

    if pos == -1:
        try:
            pos = int(mpi.current_process().name.split('-')[-1])
        except Exception as e:
            tqdm.write(e.message)

    try:
        for i in trange(gttest.GetSize()[-1], position=pos, desc=mpi.current_process().name):
            ss.append(sitk.GetArrayFromImage(sitk.BinaryFillhole(gttest[:,:,i])))
        gttest = sitk.GetImageFromArray(np.stack(ss))
        # gttest = sitk.BinaryDilate(gttest, [0, 3, 3], sitk.BinaryDilateImageFilter.Ball)
        gttest.CopyInformation(inimage)
        sitk.WriteImage(gttest, outdir)
        return 0
    except Exception as e:
        print(e.message)


def make_mask_from_dir(indir, outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    p = mpi.Pool(10)
    processes = []
    filelist = os.listdir(indir)
    filelist = [indir + '/' + f for f in filelist]

    for i, f in enumerate(filelist):
        outname = f.replace(indir, outdir)
        # make_mask(sitk.ReadImage(f), outname, 0)
        subp = p.apply_async(make_mask, (f, outname, -1))
        processes.append(subp)

    for pp in processes:
        pp.wait(50)
    p.close()
    p.join()


def main(args):
    try:
        os.makedirs(args[2], exist_ok=True)
    except:
        print("Cannot mkdir.")

    assert os.path.isdir(args[1]) and os.path.isdir(args[2]), 'Cannot locate inputs directories or output directory.'

    folders = RecursiveListDir(5, args[1])
    folders = [os.path.abspath(f) for f in folders]

    if isinstance(eval(args[3]), list):
        batch_dicom2nii(folders, args[2], eval(args[3]) if len(args) > 4 else None)
    else:
        batch_dicom2nii(folders, args[2], args[3] if len(args) > 4 else None)

if __name__ == '__main__':
    # folders = RecursiveListDir(5, '../NPC_Segmentation/00.RAW/Benign NP')
    # batch_dicom2nii(folders, out_dir='../NPC_Segmentation/00.RAW/NIFTI/Benign')
    # folders = RecursiveListDir(5, '../NPC_Segmentation/00.RAW/T1+C_Missing/t1c/')
    # folders = RecursiveListDir(5, '../NPC_Segmentation/00.RAW/MMX/840/')
    # batch_dicom2nii(folders, out_dir='../NPC_Segmentation/00.RAW/NIFTI/All')
    # dicom2nii('../NPC_Segmentation/00.RAW/MMX/769/S', '../NPC_Segmentation/00.RAW/NIFTI/MMX')
    batch_dicom2nii(RecursiveListDir(3, '../NPC_Segmentation/00.RAW/Jun16/779'),
                    '../NPC_Segmentation/0A.NIFTI_ALL/Malignant_2')
    # dicom2nii('../NPC_Segmentation/00.RAW/Transfer/Benign/NPC147/Orignial Scan/DICOM', '../NPC_Segmentation/0A.NIFTI_ALL/Benign')
    # main(sys.argv)
    # make_mask_from_dir('../NPC_Segmentation/06.NPC_Perfect/temp_t2/', '../NPC_Segmentation/06.NPC_Perfect/temp_mask')