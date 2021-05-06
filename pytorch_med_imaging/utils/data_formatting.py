import SimpleITK as sitk
import os
import numpy as np
import re
import multiprocessing as mpi
import random
sitk.ProcessObject_GlobalWarningDisplayOff()

from pytorch_med_imaging.logger import Logger

from .preprocessing import recursive_list_dir

def dicom2nii(folder: str,
              out_dir: str =None,
              seq_filters: list or str = None,
              idglobber: str = None,
              use_patient_id: bool = False) -> None:
    """
    Covert a series under specified folder into an nii.gz image.
    """

    # logger = Logger['utils.dicom2nii']
    print(f"Handling: {folder}")

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        assert os.path.isdir(out_dir), "Output dir was not made."

    if not os.path.isdir(folder):
        print(f"Cannot locate specified folder: {folder}")
        raise IOError("Cannot locate specified folder!")

    # Default globber
    if not isinstance(idglobber, str):
        idglobber = "(?i)(NPC|P|RHO|T1rhoNPC)?[0-9]{3,5}"

    folder = os.path.abspath(folder)
    f = folder.replace('\\', '/')

    matchobj = re.search(idglobber, os.path.basename(f))

    if not matchobj is None:
        prefix1 = matchobj.group()
    else:
        prefix1 = "NA"


    # Read file
    series = sitk.ImageSeriesReader_GetGDCMSeriesIDs(f)

    for ss in series:
        # print.info(f"{ss}")
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

        # Replace prefix if use patient id for file id
        if use_patient_id:
            prefix1 = headerreader.GetMetaData('0010|0020').rstrip()
        outname = out_dir + '/%s-%s+%s.nii.gz'%(prefix1,
                                              headerreader.GetMetaData('0008|103e').rstrip().replace(' ','_'),
                                              headerreader.GetMetaData('0020|0011').rstrip()) # Some series has the same series name, need this to differentiate

        # Skip if dicom tag (0008|103e) contains substring in seq_filters
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
        print(f"Writting: {outname}")
        outimage.SetMetaData('intent_name', headerreader.GetMetaData('0010|0020').rstrip())
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
    logger = Logger['nii2dicom']
    logger.log_print("Worker {}: Recieve job for {}.".format(mpi.current_process().name, out_dir))

    import time
    assert isinstance(image, sitk.Image), "Input is not an image."
    assert dicom_tags is None or isinstance(dicom_tags, dict), "Arguement dicom_tags provided has to be dictionary."

    logger.info(f"{out_dir}, {dicom_tags}")

    logger.info("Processing for: {}".format(out_dir))
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
            logger.warning("Casting failed for: {}".format(f))
            continue

        out_name = os.path.join(out_dir, patient_name, series_number)
        if not os.path.isdir(out_name):
            os.makedirs(out_name)

        try:
            nii2dicom(image, out_name, dicomtags)
        except:
            logger.exception("Failed for: {}".format(f))
            continue

    blind_id_record = open("blind_map.csv", 'w')
    for k in blind_map:
        blind_id_record.write('{},{}\n'.format(k, blind_map[k]))


def batch_dicom2nii(folderlist, out_dir,
                    workers=8,
                    seq_fileters=None,
                    idglobber = None,
                    use_patient_id = False):
    import multiprocessing as mpi
    logger = Logger['mpi_dicom2nii']
    logger.info(f"{folderlist}")

    pool = mpi.Pool(workers)
    for f in folderlist:
        # dicom2nii(f, out_dir, seq_fileters, idglobber, use_patient_id)
        pool.apply_async(dicom2nii, args=[f, out_dir, seq_fileters, idglobber, use_patient_id])
    pool.close()
    pool.join()




