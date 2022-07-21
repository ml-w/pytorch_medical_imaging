import json
import os
import pprint
import re
import shutil
import tempfile
import pandas as pd
from pathlib import Path
from typing import Iterable, Optional, Union

import SimpleITK as sitk
from mnts.mnts_logger import MNTSLogger
from mnts.scripts.dicom2nii import console_entry as dicom2nii


def process_input(in_dir: Union[Path, str],
                  out_dir: Union[Path, str],
                  idGlobber: Optional[str] = "^[\w\d]+",
                  idlist: Optional[Iterable[str]] = None,
                  num_worker: Optional[int] = 1) -> None:
    r"""Process the input directories. If the directories already are nifty, create symbolic
    links in the target dir. If the directory looks like a DICOM directory, calls
    `get_t2w_series_files` and generate a nifty. If the directory consist of many nifty
    files, link all of them to the target dir.
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)

    if in_dir.is_dir():
        # if any nii files were in the dir, treat the directory as nii directory
        if len(list(in_dir.glob("*nii???"))) == 0:
            # Get dicom files and copy to a temp directory
            _dicom_files = get_t2w_series_files(in_dir)
            with tempfile.TemporaryDirectory() as _temp_dicom_dir:
                [shutil.copy2(d, _temp_dicom_dir.name) for d in _dicom_files]
                dicom2nii(f"-i {_temp_dicom_dir.name} -o {str(out_dir)} -n {num_worker} -g '.*' "
                          f"--dump-dicom-tags --use-patient-id".split())

        else:
            #!! This is not functional yet
            for f in in_dir.glob("*.nii???"):
                if not out_dir.is_dir():
                    msg = f"Multiple nifty files detected out out_dir is not a directory, got {str(out_dir)}"
                    raise IOError(msg)
                fid = re.search(idGlobber, str(f.name))
                if fid is None:
                    raise IOError(f"ID cannot be obtained for {str(f.name)} using patterm '{idGlobber}'")
                else:
                    fid = fid.group()
                if not fid in idlist:
                    continue
                out_dir.joinpath(f.name).symlink_to(f.resolve())
                json_name = out_dir.joinpath(re.sub("\.nii(\.gz)?$", ".json", str(f.name)))
                json.dump({'0010|0010': fid,
                           '0010|0020': fid},
                            Path(json_name).open('w')
                          )
    elif in_dir.is_file() and in_dir.suffix in ('.nii', '.gz'):
        # copy that to the temp dir
        if out_dir.is_dir():
            out_dir.joinpath(in_dir.name).symlink_to(in_dir.resolve())
        else:
            out_dir.symlink_to(in_dir)
        fid = re.search(idGlobber, str(f.name))
        if fid is None:
            raise IOError(f"ID cannot be obtained for {str(in_dir.name)} using patterm '{idGlobber}'")
        else:
            fid = fid.group()
        json_name = out_dir.joinpath(re.sub("\.nii(\.gz)?$", ".json", str(f.name)))
        json.dump({'0010|0010': fid,
                   '0010|0020': fid},
                  Path(out_dir).joinpath(json_name).open('w')
                  )
    else:
        raise IOError(f"Input specified is incorrect, expect a directory or an nii file, got '{in_dir}' instead.")


def get_t2w_series_files(in_dir):
    r"""Check and see if there are T2w-fs files, if there are more than one, prompt users
    to select from a list which to use"""
    with MNTSLogger('pipeline.log', 'get_t2w_series_files') as logger:
        in_dir = Path(in_dir)
        # Check all subdirs
        _tag_list = []
        _file_list = []
        for r, d, f in os.walk(str(in_dir)):
            _curdir = Path(r)
            if not len(f) == 0:
                series_ids = sitk.ImageSeriesReader_GetGDCMSeriesIDs(str(_curdir))
                for sid in series_ids:
                    dicom_files = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(
                            str(_curdir),
                            sid
                        )
                    dicom_files = [str(Path(difil).absolute()) for difil in dicom_files]
                    logger.debug(f"Found files: {dicom_files}")

                    header_reader = sitk.ImageFileReader()
                    header_reader.SetFileName(dicom_files[0])
                    header_reader.LoadPrivateTagsOn()
                    header_reader.ReadImageInformation()
                    all_tags = {k: header_reader.GetMetaData(k) for k in header_reader.GetMetaDataKeys()}

                    tags = {
                        '0008|103e': None,  # Image description, usually they put protocol name here
                        '0018|1030': None,  # Acquisition protocol name
                        '2001|100b': None,  # Scan Plane
                    }
                    for dctag in tags:
                        try:
                            tags[dctag] = header_reader.GetMetaData(dctag).rstrip().rstrip(' ')
                        except:
                            logger.debug(f"Tag [{dctag}] missing for image {f}")
                            tags[dctag] = 'Missing'
                    logger.debug(f"Tags: {pprint.pformat(tags)}")

                    if (re.match("((?i)(?=).*T2.*)((?i)(?=).*(fs).*)|((?i)(?=).*stir.*)", tags['0018|1030']) is None) or \
                            (re.match(f"(?i)(?=).*TRA.*", tags['2001|100b']) is None):
                        logger.debug(f"Series {sid}({tags['0018|1030']}) is not T2w-fs")
                        continue

                    # If pass these two tests, return the file list
                    logger.debug(f"Find target DICOM files '{tags['0018|1030']}'")
                    _file_list.append(tuple([tags, dicom_files]))

        if len(_file_list) == 0:
            raise ArithmeticError(f"Cannot find a T2w-fs image in {in_dir}")
        elif len(_file_list) > 1:
            # force verbose
            _v = logger._verbose
            logger._verbose = True
            logger.warning(f"More than one target sequence found, please choose from below the desired sequence...")
            logger.info('\n' + '\n\n'.join([f"[{i}]: \n {pprint.pformat(_f[0], indent=4)}" for i, _f in enumerate(_file_list)]))
            # _choice = input(f"Choose sequence [0 to {len(_file_list) - 1}]: ")
            _choice = 1
            logger.debug(f"Chosen sequence: {_choice}")
            file_list = _file_list[int(_choice)][1]
            # resume
            logger._verbose = _v
        else:
            file_list = _file_list[0][1]
    return file_list

def generate_id_path_map(file_list: Iterable[Union[Path, str]],
                         idGlobber: str,
                         name: Optional[str] = None) -> pd.Series:
    r"""Glob IDs from an iterable of path or string and create a named pd.Series where
    the index are the ID and the data is the corresponding file path."""
    # glob ids
    re_obj = {str(r): re.search(idGlobber, str(Path(r).name)) for r in file_list}
    if None in re_obj.values():
        msg = f"ID cannot be globbed from some of the paths: \n"
        msg += '\n\t'.join([str(key) for key, values in re_obj.items() if values is None])
        raise ArithmeticError(msg)

    out = pd.Series(data=re_obj.keys(), index=[obj.group() for obj in re_obj.values()], name=name)
    return out