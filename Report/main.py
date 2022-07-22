import unittest
import os
import subprocess
import tempfile
import shutil
import gc
from npc_report_gen.report_gen_pipeline import main, generate_report, run_rAIdiologist
from img_proc import seg_post_main
from pathlib import Path
from mnts.mnts_logger import MNTSLogger

def main2():
    with MNTSLogger('.', 'test_report', verbose=True, keep_file=False, log_level='debug') as logger:
        p = Path('/home/lwong/Desktop/NPC_Segmentation/NPC_Segmentation/0A.NIFTI_ALL/HKU/temp/T2WFS_TRA')
        # p = Path('/home/lwong/Desktop/NPC_Segmentation/NPC_Segmentation/0A.NIFTI_ALL/All/T2WFS_TRA')
        po = Path('/media/storage/Data/NPC_Segmentation/70.Screening_report/TestOutput/')
        pof = Path('/media/storage/Data/NPC_Segmentation/70.Screening_report/TestOutput/diag.csv')
        for pp in p.glob("*nii.gz"):
            try:
                main(['-i',
                      str(pp),
                      '-o',
                      str(po),
                      '-n',
                      '16',
                      '-f',
                      str(pof),
                      '--verbose',
                      '--keep-data',
                      '--keep-log',
                      '--skip-exist'
                      ])
            except Exception as e:
                logger.exception(e)

def main_():
    from configparser import ConfigParser, ExtendedInterpolation
    from pytorch_med_imaging.med_img_dataset import ImageDataSet
    with MNTSLogger('.', 'main', verbose=True, log_level='debug', keep_file=False) as logger:
        config_path = Path("../Configs/BM_LargerStudy/BM_rAIdiologist_nyul_v2.ini")
        config = ConfigParser(interpolation=ExtendedInterpolation())
        config.read(str(config_path))
        config['General']['fold_code'] = "B00"

        # Read ini file for idlist
        idlist = ConfigParser()
        idlist.read(Path('..').joinpath(config['Filters'].get('id_list')))
        testing_ids = idlist['FileList']['testing'].split(',')

        # Read other attributes
        target_dir = Path('..').joinpath(config['Data']['target_dir'])
        input_dir  = Path('..').joinpath(config['Data']['input_dir'])
        input_dir  = Path('../NPC_Segmentation/0A.NIFTI_ALL/All/T2WFS_TRA')
        idGlobber  = config['LoaderParams']['idGlobber']

        # Small tumor list
        small_tumors = ['NPC001',
                        'NPC007',
                        'NPC017',
                        'NPC029',
                        'NPC030',
                        'NPC136',
                        'NPC169',
                        'NPC176',
                        'NPC184',
                        'NPC268',
                        'P008',
                        'P046',
                        'P085',
                        'P101',
                        'P112',
                        'P113',
                        'P129',
                        'P299']

        # list where rAIdiologist got it wrong
        wronglist = [
            '1404',
            '334'
            # "NPC030",
            # "NPC169",
            # "NPC184",
            # "NPC268",
            # "P129"
        ]

        # image_data = ImageDataSet(str(input_dir), verbose=True, filtermode='idlist',
        #                           idGlobber=idGlobber, idlist=small_tumors)

        # Run report gen
        po = Path('/media/storage/Data/NPC_Segmentation/70.Screening_report/TestOutput_B00_v3/')
        pof = Path('/media/storage/Data/NPC_Segmentation/70.Screening_report/TestOutput_B00_v3/diag.csv')
        po = Path('/media/storage/Data/NPC_Segmentation/70.Screening_report/TestOutput_small_tumors_v3/')
        pof = Path('/media/storage/Data/NPC_Segmentation/70.Screening_report/TestOutput_small_tumors_v3/diag.csv')
        try:
            main(['-i',
                  str(input_dir),
                  '-o',
                  str(po),
                  '-n',
                  '16',
                  '-f',
                  str(pof),
                  '--idGlobber',
                  str(idGlobber),
                  '--idlist',
                  ','.join(small_tumors),
                  '--verbose',
                  '--keep-data',
                  '--keep-log',
                  # '--skip-exist'
                  ])
            gc.collect()
        except Exception as e:
            logger.exception(e)
            gc.collect()

    pass

if __name__ == '__main__':
    main_()