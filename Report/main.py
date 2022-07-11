import unittest
import os
import subprocess
import tempfile
import shutil
from npc_report_gen.report_gen_pipeline import get_t2w_series_files, main, generate_report, seg_post_main
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

if __name__ == '__main__':
    main2()