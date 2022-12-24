from pytorch_med_imaging.controller import PMIControllerCFG
from pytorch_med_imaging.pmi_data_loader import PMIImageDataLoader
from pytorch_med_imaging.solvers import *
from pytorch_med_imaging.inferencers import *
from .sample_cfg import *
from pathlib import Path

class SampleControllerCFG(PMIControllerCFG):
    fold_code = 'B00'
    run_mode = 'training'
    id_list = str(Path(__file__).parent.joinpath('sample_id_setting.ini').absolute())
    id_list_val = str(Path(__file__).parent.joinpath('sample_id_setting.txt').absolute())
    # output_dir is specified in testing script because its a runtime created temp folder

class SampleSegControllerCFG(SampleControllerCFG):
    data_loader_cfg = SampleSegLoaderCFG()
    data_loader_val_cfg = SampleSegLoaderCFG()
    solver_cfg = SampleSegSolverCFG()
    solver_cls = SegmentationSolver
    inferencer_cls = SegmentationInferencer
    data_loader_cls = PMIImageDataLoader
    data_loader_val_cls = PMIImageDataLoader