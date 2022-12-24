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
    # The followings are specified in runtime because a temp folder was created to hold outputs
    # * output_dir
    # * cp_save_dir
    # * cp_load_dir

class SampleSegControllerCFG(SampleControllerCFG):
    data_loader_cfg = SampleSegLoaderCFG()
    data_loader_val_cfg = SampleSegLoaderCFG()
    solver_cfg = SampleSegSolverCFG()
    solver_cls = SegmentationSolver
    inferencer_cls = SegmentationInferencer
    data_loader_cls = PMIImageDataLoader
    data_loader_val_cls = PMIImageDataLoader