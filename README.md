
# Overview

The source code of this software was release with regards to the spirit of reproducibility of a scientific study. The software were used in the submitted article titled "_Convolutional Neural Network for Discriminating Nasopharyngeal Carcinoma and Benign Hyperplasia on MRI_".

The purpose of this study is to propose a convolutional neural network (CNN) for the discrimination between early nasopharyngeal carcinoma (NPC) and benign hyperplasia, which manifest similarly on MR images of the nasopharynx, to expand the roll of MRI in NPC screening program. To do so, we focused on a non-contrast-enhanced MR sequence, the T2-weighted fat-suppressed sequence.

Specifically, the software accepts an MR image, cropped to a fixed size and centred approximately the naospharynx, and then computes a score from 0-1 with respect to the likelihood of NPC. 

# Usage

## Normalization

Before the images are fed to the CNN, it is necessary to perform pre-processing:
1. Normalize image intensity with Nyul normalizaiton. 
2. Centre image at the centre of mass.
3. Rotate the image about the center of mass to align with the plane of lateral symmetry. 
4. Crop the image to designated size.

### Example
```python
from head_and_neck_normalization import normalization

input = '/dir/to/nii/files/'
output = '/dir/to/deposit/output/'
normalization(input, output)
```

## Config File

This software uses .ini files to configure both the training and inference process. Configuration files used for this studies is deposited in the directory ```./Config/Class_C00.ini```

Examples ID list are also given in ```./Configs/CLASS_3Folds```

Parameters are breifly explained here:

|Genre|Parameter|Function|Values|
|-----|---------|--------|------|
|General|use_cuda|Use GPU for computation.|True / False|
| |run_mode|Training or inference| train / inference|
| |run_type|_Deprecated_ |Classification|
| |plot_tb|Plot training process to tensorboard, requires setting environmental variable TENSORBOARD_DIR| True / False|
|Checkpoint|cp_save_dir|Directory to save the network parameter states| **str**|
| |cp_load_dir|Directory to load the saved network states| **str**|
|RunParam|batch_size|Batch size| **int**|
| |initial_weight|_Deprecated_||
| |learning_rate| Initial learning rate of training| **float**|
| |momentum| Initial momentum of training| **float** [0-1]|
| |num_of_epoch| Number of epoch to train| **int**|
| |decay_rate_LR| Exponential decay rate for the learning rate| **float**|
|Data|target_dir|Directory of the ground-truth class .csv file| **str**|
| |input_dir|Directory of the input images| **str**|
| |output_dir|Directory of to deposit the output in inference mode| **str**|
| |validation_dir|Directory of the image for validation| **str**|
| |validation_gt_dir|Directory of the .csv file recorded with ground-truth class|**str**|
|Filters|re_suffix|Regular expression suffix for globbing files from input directories|**str**|
| |id_list|Ini file holding the id list of the desired input|**str**|
| |validation_id_list|Text file holding the id list of the desired validation|**str**|

## Training & Inference

For training, make sure the ini file set run_mode to ```train```: 
```bash
python main.py Config/File.ini
```

For training, if you specify correct input and output dir in the same ini file, you can simply use the command:
```bash
python main.py Config/File.ini --inference
```

Otherwise, make sure you set run_mode to ```inference```.

# Thirdparty Software

## EROS

EROS stands for _accurate robust symmetry estimation_. EROS identify a line of symmetry on 2D imagesThe purpose of EROS is to identify a coronal plane of symmetry in the scans as patients were not always facing perfectly upwards. EROS allows the normalization of head tilts. 

EROS exist as a submodule in this git repo, the code was implemented with reference to [1] on python. 

## Nyul Normalization

Intensity normalization is necessary for most of the MRI quantitative studies. Nyul propose one of the earliest algorithm [2] for the task and it is selected in this software because of its robustness and applicability on all squences.

The implementation was taken from [this](https://gitlab.com/eferrante/nyul) repo and modified for python version > 3.0. 

# Reference

