# -*- coding: utf-8 -*-
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Anomalib Test
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
The largest public collection of ready-to-use deep learning anomaly detection algorithms and 
benchmark datasets.

[GITHUB](https://github.com/openvinotoolkit/anomalib)

Example running PADIM model on the MVTec bottle dataset. After training two examples of inference are 
run, one on a bad sample and one on a good sample.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
"""

## Install Library
!git clone https://github.com/openvinotoolkit/anomalib.git
!pip install anomalib
!ls

# Commented out IPython magic to ensure Python compatibility.
%cd anomalib
!ls

"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## TRAIN MODEL:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
* Default train = PADIM on MVTec AD Bottle
* MVTec dimensions = [1024, 1024, 3] + Ground truths
* Change Catagory - bottle to leather:
    * /configs/models/padim.yaml
    * /anomalib/models/padim/config.yaml
* Models:
  * cflow       - dfm       - dfkde - draem
  * fastflow    - ganomaly  - padim - patchcore
  * stfpm       - reverse_distillation
"""
!python tools/train.py    # Train PADIM on MVTec AD leather


"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Test the Model:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
Inference (results saved in directory: "test_results"):
"""
!mkdir test_result

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Test Inference & Visualise:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# LIGHTENING
!python tools/inference/lightning_inference.py \
    --config anomalib/models/padim/config.yaml \
    --weights results/padim/mvtec/leather/weights/model.ckpt \
    --input datasets/MVTec/leather/test/cut/003.png \
    --output test_result

# TORCH
!python tools/inference/torch_inference.py \
    --config anomalib/models/padim/config.yaml \
    --weights results/padim/mvtec/leather/weights/model.ckpt \
    --input datasets/MVTec/leather/test/cut/003.png \
    --output test_result

# 
!python tools/inference/torch_inference.py \
    --config anomalib/models/padim/config.yaml \
    --weights results/padim/mvtec/leather/weights/model.ckpt \
    --input datasets/MVTec/leather/test/cut/003.png \
    --output test_result




"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Other Models:

Training a model on a specific dataset and category requires further configuration. 
Each model has its own configuration file, config.yaml , which contains data, model and 
training configurable parameters. To train a specific model on a specific dataset and category, 
the config file is to be provided:

python tools/train.py --config <path/to/model/config.yaml>
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
"""

!python tools/train.py --config anomalib/models/padim/config.yaml

"""
Alternatively, a model name could also be provided as an argument, 
where the scripts automatically finds the corresponding config file.

**Available Models:**
CFlow, DFM, DFKDE, FastFlow, PatchCore, PADIM, STFPM, GANomaly
"""

!python tools/train.py --model padim


"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Custom Datasets:

### Continue from here...
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
"""