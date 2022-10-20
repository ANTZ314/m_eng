# ANOMALIB on LEATHER


[Anomalib Colab Test](https://colab.research.google.com/drive/1K4a4z2iZGBNhWdmt9Aqdld7kTAxBfAmi#scrollTo=Q4_JcO_RAt_v)

[GITHUB](https://github.com/openvinotoolkit/anomalib)


**From Anomalib readme**
MVTec AD dataset is one of the main benchmarks for anomaly detection, and is released under the
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License [(CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).


### TEST CONCEPT (MVTec-AD):

* Train (8x) EACH model on **MVTec_AD - Leather** dataset
* Perform (4x) infrence on (6x) each defect type (test image)
* 4 (inf) x 6 (defect) = 24 results
* 8 (models) x 24 (results) = 192 final results

ALSO:
* Graph 3 full result tables on entire MVTec Dataset


### ATTEMPT:

CUSTOM DATASET TRAIN - **kaggle**


---
## MODEL VS INFERENCE CHECKLIST:

**MODELS:**
clfow, dfkde, dfm, draem, fastflow, ganomaly
padim, patchcore, reverse_distillation, stfpm

**INFERENCERS:**

* [1] gradio, [2] lightening, 
* [3] openvino, [4] torch

**IMAGES TESTED:**

* color_015, cut_003, fold_000
* glue_008, good_18, poke_008

**Models Run & Inferred:**

| Models     | COLOR   | FOLD     | CUT      | GLUE     | GOOD     | POKE     |
| -------   | ------  | -------  | -------  | -------  | -------  | -------  |
| PADIM     | 2-4    | 2-4  | 2-4    | 2-4    | 2-4    | 2-4    |
| CFlow     | -        | -        | -        | -        | -        | -        |
| DFM        | -        | -        | -        | -        | -        | -        |
| DFKDE     | -        | -       | -        | -        | -        | -        |
| FastFlow  | -        | -       | -        | -        | -        | -        |
| PatchCore | -       | -       | -        | -        | -        | -        |
| STFPM     | -        | -       | -        | -        | -        | -	|
| GANomaly  | -      | -      | -        | -        | -        | -        |




## Inferencers:


### [1] Lightning

**RESULT:** 
```
!python tools/inference/lightning_inference.py \
    --config anomalib/models/padim/config.yaml \
    --weights results/padim/mvtec/leather/weights/model.ckpt \
    --input datasets/MVTec/leather/test/poke/008.png \
    --output test_result
```

### [2] Torch

**RESULT:** 

```
!python tools/inference/torch_inference.py \
    --config anomalib/models/padim/config.yaml \
    --weights results/padim/mvtec/leather/weights/model.ckpt \
    --input datasets/MVTec/leather/test/poke/008.png \
    --output test_result
```

### [3] OpenVINO

**RESULT:** 

**FAIL** NameError: name 'IECore' is not defined
```
!python tools/inference/openvino_inference.py \
    --config anomalib/models/padim/config.yaml \
    --weights results/padim/mvtec/leather/openvino/openvino_model.bin \
    --meta_data results/padim/mvtec/leather/openvino/meta_data.json \
    --input datasets/MVTec/leather/test/poke/008.png \
    --output results/padim/mvtec/leather/images
```

### [4] Gradio

**RESULT:** 

**FAIL** Exception..?
```
!python tools/inference/gradio_inference.py \
        --config ./anomalib/models/padim/config.yaml \
        --weights ./results/padim/mvtec/leather/weights/model.ckpt
```


## RESULTS:

**Documented in Excel File**

---
## From Github Results:

**NOTE:** 'Avg' is is over all Catagories of MVTec-AD

### Image-Level AUC 

| Model         |                    |    Avg    | Leather |
| ------------- | ------------------ | :-------: | :-----: |
| **PatchCore** | **Wide ResNet-50** | **0.980** |  1.000  |
| PatchCore     | ResNet-18          |   0.973   |  1.000  |
| CFlow         | Wide ResNet-50     |   0.962   | **1.0** |
| PaDiM         | Wide ResNet-50     |   0.950   |   1.0   |
| PaDiM         | ResNet-18          |   0.891   |  0.982  |
| STFPM         | Wide ResNet-50     |   0.876   |  0.981  |
| STFPM         | ResNet-18          |   0.893   |  0.989  |
| DFM           | Wide ResNet-50     |   0.891   |  0.979  |
| DFM           | ResNet-18          |   0.894   |  0.945  |
| DFKDE         | Wide ResNet-50     |   0.774   |  0.905  |
| DFKDE         | ResNet-18          |   0.762   |  0.669  |
| GANomaly      |                    |   0.421   |  0.413  |


### Pixel-Level AUC

| Model         |                    |    Avg    |  Leather  |
| ------------- | ------------------ | :-------: | :-------: |
| **PatchCore** | **Wide ResNet-50** | **0.980** |   0.991   |
| PatchCore     | ResNet-18          |   0.976   |   0.990   |
| CFlow         | Wide ResNet-50     |   0.971   |   0.993   |
| PaDiM         | Wide ResNet-50     |   0.979   |   0.993   |
| PaDiM         | ResNet-18          |   0.968   | **0.994** |
| STFPM         | Wide ResNet-50     |   0.903   |   0.980   |
| STFPM         | ResNet-18          |   0.951   |   0.991   |


### Image F1 Score

| Model         |                    |    Avg    |  Leather  |
| ------------- | ------------------ | :-------: | :-------: |
| **PatchCore** | **Wide ResNet-50** |   0.974   | **1.000** |
| PatchCore     | ResNet-18          |   0.946   | **1.000** |
| CFlow         | Wide ResNet-50     |   0.932   |  **1.0**  |
| PaDiM         | Wide ResNet-50     |   0.930   |  **1.0**  |
| PaDiM         | ResNet-18          |   0.893   |   0.984   |
| STFPM         | Wide ResNet-50     |   0.973   |   0.974   |
| STFPM         | ResNet-18          | **0.982** |   0.989   |
| DFM           | Wide ResNet-50     |   0.844   |   0.990   |
| DFM           | ResNet-18          |   0.844   |   0.926   |
| DFKDE         | Wide ResNet-50     |   0.844   |   0.905   |
| DFKDE         | ResNet-18          |   0.844   |   0.854   |
| GANomaly      |                    |   0.844   |   0.852   |

## NOTES:

Introduced by Zagoruyko et al. in Wide Residual Networks - [WideResNet](https://paperswithcode.com/method/wideresnet)
