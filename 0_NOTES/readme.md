# ANOMALY CODE TESTING:

### AE

AutoEncoder Examples:

* **AE_1.py** - Image Reconstruction (FMNIST Dataset)
* **AE_2.py** - De-Noising images (FMNIST Dataset)
* **AE_3.py** - Anomaly detection (ECG Dataset)

<br>
---
### KAGGLE

* All data modified to test on **Kaggle dataset** (6 training folders + 6 test folders)
* Stored **pre-processed** files ".npy". Can be callled for training
* Stored **trained-ResNet model**. Can be called for immediateprediction.
* **class_6.py** - Original Colab version to run loacally in Spyder
* **class_6a.py** - Edited above to load pre-processed dataset & train ResNet model + evaluation + prediction
* **class_6b.py** - Edited above to load pre-processed dataset & train Custom built model + prediction

<br>
---
### transfer:

* transfer learning course - adapted 'n' class classifiers
* 2, 4, 5, 10 classes + inception + resnet + vgg16
* 2x custom networks
* ResNet-50 modified to 6 classes for 'Kaggle' dataset (3600 leather images)
--> **RUN IN SPYDER - UNTESTED**

<br>
---
### MVTEC_AD:

There is a total of 5 models based on the **Convolutional Auto-Encoder** (CAE) architecture implemented in this project:

* **mvtecCAE** is the model implemented in the [MVTec Paper](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/mvtec_ad.pdf)
* **baselineCAE** is inspired by: [baselineCAE](https://github.com/natasasdj/anomalyDetection)
* **inceptionCAE** is inspired by: [inceptionCAE](https://github.com/natasasdj/anomalyDetection)
* **resnetCAE** is inspired by: [resnetCAE](https://arxiv.org/pdf/1606.08921.pdf)
* **skipCAE** is inspired by: [skipCAE](https://arxiv.org/pdf/1606.08921.pdf)
* **TEST:**
	* Run in Spyder and working on MVTEC Leather dataset
	* ?? Tabulating results & tweaking code?
	* ?? Editing overall code in any way?
	* Test on other datasets (without Ground truths)

<br>
---
### VIT:

* **VT_ADL** (~Did Not Run~):
	* **ERROR**: TypeError: 'NoneType' object is not subscriptable
	* Our proposed model is a combination of a reconstruction-based approach and patch embedding. 
	* The use of transformer networks helps preserving the spatial information of the embedded patches, which is later processed by a **Gaussian mixture density network** to localize the anomalous areas. 
* **ViT_0** 
	* **DeiT** - Data-Efficient Image Transformers
	* **CaiT** - (Going deeper with Image Transformers)
	* **ResMLP** - (ResMLP: Feedforward networks for image classification with data-efficient training)
	* **PatchConvnet** - (Augmenting Convolutional networks with attention-based aggregation)
	* **3Things** - (Three things everyone should know about Vision Transformers)
	* **DeiT III** - (DeiT III: Revenge of the ViT)
* **ViT_1** 
	* Tested in Colab & Local - Custom image classified correctly
* **ViT_2** 
	* Tested in Colab with CIFAR-100 dataset 
	* With GPU - 1h53min to fit model COLAB)
	* Trained model stored - need to test
* **ViT_3** 
	* Tested in Colab with CIFAR-10 dataset - Classification??
* **ViT_4** 
	* Large originial file - break down to functions??
	* Tested in Colab with CIFAR-10 dataset - Classification??

<br>
---
### zJupyter:

Test files to be run in Jupyter Notebook

**Run Jupyter Notebook:**
```
cd /home/antz/Desktop/models/zJupyter/
jupyter notebook file.ipynb
```

<br>
## DATASETS:

#### ViT  Models

* Most models tested on CIFAR-10 & CIFAR-100
* **CIFAR-10**:
	* 60 000 32x32 colour images in (6 000 per class)
	* 50 000 training images & 10 000 test images
	* airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
* **CIFAR-100**:
	* 100 classes containing 600 images each
	* 500 training images & 100 testing images per class
	* Class list online...

#### kaggle: 

* **Description**: 3600 leather images [227x227p]
* 6 Classes (600 samples of each):
	* Folding marks, Grain off, Growth marks, 
	* loose grains, non defective, pinhole
* Used in Moganam(2022)
* [kaggle download](https://www.kaggle.com/datasets/praveen2084/leather-defect-classification)


#### data:
* **Description**: Anomaly Course Dog_vs_Cat Dataset
* test - 12 500 images (shuffled cats & dogs)
* train - 25 000 images (12 500 cats + 12 500 dogs)

#### leather:
* MVTEC Leather Dataset
* test - 6 folders (124 images total)
* train - 245 mixed leather images
* ground - 5 folders (92 images total)
* [LINK](https://www.mvtec.com/company/research/datasets/mvtec-ad)

    ├── leather	
    │   ├── ground_truth
    │   │   ├── color
    │   │   ├── cut
    │   │   ├── fold
    │   │   ├── glue
    │   │   └── poke
    │   ├── test
    │   │   ├── color
    │   │   ├── cut
    │   │   ├── fold
    │   │   ├── glue
    │   │   └── poke
    │   │   └── good
    │   └── train
    │       └── good

