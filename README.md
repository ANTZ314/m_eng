# d_learning

To update GIT:

	git push https://github.com/ANTZ314/d_learning.git --force

## anomaly:

### AE
* Simple reconstructive AutoEncoder
* Reconstructive AutoEncoder for denoising demo
* Reconstructive AutoEncoder for Anomaly Detection score

### mvtec_ad
* train.py
* finetune.py
* test.py
* stored_models 
	* 2 trained models with loss plots
	* [21-02-2022] model trained on kaggle leather dataset

### novelty_det1
* test.py - CHECK IF TESTED??

### novelty_det2
* UNTESTED

## transfer:

### kaggle:

Description:

* class6.py - Modified ResNet-50 model from 100 classes to 6 classes, corresponding to the 6 **Kaggle** leather dataset catagories
* class6a.py - Above code split and modified for direct testing/classification + Training evaluation
* class6b.py - Custom model created & tested with stored pre-processed dataset + test prediction?
	
### transfer

Description:

* class_4.py
* class_5.py
* class_10.py
* custom1.py
* custom2.py
* inception.py
* resnet50.py
* vgg16.py

## ViT

Description: Various ViT code examples tested up to classification (at least)

* ViT_0 - UNTESTED
* ViT_1
	* Classifier tested in Locally & in Colab
	* ViT_1a.py - Original Vision Transformer example from Colab (link in code description)
	* ViT_1b.py - Removed visualisations and other superfluous code
	* ilsvrc2012... - ImageNet Labels for class prediction
* ViT_2 - Classifier tested in Colab
* ViT_3
	* Tested in Colab - Not with custom test image (only Cifar-10)
* ViT_4 
	* Good test metrics at the end (Activation Map / Confusion Matrix)
	* Testing incomplete (Colab / Custom Image)

