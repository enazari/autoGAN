## AutoGAN: An Automated Human-Out-Of-The-Loop Approach for Training GANs

This repository has all the code used in the experiments carried out in the paper *"AutoGAN: An Automated Human-Out-Of-The-Loop Approach for Training GANs"* [1].


### AutoGAN Algorithm

To use GANs, researchers and developers need to answer the question: “Is the GAN sufficiently trained?". However, understanding when a GAN is well trained for a given problem is a challenging and laborious task that usually requires monitoring the training process and human intervention for assessing the quality of the GAN generated outcomes.

**AutoGAN is a human-out-of-the-loop algorithm**, where the usage of quantitative measures is fully automated. AutoGAN requires minimal human intervention at the design phase and is applicable to different data modalities (tabular and images). Our extensive experiments show the clear advantage of using AutoGAN, even when compared to GANs trained under a thorough human visual inspection of the generated images.


### Repository Organization

This repository is organized as follows:

* **main.py** - includes the classifier model and root dataset, as well as all parameters for an instance of AutoGAN.
* **utils.py** - includes all AutoGAN classes, with each oracle having its own class. It also includes an implementation of CGAN and utility classes for generating datasets from the root dataset.
* **Datasets folder** - this directory contains two folders: one with the root datasets and another one with the derived datasets. The root datasets of all tabular datasets, alongside Kuzushiji-MNIST are provided in the **Root Dataset folder**. The remaining imagery datasets can be imported from tensorflow.keras.datasets library. The **Derived Datasets folder** contains all the derived datasets with different majority class count and imbalance ratios that were used in our experiments.

### Requirements

The experimental design was implemented in Python 3.7 with the following libraries:
tensorflow 2.7.0
keras 2.7.0
scikit_learn 1.0.1
numpy 1.21.0
pandas 1.3.5
matplotlib 3.2.2



To replicate the experiments, ensure that you have a working installation of the libraries listed above. Then, set the input dataset in the main.py file and configure the classifier model according to the dataset (the specifications of the classifier used for each root dataset is mentioned in the paper). For example, if the root dataset is set to fashion MNIST, and you want to use classes 7 and 9 as the majority and minority classes, and you want to use the CDS oracle for AutoGAN, you need to first import fashion MNIST dataset in main.py, then you can execute the following command:

python main.py --dataset_name='fmnist79' --oracle_name='CDS' --majority_class=7 --minority_class=9

*Once the root dataset is imported in main.py, the program will generate the derived datasets for the experiments on the fly.
*****

### References
[1] Nazari, E. and Branco, P. and Jourdan, G.V. (2023) *"AutoGAN: An Automated Human-Out-Of-The-Loop Approach for Training Generative Adversarial Networks"* Mathematics  (submitted).
