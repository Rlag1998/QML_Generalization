# QML_Generalization

_Authors: Jonathan Kim and Stefan Bekiranov_

This code demonstrates the generalization performance of quantum metric learning ('quantum embedding') classifiers.

The resulting models attempt to classify breast cancer data and ant/bee image data.

- The breast cancer data comes from the UCI ML Breast Cancer (Diagnostic) Dataset and each sample can be classified as either 'benign' or 'malignant'.

- The ant/bee image data comes from the ImageNet Hymenoptera dataset and each image can be classified as either 'ant' or 'bee'.

Data preparation files and resulting txt files can be found in the _embedding_metric_learning_ folder.

The tutorial_QML_antebee_original.py file can be run alongside the antbees_original.py data preparation file to produce results similar to those seen in Lloyd et al.'s 2020 work, 'Quantum Embeddings for Machine Learning' (https://arxiv.org/abs/2001.03622). Hilbert space mutual overlap gram matrices and intermediary scatter plots are produced when run.

The tutorial_QML_antbee_general.py file can be run alongside the antsbees_general.py data preparation file (starting from 512 classical input features) or antbees_general_noresnet.py data preparation file (starting from 150,528 classical input features) when applying PCA to the classical input to the circuit. Hilbert space mutual overlap gram matrices and intermediary scatter plots are produced when run.

The tutorial_QML_breast_cancer.py file can be run alongside the cancer_general.py data preparation file (which includes a PCA step) or the cancer_non-PCA.py data preparation file (which does not include a PCA step) to classify breast cancer data. Hilbert space mutual overlap gram matrices and intermediary scatter plots are produced when run.
