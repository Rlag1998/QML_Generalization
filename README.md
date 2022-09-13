# QML_Generalization

_Authors: Jonathan Kim and Stefan Bekiranov_

This code demonstrates the generalization performance of quantum metric learning ('quantum embedding') classifiers.

The resulting models attempt to classify breast cancer data and ant/bee image data.

- The breast cancer data comes from the UCI ML Breast Cancer (Diagnostic) Dataset and each sample can be classified as either 'benign' or 'malignant'.

- The ant/bee image data comes from the ImageNet Hymenoptera dataset and each image can be classified as either 'ant' or 'bee'.

Data preparation steps and resulting prepared files can be found in the _embedding_metric_learning_ folder.

The tutorial_QML_antebee_original.py file can be run to produce results similar to those seen in Lloyd et al.'s 2020 work, 'Quantum Embeddings for Machine Learning' (https://arxiv.org/abs/2001.03622). Mutual overlap gram matrices and intermediary scatter plots are produced.

The tutorial_QML_antbee_general.py file can be used when applying PCA to the classical input to the circuit prior to classification.

The tutorial_QML_breast_cancer.py file can be used to classify breast cancer data, both with and without a prior PCA step.
