# Quantum Metric Learning Generalization

_Authors: Jonathan Kim and Stefan Bekiranov_

_Adapted from work authored by Maria Schuld and Aroosa Ijaz_

_Correspondence: sb3de@virginia.edu_

More details on the following topic can be found in ["Generalization Performance of Quantum Metric Learning Classifiers"](https://doi.org/10.3390/biom12111576).

============================================================================================

This code demonstrates the generalization performance of quantum metric learning ('quantum embedding') classifiers.

The resulting models are used to classify breast cancer data and ant/bee image data.

- The breast cancer data comes from the UCI ML Breast Cancer (Diagnostic) Dataset and each sample can be classified as either 'benign' or 'malignant'.

- The ant/bee image data comes from the ImageNet Hymenoptera dataset and each image can be classified as either 'ant' or 'bee'.

Data preparation files and resulting txt files can be found in the _embedding_metric_learning_ folder.

The tutorial_embedding_generalization_antbee_original.py file can be run after running the antbees_original.py data preparation file to produce results similar to those seen in Lloyd et al.'s 2020 work, ["Quantum Embeddings for Machine Learning"](https://arxiv.org/abs/2001.03622). Hilbert space mutual data overlap gram matrices and intermediary scatter plots are produced when tutorial_embedding_antbee_original.py is run.

The tutorial_embedding_generalization_antbee_pca.py file can be run after running the antsbees_general.py data preparation file (starting from 512 classical input features) or the antbees_general_noresnet.py data preparation file (starting from 150,528 classical input features) when applying PCA to the classical input to the circuit. Hilbert space mutual data overlap gram matrices and intermediary scatter plots are produced when tutorial_embedding_antbee_general.py is run.

The tutorial_embedding_generalization.py file can be run after running the cancer_general.py data preparation file (which includes a PCA step) or the cancer_non-PCA.py data preparation file (which does not include a PCA step) to classify breast cancer data. Hilbert space mutual data overlap gram matrices and intermediary scatter plots are produced when tutorial_embedding_breast_cancer.py is run.

By running the above files, it can be seen in the resulting outputs that:
- Quantum Metric Learning cannot generalize well for test data when using a dataset with too many initial features (e.g., the ants/bees image dataset).
- Quantum Metric Learning results in good classification for datasets that have a much greater number of samples than the intial number of parameters (e.g. the breast cancer diagnostic dataset).
- PCA can help improve generalizability and classification performance when applied to the initial input features of a dataset.

References:
Seth Lloyd, Maria Schuld, Aroosa Ijaz, Josh Izaac, Nathan Killoran: "Quantum embeddings for machine learning" 
arXiv preprint arXiv:2001.03622.

Andrea Mari, Thomas R. Bromley, Josh Izaac, Maria Schuld, Nathan Killoran: "Transfer learning 
in hybrid classical-quantum neural networks" arXiv preprint arXiv:1912.08278.

Jonathan Kim and Stefan Bekiranov: "Generalization of Quantum Metric Learning Classifiers", 
https://doi.org/10.3390/biom12111576.
