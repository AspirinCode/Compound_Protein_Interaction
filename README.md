# Compound-protein Interaction Prediction

The code for "Compound-protein Interaction Prediction
with End-to-end Learning of Neural Networks for Graphs and Sequence" (Bioinformatics, 2018).

The code provides two major scripts:

- code/preprocess_data.py creates the input data of compound-protein interaction
(see dataset/sample/original/smiles_sequence_interaction.txt).
- code/run_training.py trains a neural network to predict the compound-protein interaction
using the above created input data.

The following paper describes the details of the neural network:

[Compound-protein Interaction Prediction with End-to-end Learning of Neural Networks for Graphs and Sequences](https://academic.oup.com/bioinformatics/advance-article-abstract/doi/10.1093/bioinformatics/bty535/5050020?redirectedFrom=PDF)


## Requirement

The code requires:

- Chainer (version 3.2.0)
- scikit-learn
- RDKit

(i) To create the input data, run the code as follows:
```
cd code
bash preprocess_data.sh
```

(ii) Then, to train a neural network with the preprocessed data, run the code as follows:
```
bash run_training.sh
```

You can change the hyperparameters in preprocess_data.sh and run_training.sh.


## Train your data of compound-protein interaction
In the directory of dataset/sample/original/, we have "smiles_sequence_interaction.txt."
In this file, 1 means that the compound-protein pair interacts
and 0 means that the compound-protein pair does not interact.
If you prepare your data with the same format as "smiles_sequence_interaction.txt"
in a new directory (e.g., dataset/yourdata/original/),
you can train a neural network by (i) run preprocess_data.sh and (ii) run_training.sh.


## TODO

- Provide a pre-trained model with a large dataset.
