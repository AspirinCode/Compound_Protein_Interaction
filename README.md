# Compound_Protein_Interaction

The code for "Compound-protein Interaction Prediction with End-to-end Learning of Neural Networks for Graphs and Sequence" (Bioinformatics, 2018).

The code provides two major functions:

- code/preprocess_data.sh creates the input data of compound protein interactions
(see dataset/sample/original).
- code/run_training.sh trains a neural network to predict the compound protein interaction
using the preprocessed input data.

The following paper describes the details of the neural network:

[Compound-protein Interaction Prediction with End-to-end Learning of Neural Networks for Graphs and Sequences](https://academic.oup.com/bioinformatics/advance-article-abstract/doi/10.1093/bioinformatics/bty535/5050020?redirectedFrom=PDF)

## Requirement

The code requires:

- Chainer (version 3.2.0)
- scikit-learn
- RDKit

To create the input data for a neural network, run the code as follows:
```bash
cd code
bash preprocess_data.sh
```

Then, to train a neural network with the preprocessed data, run the code as follows:
```bash
bash run_training.sh
```

## Train your data of compound protein interaction
In the directory of dataset/sample/original, we have smiles_sequence_interaction.txt.
For other data of compound protein interaction,
if you prepare the data with the same format as smiles_sequence_interaction.txt in a new directory,
you can train a neural network by run preprocess_data.sh and run_training.sh.
