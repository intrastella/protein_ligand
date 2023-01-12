# Ligand - Protein 
A repo for experimenting on creating ligands w/ ML.

Architecture consits of GAN and Transformer model. Smiles data was used for ligands and sequencer for proteins. 

**Database**: 
1. ATOM3D - [zenodo.org](https://zenodo.org/record/4914718#.Y7_vRafMKV6)
2. UniProt - [uniprot.org](https://www.uniprot.org/help/downloads#embeddings)


<h2> Organization </h2>

You can execute a run via `./run_experiment.py` and see metrics and results via Tensorboard by running:

> 1. Model and path to dataset will be required:  `--model [model name] --path [dataset path]` <br>
> 2. After a run write in command line:
`tensorboard --logdir [path to runs folder]` and open the link provided to see results.

Your SQL credentials needs to be saved in `./data/data_cinf.yaml.py` :

> USER: <br>
> PASSWORD: <br>
> HOST: <br>
> PORT:

<h2> Goals </h2>
Create a proper structure for my own DL project.
