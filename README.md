# Ligand - Protein 
A repo for experimenting on creating ligands w/ ML.

Architecture consits of GAN and Transformer model. Smiles data was used for ligands and sequencer for proteins. 

**Databases**:

1. BindingDB - [bindingdb.org](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp) - [Discription](https://www.bindingdb.org/bind/chemsearch/marvin/BindingDB-SDfile-Specification.pdf)
2. ATOM3D - [zenodo.org](https://zenodo.org/record/4914718#.Y7_vRafMKV6)
3. UniProt - [uniprot.org](https://www.uniprot.org/help/downloads#embeddings)


<h2> Organization </h2>

You can execute a run via `./run_experiment.py` and see metrics and results via Tensorboard by running:

> 1. Model and path to dataset will be required:  `--model [model name] --path [dataset path]` <br>
> 2. After a run write in command line:
`tensorboard --logdir [path to runs folder]` and open the link provided to see results.

Your SQL credentials needs to be saved in `./data/data_conf.yaml.py` :

> USER: <br>
> PASSWORD: <br>
> HOST: <br>
> PORT: <br>
> DATABSE

<h2> Goals </h2>
Create a proper structure for my own DL project.
