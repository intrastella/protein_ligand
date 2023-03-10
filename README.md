# Ligand - Protein 
A repo for experimenting on creating ligands w/ high binding affinity to a target protein using ML.

Architecture consits of GAN and Transformer model. Smiles data was used for ligands and sequences for proteins. 

**Databases**:

1. BindingDB - [bindingdb.org](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp) - [Discription](https://www.bindingdb.org/bind/chemsearch/marvin/BindingDB-SDfile-Specification.pdf)
2. ATOM3D - [zenodo.org](https://zenodo.org/record/4914718#.Y7_vRafMKV6)
3. UniProt - [uniprot.org](https://www.uniprot.org/help/downloads#embeddings)


<h2> Organization </h2>

You can execute a run via `./run_experiment.py` and see metrics and results via Tensorboard by running:

> 1. Model will be required and path to dataset if you haven't stored them in a sql table from a previous experiment with this model:  `--model [model name] --data_path [dataset path]` <br>
> 2. If it's your first experiment w/ this model you can save the given dataset in a table so they won't need to be processed again for another experiment by additionally writing `-sql [-ho HOST] [-po PORT] [-db DATABASE]` in command line.
> 3. If you trained that model already several times you can specify a checkpoint by giving the absolute path `[--ckpt_path CKPT_PATH]` . As default it will choose the best checkpoint - meaning the one with the least amount of loss.
> 4. If you want to use a saved dataset in a table enter `-sql` without specifying the sql connection info.
> 5. After a run write in command line:
`tensorboard --logdir [path to runs folder]` and open the link provided to see results.

Your SQL credentials will be asked when executing an experiment:

> USER: <br>
> PASSWORD: <br>

<h2> Goals </h2>
Create a proper structure for my own DL project.
