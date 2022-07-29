# Learning the language of rpoB evolution and escape

Extending [Learning the language of viral evolution and escape](https://www.science.org/doi/10.1126/science.abd7331) to predict multi-drug resistance mutations in bacterial RNA Polymerase B.

### Data
RpoB data for pretraining was selected from UniProtKB under the taxonomic label of gammaproteobacteria, as these strains have been the ones most extensively shown to be prone to antibiotic resistance. Protein isoforms are excluded and redundant sequences are eliminated. The data is stored in the `uniprot_gpb_rpob.fasta` file. 

### Dependencies
The major Python package requirements and their tested versions are in requirements.txt.

### Experiments
Our experiments were run with Python version 3.7 on a gcloud machine with `IMAGE_URI='gcr.io/cloud-ml-public/training/pytorch-gpu.1-6`.

Key results from our experiments can be found in the [`results/`](results) directory and can be reproduced with the commands in the [`scripts/`](scripts) for either gcloud or local environments. The [`saved_models/`](saved_models) directory contains weight files for a trained transformer. 

Pretraining and subsequent analysis can both be done by passing arugments into `trainer/main.py`.

### Analysis

A fasta file with wildtype sequence and a json file with regions of interest for subsequent CSCS analysis are present in the [`escape_validation/`](escape_validation) folder along with a file `file_column_dict.json` that tracks endpoint columns for each individual DMS study with which to compute correlations.




