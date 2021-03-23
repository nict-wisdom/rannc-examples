# BERT

This page briefly explains how to use RaNNC to train BERT models.

## Setup

We use BERT [pretraining scripts by NVIDIA](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT).
Clone the repository and install required modules.
We tested RaNNC with the revision shown below.
Before using RaNNC, follow the steps to set up the datasets and make sure the original script correctly works.

```bash
git clone https://github.com/NVIDIA/DeepLearningExamples
git checkout 65211bd9621c781bc29d3a4e1c3c287983645d50
cd PyTorch/LanguageModeling/BERT
pip install -r requirements.txt
```

To use RaNNC, place the following files with ones offered in this repository.

- run_pretraining_rannc.py (Path: PyTorch/LanguageModeling/BERT/)
- run_pretraining_rannc.sh (Path: PyTorch/LanguageModeling/BERT/scripts/)




