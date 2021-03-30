# Using RaNNC for ResNet training

This page briefly explains how to use RaNNC to train enlarged ResNet models.
Before running scripts in this repository, ensure that prerequisites to use RaNNC are satisfied 
(RaNNC requires several libraries including MPI, NCCL, etc.).

## Setup

We modified the training script in [PyTorch examples](https://github.com/pytorch/examples/tree/master/imagenet).
Training data needs to be downloaded following the [instructions](https://github.com/pytorch/examples/tree/master/imagenet) for the original script.
Before using RaNNC, ensure that the original script (`main.py`) correctly works with the dataset.


## Run

The following command shows a usage of the training script for RaNNC.
You can pass some configurations with environment variables.

```python
NP=32 \
MODEL=resnet152x8 BATCH_SIZE=1024 \
DATA_DIR=[PATH_TO_DATASET] \
sh ./run_rannc.sh
```

`MODEL` allows you to choose a predefined model. `BATCH_SIZE` and `DATA_DIR` are 
the batch size and the path to the dataset. 

We defined enlarged versions of ResNet in `resnet_wf.py` so that we can increase the number of filters for convolutions.
The models are named `resnet50x2`, `resnet50x4`, `resnet50x8`, `resnet101x2`, `resnet101x4`, `resnet101x8`,
`resnet152x2`, `resnet152x4`, and `resnet152x8`.
The numbers in the model name indicate the number of layers (50/101/152) and the *width factor* of 
convolution filters (x2/x4/x8).
The width factors are used in some modern networks including [Big Transfer (BiT)](https://github.com/google-research/big_transfer).
For example, the above example trains `resnet152x8`, where the number of layers is 152 and the width factor is 8 
(3.7 billion parameters).

`run_rannc.sh` starts `main_rannc.py` using OpenMPI. The launched processes communicate using NCCL.
Edit MPI configurations and NCCL options in `run_rannc.sh` as needed by your environment.

After the first forward pass is launched, RaNNC starts to analyze the given model to partition it. 
The example output below shows the result of partitioning.
Note that the partitioning may take hours for a model with billion-scale parameters. 

After partitioning is successfully finished, RaNNC shows the overview of the results of the partitioning and the training starts.

```bash
<DPStaging>: Estimated profiles of subgraphs (#partition(s))=3: batch_size=128 ranks=32 pipeline_num=4
<DPStaging>:   graph=MERGE_0_8 repl=16 cp=true fwd_time=164066 bwd_time=455616 ar_time=489086 in_size=77070336 out_size=1027604480 mem=23796547584 (fwd+bwd=13293490176 opt=10503057408)
<DPStaging>:   graph=MERGE_9_13 repl=8 cp=true fwd_time=143170 bwd_time=460086 ar_time=372131 in_size=1027604480 out_size=1027604480 mem=17315659776 (fwd+bwd=9324199936 opt=7991459840)
<DPStaging>:   graph=MERGE_14_24 repl=8 cp=true fwd_time=113227 bwd_time=410329 ar_time=528238 in_size=1027604480 out_size=512000 mem=23978737376 (fwd+bwd=12634906528 opt=11343830848)
<Decomposer>:  Assigned subgraph MERGE_9_13 to rank[28,24,20,4,0,8,12,16]
<Decomposer>:  Assigned subgraph MERGE_0_8 to rank[31,27,23,10,6,30,7,18,2,26,3,14,11,22,15,19]
<Decomposer>:  Assigned subgraph MERGE_14_24 to rank[29,25,21,5,1,9,13,17]
<RaNNCModule>: Saving deployment state to ...
... # Training starts

```

## Modifications for RaNNC

To use RaNNC, we modified `main.py` in [PyTorch examples](https://github.com/pytorch/examples/tree/master/imagenet)
as `main_rannc.py`.
We explain some important modifications below.


### Training mode

`main_rannc.py` skips the validation during training because it takes a long time to switch off the training mode when using RaNNC.

RaNNC uses a traced graph resulting from PyTorch's `torch.jit.trace()`.
Once the graph is generated, however, the training mode does not affect the graph.
Therefore, RaNNC *re-*runs `torch.jit.trace()` and determines the model partitioning again when the training mode is changed.
This often takes a very long time.

To save time for partitioning, you can use a precomputed partitioning.
By setting `save_deployment=true` and `deployment_file=[PATH_TOFILE]`, RaNNC saves the partitioning result and 
allocation to a file.
RaNNC can load the file by passing the path to the file to the argument `load_deployment` of RaNNCModule.  

### Gradient accumulation

In `main_rannc.py`, we disabled gradient accumulation for simplicity.
To implement gradient accumulation, see the [BERT pretraining example](../bert/README.md).


