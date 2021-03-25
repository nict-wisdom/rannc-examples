# Using RaNNC for BERT pretraining

This page briefly explains how to use RaNNC to train BERT models.
Before running scripts in this repository, ensure that prerequisites to use RaNNC are satisfied 
(RaNNC requires some libraries including MPI, NCCL, etc.).

## Setup

We use BERT [pretraining scripts by NVIDIA](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT).
Clone the repository and install required modules.
We tested RaNNC with the revision shown below.
Before using RaNNC, follow the steps described in the original repository to set up the datasets and make sure the original script correctly works.
You may also need [Apex amp](https://nvidia.github.io/apex/amp.html) to enable mixed-precision training.

```bash
git clone https://github.com/NVIDIA/DeepLearningExamples
git checkout 65211bd9621c781bc29d3a4e1c3c287983645d50
cd PyTorch/LanguageModeling/BERT
pip install -r requirements.txt
```


To use RaNNC, copy the following files in this repository.
*Path*s show the paths where the files should be placed.

- rl (Path: PyTorch/LanguageModeling/BERT/)
- run_pretraining_rannc.py (Path: PyTorch/LanguageModeling/BERT/)
- modeling.py (Path: PyTorch/LanguageModeling/BERT/)
- run_pretraining_rannc.sh (Path: PyTorch/LanguageModeling/BERT/scripts/)

`run_pretraining_rannc.sh` requires some variables to be set for your environment.
The following shows an example to start the script.

```bash
BERT_CONFIG=my_bert_config.json
CODEDIR=/.../DeepLearningExamples/PyTorch/LanguageModeling/BERT \
BERT_PREP_WORKING_DIR=/.../DeepLearningExamples/PyTorch/LanguageModeling/BERT/data/prep \
sh ./scripts/run_pretraining_rannc.sh 
```

`BERT_CONFIG` allows you to change configurations of your BERT model.
For example, the configuration below produces an enlarged BERT model with 4.9 billion parameters.
`megatron_residual_connection`, `scale_query_key`, and `deep_weight_init` are options to enable
modifications proposed for [Megatron-LM](https://github.com/NVIDIA/Megatron-LM).
 
```json
{
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 8192,
  "max_position_embeddings": 512,
  "num_attention_heads": 32,
  "num_hidden_layers": 96,
  "type_vocab_size": 2,
  "vocab_size": 30522,
  "megatron_residual_connection": true,
  "scale_query_key": true,
  "deep_weight_init": true
}
```

## Run

`run_pretraining_rannc.sh` starts `run_pretraining_rannc.py` using MPI.
After the first forward pass is launched, RaNNC starts to analyze the given model and tries to partition it. 
The example output below shows the result of partitioning.
Note that the partitioning may take hours for a model with billion-scale parameters. 

```
<RaNNCModule>: Tracing model ...
<RaNNCModule>: Converting torch model to IR ...
<RaNNCModule>: Running profiler ...
<RaNNCModule>: Profiling finished
<RaNNCModule>: Assuming batch size: 64
<Decomposer>: Decomposer: ml_part
<Decomposer>: Available device memory: 30680756224
# ... often takes hours ...
<DPStaging>: Estimated profiles of subgraphs (#partition(s))=4: batch_size=64 ranks=8 pipeline_num=8
<DPStaging>:   graph=MERGE_0_6 repl=2 cp=true fwd_time=35409 bwd_time=110886 ar_time=233871 in_size=196608 out_size=167788544 mem=27084284928 (fwd+bwd=6998778880 opt=20085506048)
<DPStaging>:   graph=MERGE_7_12 repl=2 cp=true fwd_time=34849 bwd_time=110344 ar_time=222027 in_size=167788544 out_size=67108864 mem=25738417152 (fwd+bwd=6670242816 opt=19068174336)
<DPStaging>:   graph=MERGE_13_21 repl=2 cp=true fwd_time=36715 bwd_time=115215 ar_time=234537 in_size=67125248 out_size=67108864 mem=27194624000 (fwd+bwd=7052134400 opt=20142489600)
<DPStaging>:   graph=MERGE_22_29 repl=2 cp=true fwd_time=36657 bwd_time=112948 ar_time=235245 in_size=67191296 out_size=4 mem=27413166756 (fwd+bwd=7209715332 opt=20203451424)
<Decomposer>:  Assigned subgraph MERGE_13_21 to rank[7,3]
<Decomposer>:  Assigned subgraph MERGE_22_29 to rank[6,2]
<Decomposer>:  Assigned subgraph MERGE_7_12 to rank[4,0]
<Decomposer>:  Assigned subgraph MERGE_0_6 to rank[5,1]
<RaNNCModule>: Saving deployment state to /.../prof_cache/prof_deployment_...
<RaNNCModule>: Routes verification passed.
<RaNNCModule>: Saving graph profiles to /.../prof_cache/graph_profile_...
<RaNNCModule>: Partitioning verification passed.
<RaNNCModule>: RaNNCModule is ready. (rank0)
...
Iteration:   0%|          | 1/25570 [4:16:00<109094:36:08, 15360.03s/it]
...
```


## Modifications for RaNNC

To enable RaNNC, we modified `run_pretraining_rannc.py`.
We explain some important modifications below.

### Import RaNNC

First you need to import RaNNC.

```python
import pyrannc
```

### Set profiling flag

Set `torch._C._jit_set_profiling_executor(True)`.
This is necessary to ensure coherent results of dropout.
(See [this issue](https://github.com/pytorch/pytorch/issues/41909))

```python
torch._C._jit_set_profiling_executor(True)
```

### Combine computation of loss value

The original script separates the computation of a loss value from the model (`BertForPreTraining`).
To make the computation distributed using RaNNC, we combined it with the model.

```python
class BertPretrainingCriterion(torch.nn.Module):
...
    def forward(self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels):
        masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size).float(), masked_lm_labels.view(-1))
        next_sentence_loss = self.loss_fn(seq_relationship_score.view(-1, 2).float(), next_sentence_labels.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss

class BertForPreTrainingWithCriterion(modeling.BertForPreTraining):
    def __init__(self, config):
        super(BertForPreTrainingWithCriterion, self).__init__(config)
        self.criterion = BertPretrainingCriterion(config.vocab_size)

    def forward(self, input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sentence_labels):
        prediction_scores, seq_relationship_score = super().forward(input_ids, token_type_ids, attention_mask)
        return self.criterion(prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels)
```

### Gradient accumulation

The original script implements gradient accumulation, which delays parameter update over multiple mini-batches.
Since RaNNC implicitly *allreduce*s (sums) gradients across processes, we need to disable the implicit allreduce using
`pyrannc.delay_grad_allreduce`.

```python
pyrannc.delay_grad_allreduce(args.allreduce_post_accumulation)
```

In `take_optimizer_step()`, the original script explictly performs allreduce by directly manipulating data on CUDA device memory.
However, it does not work with RaNNC because RaNNC relies on its own memory management.
Instead, you can use a utility function `allreduce_grads_rannc()`.

```python
had_overflow = allreduce_grads_rannc(model, optimizer, 1./args.gradient_accumulation_steps)
```

This function *allreduce*s gradients across processes.
When the mixed precision using *apex amp* is enabled, it properly manages the gradient scaling and detects overflow.


### Gradient clipping

Since each process has only a part of gradients, such PyTorch's functions as `torch.nn.utils.clip_grad` produce incorrect results.
Therefore, RaNNCModule has a dedicated method to clip gradients.

```python
model.clip_grad_norm(1.0)
```

### Saving/Loading checkpoints

You can save or load model parameters using `state_dict` and `load_state_dict` of `RaNNCModule`.
Note that you can call `state_dict` from all processes to collect parameters across processes.

```python
state_dict = model.state_dict(sync_all_ranks=False, no_hook=True)
```

When `sync_all_ranks=False`, the return value is valid only on rank 0.
`no_hook=True` disables hooks on `state_dict`.
This option aims to disable an apex amp's hook that converts all parameters to FP32.

RaNNCModule's constructor also modifies the optimizer's `state_dict` and `load_state_dict` for RaNNC.
`state_dict` collects the state of the optimizer across all processes.
Therefore, the function must be called from all processes.

```python
global_opt_state_dict = optimizer.state_dict()
```

The return value of `optimizer.state_dict()` contains the state for all parameters.
When properly loading it, `load_state_dict` must be called after the model partitioning is determined.
The following shows a typical usage of `load_state_dict`, where `load_state_dict` is called once after a backward pass.

```python
# After backward pass
if global_optimizer_state:
     optimizer.load_state_dict(global_optimizer_state)
     global_optimizer_state = None
```


