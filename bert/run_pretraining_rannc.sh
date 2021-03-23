#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2021 Data-driven Intelligent System Research Center (DIRECT), National Institute of Information and Communications Technology (NICT).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/python3.7/site-packages/torch/lib:$LD_LIBRARY_PATH
RL=/home/IAL/mtnk/work/pyrannc/bin/rannc_launcher
PYTHON=${PYTHON:-python}

export PYTHON
echo "python_bin=${PYTHON}"

BERT_PREP_WORKING_DIR=/share01/mtnk/work/DeepLearningExamples/PyTorch/LanguageModeling/BERT/data/prep

echo "conda_env=${CONDA_ENV}"

python -c "import pkg_resources; print(pkg_resources.require('pyrannc'))"

NP=${NP:-8}
NGPUS=${NGPUS:-8}
TIMESTAMP=${TIMESTAMP:-0}
PHASE1_ONLY=${PHASE1_ONLY:-0}
PHASE2_ONLY=${PHASE2_ONLY:-0}
TRAIN_STEPS_P1=${TRAIN_STEPS_P1:-1000000}
BATCH_SIZE_PROC_P1=${BATCH_SIZE_PROC_P1:-8}
ACC_STEPS_P1=${ACC_STEPS_P1:-1}
TRAIN_STEPS_P2=${TRAIN_STEPS_P2:-100000}
BATCH_SIZE_PROC_P2=${BATCH_SIZE_PROC_P2:-4}
ACC_STEPS_P2=${ACC_STEPS_P2:-1}
LR=${LR:-1e-4}
PREC=${PREC:-fp16}

CONSOLIDATE_GRADS=${CONSOLIDATE_GRADS:-false}
SKIP_GRAD_SCALING=${SKIP_GRAD_SCALING:-false}

DECOMPOSER=${DECOMPOSER:-ml_part}
DO_UNCOARSENING=${DO_UNCOARSENING:-true}
DO_COARSENING=${DO_COARSENING:-true}
SAVE_DEPLOYMENT=${SAVE_DEPLOYMENT:-true}
LOAD_DEPLOYMENT=${LOAD_DEPLOYMENT:-false}
SAVE_GRAPH_PROFILE=${SAVE_GRAPH_PROFILE:-true}
LOAD_GRAPH_PROFILE=${LOAD_GRAPH_PROFILE:-false}

MIN_PIPELINE=${MIN_PIPELINE:-1}
MAX_PIPELINE=${MAX_PIPELINE:-32}
MAX_PARTITION_NUM=${MAX_PARTITION_NUM:-30}
MEM_MARGIN=${MEM_MARGIN:-0.1}
TRACE_EVENTS=${TRACE_EVENTS:-false}
DP_SEARCH_ALL=${DP_SEARCH_ALL:-false}
MIN_PIPELINE_BS=${MIN_PIPELINE_BS:-1}

PART=${PART:-1}
REPL=${REPL:-8}
PIPE=${PIPE:-1}

PARAMS="NP${NP}"
PARAMS+="_MINPL${MIN_PIPELINE}"
PARAMS+="_MAXPL${MAX_PIPELINE}"
PARAMS+="_MEM_MARGIN${MEM_MARGIN}"
PARAMS+="_P1BS${BATCH_SIZE_PROC_P1}_P1ACC${ACC_STEPS_P1}_P1STEPS${TRAIN_STEPS_P1}_P1ONLY${PHASE1_ONLY}"
PARAMS+="_P2BS${BATCH_SIZE_PROC_P2}_P2ACC${ACC_STEPS_P2}_P2STEPS${TRAIN_STEPS_P2}_P2ONLY${PHASE2_ONLY}"
PARAMS+="_CG${CONSOLIDATE_GRADS}_SGS${SKIP_GRAD_SCALING}"
PARAMS+="_PREC${PREC}"
PARAMS+="_PT${MAX_PARTITION_NUM}_DPALL${DP_SEARCH_ALL}"
PARAMS+="_MPBS${MIN_PIPELINE_BS}"
PARAMS+="_T${TIMESTAMP}"

PROF_CACHE_DIR=/share01/mtnk/prof_cache
mkdir -p ${PROF_CACHE_DIR}
DEPLOYMENT_FILE=${DEPLOYMENT_FILE:-"${PROF_CACHE_DIR}/prof_deployment_${PARAMS}.bin"}
GRAPH_PROFILE_FILE=${GRAPH_PROFILE_FILE:-"${PROF_CACHE_DIR}/graph_profile_${PARAMS}.bin"}

train_batch_size=${1:-${BATCH_SIZE_PROC_P1}}
learning_rate=${2:-${LR}}
precision=${3:-${PREC}}
num_gpus=${4:-${NGPUS}}
warmup_proportion=${5:-"0.01"}
train_steps=${6:-${TRAIN_STEPS_P1}}
save_checkpoint_steps=${7:-10000}
resume_training=${8:-"false"}
create_logfile=${9:-"true"}
accumulate_gradients=${10:-"true"}
gradient_accumulation_steps=${11:-${ACC_STEPS_P1}}
seed=${12:-42}
job_name=${13:-"bert_lamb_pretraining"}
allreduce_post_accumulation=${14:-"true"}
allreduce_post_accumulation_fp16=${15:-"true"}
train_batch_size_phase2=${17:-${BATCH_SIZE_PROC_P2}}
learning_rate_phase2=${18:-"${LR}"}
warmup_proportion_phase2=${19:-"0.01"}
train_steps_phase2=${20:-${TRAIN_STEPS_P2}}
gradient_accumulation_steps_phase2=${21:-${ACC_STEPS_P2}}
DATASET=hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus # change this for other datasets
DATA_DIR_PHASE1=${22:-$BERT_PREP_WORKING_DIR/${DATASET}/}
BERT_CONFIG=${BERT_CONFIG:-bert_config.json}
CODEDIR=${24:-"/share01/mtnk/work/pyrannc-examples/bert/DeepLearningExamples/PyTorch/LanguageModeling/BERT"}
init_checkpoint=${25:-"None"}
RESULTS_DIR=$CODEDIR/results
CHECKPOINTS_DIR=$RESULTS_DIR/rannc_${PARAMS}/checkpoints

mkdir -p $CHECKPOINTS_DIR


if [ ! -d "$DATA_DIR_PHASE1" ] ; then
   echo "Warning! $DATA_DIR_PHASE1 directory missing. Training cannot start"
fi
if [ ! -d "$RESULTS_DIR" ] ; then
   echo "Error! $RESULTS_DIR directory missing."
   exit -1
fi
if [ ! -d "$CHECKPOINTS_DIR" ] ; then
   echo "Warning! $CHECKPOINTS_DIR directory missing."
   echo "Checkpoints will be written to $RESULTS_DIR instead."
   CHECKPOINTS_DIR=$RESULTS_DIR
fi
if [ ! -f "$BERT_CONFIG" ] ; then
   echo "Error! BERT large configuration file not found at $BERT_CONFIG"
   exit -1
fi

PREC_ARG=""
if [ "$precision" = "fp16" ] ; then
   PREC_ARG="--fp16"
elif [ "$precision" = "fp32" ] ; then
   PREC_ARG=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

ACCUMULATE_GRADIENTS=""
if [ "$accumulate_gradients" == "true" ] ; then
   ACCUMULATE_GRADIENTS="--gradient_accumulation_steps=$gradient_accumulation_steps"
fi

CHECKPOINT=""
if [ "$resume_training" == "true" ] ; then
   CHECKPOINT="--resume_from_checkpoint"
fi

ALL_REDUCE_POST_ACCUMULATION=""
if [ "$allreduce_post_accumulation" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION="--allreduce_post_accumulation"
fi

ALL_REDUCE_POST_ACCUMULATION_FP16=""
if [ "$allreduce_post_accumulation_fp16" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION_FP16="--allreduce_post_accumulation_fp16"
fi

INIT_CHECKPOINT=""
if [ "$init_checkpoint" != "None" ] ; then
   INIT_CHECKPOINT="--init_checkpoint=$init_checkpoint"
fi

echo $DATA_DIR_PHASE1
INPUT_DIR=$DATA_DIR_PHASE1

MPI_OPTS="-np ${NP}"
MPI_OPTS+=" --tag-output"
MPI_OPTS+=" -bind-to none -map-by slot"
MPI_OPTS+=" --mca pml ucx --mca btl ^vader,tcp,openib"
MPI_OPTS+=" -x PATH -x LD_LIBRARY_PATH"
MPI_OPTS+=" -x UCX_MEMTYPE_CACHE=n -x UCX_NET_DEVICES=mlx5_2:1"
MPI_OPTS+=" -x NCCL_DEBUG=WARNING -x NCCL_IB_HCA=mlx5_2"
MPI_OPTS+=" -x PYTHON"
MPI_OPTS+=" -x RANNC_DECOMPOSER=${DECOMPOSER}"
MPI_OPTS+=" -x RANNC_DO_UNCOARSENING=${DO_UNCOARSENING}"
MPI_OPTS+=" -x RANNC_DO_COARSENING=${DO_COARSENING}"
MPI_OPTS+=" -x RANNC_MIN_PIPELINE=${MIN_PIPELINE}"
MPI_OPTS+=" -x RANNC_MAX_PIPELINE=${MAX_PIPELINE}"
MPI_OPTS+=" -x RANNC_MAX_PARTITION_NUM=${MAX_PARTITION_NUM}"
MPI_OPTS+=" -x RANNC_MEM_MARGIN=${MEM_MARGIN}"
MPI_OPTS+=" -x RANNC_SAVE_DEPLOYMENT=${SAVE_DEPLOYMENT}"
MPI_OPTS+=" -x RANNC_LOAD_DEPLOYMENT=${LOAD_DEPLOYMENT}"
MPI_OPTS+=" -x RANNC_DEPLOYMENT_FILE=${DEPLOYMENT_FILE}"
MPI_OPTS+=" -x RANNC_SAVE_GRAPH_PROFILE=${SAVE_GRAPH_PROFILE}"
MPI_OPTS+=" -x RANNC_LOAD_GRAPH_PROFILE=${LOAD_GRAPH_PROFILE}"
MPI_OPTS+=" -x RANNC_GRAPH_PROFILE_FILE=${GRAPH_PROFILE_FILE}"
MPI_OPTS+=" -x RANNC_CONSOLIDATE_GRADS=${CONSOLIDATE_GRADS}"
MPI_OPTS+=" -x RANNC_SKIP_GRAD_SCALING=${SKIP_GRAD_SCALING}"
MPI_OPTS+=" -x RANNC_TRACE_EVENTS=${TRACE_EVENTS}"
MPI_OPTS+=" -x RANNC_DP_SEARCH_ALL=${DP_SEARCH_ALL}"
MPI_OPTS+=" -x RANNC_MIN_PIPELINE_BS=${MIN_PIPELINE_BS}"

RL_OPTS="${RL}"
RL_OPTS+=" $(uname -n)"
RL_OPTS+=" 20010"
RL_OPTS+=" ${PART} ${REPL} ${PIPE}"

CMD="time mpirun"
CMD+=" ${MPI_OPTS}"
CMD+=" ${RL_OPTS}"
CMD+=" $CODEDIR/run_pretraining_rannc.py"
CMD+=" --input_dir=$DATA_DIR_PHASE1"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --bert_model=bert-large-uncased"
CMD+=" --train_batch_size=$train_batch_size"
CMD+=" --max_seq_length=128"
CMD+=" --max_predictions_per_seq=20"
CMD+=" --max_steps=$train_steps"
CMD+=" --warmup_proportion=$warmup_proportion"
CMD+=" --num_steps_per_checkpoint=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate"
CMD+=" --seed=$seed"
CMD+=" $PREC_ARG"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION_FP16"
CMD+=" $INIT_CHECKPOINT"
CMD+=" --do_train"
CMD+=" --json-summary ${RESULTS_DIR}/dllogger.json"

echo ${CMD}

if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size \* $num_gpus)
  printf -v TAG "pyt_bert_pretraining_phase1_%s_gbs%d" "$precision" $GBS
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$RESULTS_DIR/$job_name.$TAG.$DATESTAMP.log
  printf "Logs written to %s\n" "$LOGFILE"
fi

set -x

if [ "$PHASE2_ONLY" != "0" ] ; then
    echo "Skipping phase 1"
else
    if [ -z "$LOGFILE" ] ; then
        $CMD
    else
       (
         $CMD
       ) |& tee $LOGFILE
    fi
    echo "finished pretraining"
fi

set +x


if [ "$PHASE1_ONLY" != "0" ] ; then
    echo "Skipping phase 2"
    exit 0
fi

#Start Phase2

DATASET=hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus # change this for other datasets
DATA_DIR_PHASE2=${23:-$BERT_PREP_WORKING_DIR/${DATASET}/}

PREC_ARG=""
if [ "$precision" = "fp16" ] ; then
   PREC_ARG="--fp16"
elif [ "$precision" = "fp32" ] ; then
   PREC_ARG=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

ACCUMULATE_GRADIENTS=""
if [ "$accumulate_gradients" == "true" ] ; then
   ACCUMULATE_GRADIENTS="--gradient_accumulation_steps=$gradient_accumulation_steps_phase2"
fi

ALL_REDUCE_POST_ACCUMULATION=""
if [ "$allreduce_post_accumulation" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION="--allreduce_post_accumulation"
fi

ALL_REDUCE_POST_ACCUMULATION_FP16=""
if [ "$allreduce_post_accumulation_fp16" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION_FP16="--allreduce_post_accumulation_fp16"
fi

RESUME_OPTS=" --phase2 --resume_from_checkpoint --phase1_end_step=$train_steps"
if [ "$PHASE2_ONLY" != "0" ] ; then
RESUME_OPTS=""
fi

echo $DATA_DIR_PHASE2
INPUT_DIR=$DATA_DIR_PHASE2
CMD="time mpirun"
CMD+=" ${MPI_OPTS}"
CMD+=" ${RL_OPTS}"
CMD+=" $CODEDIR/run_pretraining_rannc.py"
CMD+=" --input_dir=$DATA_DIR_PHASE2"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --bert_model=bert-large-uncased"
CMD+=" --train_batch_size=$train_batch_size_phase2"
CMD+=" --max_seq_length=512"
CMD+=" --max_predictions_per_seq=80"
CMD+=" --max_steps=$train_steps_phase2"
CMD+=" --warmup_proportion=$warmup_proportion_phase2"
CMD+=" --num_steps_per_checkpoint=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate_phase2"
CMD+=" --seed=$seed"
CMD+=" $PREC_ARG"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION_FP16"
CMD+=" --do_train ${RESUME_OPTS}"
CMD+=" --json-summary ${RESULTS_DIR}/dllogger.json "

if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size_phase2 \* $num_gpus)
  printf -v TAG "pyt_bert_pretraining_phase2_%s_gbs%d" "$precision" $GBS
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$RESULTS_DIR/$job_name.$TAG.$DATESTAMP.log
  printf "Logs written to %s\n" "$LOGFILE"
fi

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi

set +x

echo "finished phase2"
