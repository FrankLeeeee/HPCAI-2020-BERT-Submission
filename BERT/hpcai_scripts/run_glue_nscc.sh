#!/usr/bin/env bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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
export BERT_DIR=/home/users/ntu/c170166/projects/HPCAI-2020/model/wwm_uncased_L-24_H-1024_A-16
export GLUE_DIR=/home/users/ntu/c170166/projects/HPCAI-2020/dataset/glue_data

echo "Container nvidia build = " $NVIDIA_BUILD_ID

task_name=${1:-"MNLI"}
batch_size=${2:-"24"}
learning_rate=${3:-"5e-5"}
precision=${4:-"fp16"}
use_xla=${5:-"true"}
num_gpu=${6:-"1"}
seq_length=${7:-"128"}
doc_stride=${8:-"64"}
epochs=${9:-"3.0"}
ws=${10:-"0.1"}
init_checkpoint=${11:-"$BERT_DIR/bert_model.ckpt"}

echo "GLUE directory set as " $GLUE_DIR " BERT directory set as " $BERT_DIR

use_fp16=""
if [ "$precision" = "fp16" ] ; then
    echo "fp16 activated!"
    use_fp16="--amp"
else
    echo "fp32/tf32 activated!"
    use_fp16="--noamp"
fi

if [ "$use_xla" = "true" ] ; then
    use_xla_tag="--use_xla"
    echo "XLA activated"
else
    use_xla_tag="--nouse_xla"
fi


#if [ $num_gpu -gt 1 ] ; then
#    mpi_command="mpirun -np $num_gpu -H localhost:$num_gpu \
#    --allow-run-as-root -bind-to none -map-by slot \
#    -x NCCL_DEBUG=INFO \
#    -x LD_LIBRARY_PATH \
#    -x PATH -mca pml ob1 -mca btl ^openib"
#else
#    mpi_command=""
#fi

#export GBS=$(expr $batch_size \* $num_gpu)
printf -v TAG "tf_bert_finetuning_glue_%s_%s_%s_gbs%d" "$task_name" "$bert_model" "$precision" $GBS
#DATESTAMP=`date +'%y%m%d%H%M%S'`
#Edit to save logs & checkpoints in a different directory
#RESULTS_DIR=/home/users/ntu/c170166/projects/HPCAI-2020/results
#LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
mkdir -m 777 -p $RESULTS_DIR
printf "Saving checkpoints to %s\n" "$RESULTS_DIR"
printf "Logs written to %s\n" "$LOGFILE"

#Check if all necessary files are available before training
#for DIR_or_file in $GLUE_DIR/${task_name} $RESULTS_DIR $BERT_DIR/vocab.txt $BERT_DIR/bert_config.json; do
#  echo $DIR_or_file
#  if [ ! -d "$DIR_or_file" ] && [ ! -f "$DIR_or_file" ]; then
#     echo "Error! $DIR_or_file directory missing. Please mount correctly"
#     exit -1
#  fi
#done

#$mpi_command 

python /home/users/ntu/c170166/projects/HPCAI-2020/DeepLearningExamples/TensorFlow/LanguageModeling/BERT/run_classifier.py \
  --task_name=$task_name \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/$task_name \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$init_checkpoint \
  --max_seq_length=$seq_length \
  --doc_stride=$doc_stride \
  --train_batch_size=$batch_size \
  --learning_rate=$learning_rate \
  --num_train_epochs=$epochs \
  --output_dir=$RESULTS_DIR \
  --horovod \
  "$use_fp16" \
  --dllog_path $RESULTS_DIR/bert_dllog.json \
  $use_xla_tag \
  --warmup_proportion=$ws |& tee $LOGFILE
