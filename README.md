# HPCAI-2020-BERT-Submission

## Introduction
This codebase is to reproduce the results in the report submitted to APAC HPCAI 2020. We optimize the distributed performance by using gradient checkpointing. The baseline is run on 1 V100 GPU on NSCC dgx-dev and the optimized code is run on 2 nodes with 4 GPUs on each node. The optimized code can achieve more than 8 times throughput compared to baseline experiment.


## Gradient checkpointing
Gradient checkpointing is a method to save GPU memory and boost the batch size in expense of some computation time. The paper is published here https://arxiv.org/abs/1604.06174. This method does not modify the structure of the model, it can be integrated into the code seamlessly. The original code is in `modeling_v0.py` and the code with gradient checkpointing is in `modeling_v2.py`.

## Checkpoint saving
This code will save checkpoints at 20 min automatically as configured in the TensorFlow estimator:
```python
run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir if master_process else None,
        session_config=config,
        save_checkpoints_secs=60*20 if master_process else None,
        log_step_count_steps=FLAGS.display_loss_steps,
        keep_checkpoint_max=1)
```

This ensures the experiments are compared consistently.

## Set up
```shell
git clone https://github.com/FrankLeeeee/HPCAI-2020-BERT-Submission.git
```


## Run Baseline

To run the baseline experiment, you need to follow the following steps:

1. get interactive job
```
qsub -I -q dgx -l walltime=1:00:00,select=1:ngpus=1:ncpus=5:mpiprocs=1 -P $ProjectID 
```

2. change the variables `BERT_DIR`, `GLUE_DIR` and `RESULTS_DIR` in `hpcai_scripts/run_glue.sh`

3. The baseline is using the original model implementation provided by Nvidia. Thus, you need to edit the `run_classifier.py` like below:
```
# change line 32
# import modeling_v2 as modeling 
# to the line below 
import modeling_v0 as modeling
``` 

4. Also, make sure the config is consistent with the baseline config in the `run_glue.sh`
```shell
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
```

5. run the script
```shell
cd ./HPCAI-2020-BERT-Submission/BERT
singularity exec /home/projects/ai/singularity/nvcr.io/nvidia/tensorflow:20.02-tf1-py3.sif ./hpcai_scripts/run_glue.sh
```
6. change the variables in `eval.sh` and run it
```shell
bash ./hpcai_scripts/eval.sh
```

## Run Optimized Code

To run the optimized code on 2 nodes with 4 GPUs on each node, you need to follow the following steps:
1. setup multinode communcation ssh 
```shell
to be completed by yuhao 
```

2. Make sure the `GLUE_SCRIPT` points to the `BERT/hpcai_scripts/run_glue_nscc.sh` in the `job_tensorflow_gloo.sh`
```
GLUE_SCRIPT=$PATH_TO_SUBMISSION/BERT/hpcai_scripts/run_glue_nscc.sh
```

3. The optimized code is using the model with gradient checkpointing. Thus, you need to edit the `run_classifier.py` like below:
```
# change line 32
# import modeling_v0 as modeling 
# to the line below
import modeling_v2 as modeling
``` 

4. Also, make sure the config is consistent with the baseline config in the `run_glue.sh`
```shell
task_name=${1:-"MNLI"}
batch_size=${2:-"48"}
learning_rate=${3:-"5e-5"}
precision=${4:-"fp16"}
use_xla=${5:-"false"}
num_gpu=${6:-"1"} 
seq_length=${7:-"128"}
doc_stride=${8:-"64"}
epochs=${9:-"3.0"}
ws=${10:-"0.1"}

# NOTE
# variable num_gpu is not in effect in the optimized code
```

5. 
```shell
bash $PATH_TO_SSHCONT/sshcont/invocation
```