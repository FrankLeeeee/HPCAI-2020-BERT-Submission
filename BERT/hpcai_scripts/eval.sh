export MODEL_PATH=/home/shenggui/projects/02-HPCAI/model/wwm_uncased_L-24_H-1024_A-16
export DATA_PATH=/home/shenggui/projects/02-HPCAI/dataset/glue_data/MNLI
export RESULT_PATH=/home/shenggui/projects/02-HPCAI/results

TASK_NAME="MNLI"

python run_classifier.py \
  --do_train=false --do_eval=true --do_predict=false \
  --task_name=$TASK_NAME \
  --data_dir=$DATA_PATH \
  --vocab_file=$MODEL_PATH/vocab.txt \
  --bert_config_file=$MODEL_PATH/bert_config.json \
  --output_dir=$RESULT_PATH \
  --dllog_path=$RESULT_PATH/bert_dllog.json \
  --use_fp16 
