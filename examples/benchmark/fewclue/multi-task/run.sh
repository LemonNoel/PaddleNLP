task_name=$1
device=$2
is_train=True

batch_size=8

if [ $task_name == "csl" ]; then
    max_length=320
elif [ $task_name == "eprstmt" ]; then
    max_length=128
elif [ $task_name == "csldcp" ]; then
    max_length=256
elif [ $task_name == "tnews" ]; then
    max_length=64
elif [ $task_name == "iflytek" ]; then
    max_length=320
elif [ $task_name == "ocnli" ]; then
    max_length=128
elif [ $task_name == "bustm" ]; then
    max_length=64
elif [ $task_name == "chid" ]; then
    max_length=256
elif [ $task_name == "cluewsc" ]; then
    max_length=128
elif [ $task_name == "cmnli" ]; then
    max_length=128
    batch_size=32
fi

CUDA_VISIBLE_DEVICES=$device python train_single.py \
--output_dir ./checkpoints/ \
--prompt "$prompt" \
--max_seq_length $max_length \
--per_device_eval_batch_size $batch_size \
--per_device_train_batch_size $batch_size \
--model_name_or_path ernie-3.0-base-zh \
--split_id few_all \
--task_name $task_name \
--metric_for_best_model accuracy \
--disable_tqdm True \
--do_test \
--eval_steps 100 \
--save_steps 100 \
--num_train_epochs 50 \
--logging_steps 10 \
--learning_rate 1e-5 \
--ppt_learning_rate 1e-4 \
--load_best_model_at_end $is_train \
--do_train $is_train \
--do_eval $is_train \
--do_save $is_train \
--do_predict $is_train \
#--early_stop_patience 10
#--soft_encoder mlp \
#--save_strategy no
#--evaluation_strategy epoch \
#--save_strategy epoch \
--freeze_plm \
