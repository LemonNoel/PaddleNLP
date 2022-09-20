task_name=$1
device=$2

if [ $task_name == "csl" ]; then
    prompt="{'text': 'text_a'}{'sep'}{'text': 'text_b'}{'hard': '是真实的关键词。'}"
    max_length=320
elif [ $task_name == "eprstmt" ]; then
    prompt="{'text':'text_a'}{'hard':'我很满意。'}"
    max_length=128
elif [ $task_name == "csldcp" ]; then
    prompt="{'text':'text_a'}{'hard':'这篇论文描述了什么知识？'}"
    max_length=256
elif [ $task_name == "tnews" ]; then
    prompt="{'text':'text_a'}{'hard':'这是一条什么新闻？'}"
    max_length=64
elif [ $task_name == "iflytek" ]; then
    prompt="{'text':'text_a'}{'hard':'这段文本的应用描述主题是什么？'}"
    max_length=320
elif [ $task_name == "ocnli" ]; then
    prompt="{'text':'text_a'}{'sep'}{'text':'text_b'}"
    max_length=64
elif [ $task_name == "bustm" ]; then
    prompt="{'text':'text_a'}{'sep'}{'text':'text_b'}"
    max_length=40
elif [ $task_name == "chid" ]; then
    # prompt="{'soft':'已知候选词有'}{'text':'text_b'}{'sep'}{'text':'text_a'}{'soft':'问：这句话的空格处应该填第'}{'mask'}{'soft':'个词'}"
    prompt="{'text':'text_a'}"
    max_length=256
elif [ $task_name == "cluewsc" ]; then
    prompt="{'text':'text_a'}"
    max_length=128
fi

CUDA_VISIBLE_DEVICES=$device python train_cls.py \
--output_dir ./checkpoints/ \
--prompt "$prompt" \
--max_seq_length $max_length \
--learning_rate 1e-5 \
--ppt_learning_rate 1e-4 \
--do_train \
--num_train_epochs 50 \
--logging_steps 10 \
--do_predict \
--do_eval \
--do_save True \
--do_test \
--eval_steps 50 \
--save_steps 50 \
--per_device_eval_batch_size 2 \
--per_device_train_batch_size 2 \
--model_name_or_path  ~/ernie-3.0-1.5b-zh \
--split_id few_all \
--task_name $task_name \
--metric_for_best_model accuracy \
--load_best_model_at_end \
--disable_tqdm True \
--do_predict \
--early_stop_patience 6 \
--freeze_plm \
#--soft_encoder mlp \
#--save_strategy no
#--evaluation_strategy epoch \
#--save_strategy epoch \
