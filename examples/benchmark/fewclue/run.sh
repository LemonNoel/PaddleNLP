task_name=$1
device=$2

if [ $task_name == "csl" ]; then
    prompt="{'text': 'text_a'}{'soft':'上文中找'}{'mask'}{'soft': '出这些关键词：'}{'text':'text_b'}"
    max_length=320
elif [ $task_name == "eprstmt" ]; then
    prompt="{'text':'text_a'}{'soft':'我感觉'}{'mask'}{'soft':'喜欢。'}"
    max_length=128
elif [ $task_name == "csldcp" ]; then
    prompt="{'soft':'阅读下边'}{'mask'}{'mask'}{'soft':'相关的材料'}{'text':'text_a'}"
    max_length=256
elif [ $task_name == "tnews" ]; then
    prompt="{'soft':'下边播报一则'}{'mask'}{'mask'}{'soft':'新闻：'}{'text':'text_a'}"
    max_length=64
elif [ $task_name == "iflytek" ]; then
    prompt="{'mask'}{'mask'}{'soft':'APP更新日志：'}{'text':'text_a'}"
    max_length=320
elif [ $task_name == "ocnli" ]; then
    prompt="{'soft':None, 'duplicate':10}{'text':'text_a'}{'mask'}{'mask'}{'text':'text_b'}"
    max_length=64
elif [ $task_name == "bustm" ]; then
    prompt="{'text':'text_a'}{'sep'}{'text':'text_b'}{'sep'}{'soft':'前两句话'}{'mask'}{'soft':'像'}"
    max_length=40
elif [ $task_name == "chid" ]; then
    prompt="{'text':'text_a'}{'sep'}{'soft':'这句话读起来'}{'mask'}{'soft':'通顺。'}"
    max_length=256
elif [ $task_name == "cluewsc" ]; then
    prompt="{'text':'text_a'}{'sep'}{'soft':'其中代词用'}{'mask'}{'soft':'了。'}"
    max_length=128
fi

CUDA_VISIBLE_DEVICES=$device python train_single.py \
--output_dir ./checkpoints/ \
--prompt "$prompt" \
--max_seq_length $max_length \
--learning_rate 3e-2 \
--ppt_learning_rate 3e-1 \
--do_train \
--num_train_epochs 50 \
--logging_steps 10 \
--do_predict \
--do_eval \
--do_save True \
--do_test \
--eval_steps 20 \
--save_steps 20 \
--per_device_eval_batch_size 8 \
--per_device_train_batch_size 8 \
--model_name_or_path ~/ernie-3.0-1.5b-zh \
--split_id few_all \
--task_name $task_name \
--metric_for_best_model accuracy \
--load_best_model_at_end \
--disable_tqdm True \
--do_predict \
--freeze_plm \
#--early_stop_patience 10
#--soft_encoder mlp \
#--save_strategy no
#--evaluation_strategy epoch \
#--save_strategy epoch \
