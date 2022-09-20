task_name=$1
device=$2

if [ $task_name == "csl" ]; then
    prompt="{'text': 'text_a'}{'soft': '这句话中讨论的关键词'}{'mask'}{'soft':'包括'}{'text': 'text_b'}"
    max_length=320
elif [ $task_name == "eprstmt" ]; then
    prompt="{'text':'text_a'}{'soft':'这个句话表示我'}{'mask'}{'soft':'喜欢这个东西'}"
    max_length=128
elif [ $task_name == "csldcp" ]; then
    prompt="{'text':'text_a'}{'soft':'这属于'}{'mask'}{'soft':'的知识。'}"
    max_length=256
elif [ $task_name == "tnews" ]; then
    prompt="{'text':'text_a'}{'soft':'这篇文章出自'}{'mask'}{'mask'}{'soft':'栏目。'}"
    max_length=64
elif [ $task_name == "iflytek" ]; then
    prompt="{'text':'text_a'}{'soft':'这款应用属于'}{'mask'}{'mask'}{'soft':'类别.'}"
    max_length=320
elif [ $task_name == "ocnli" ]; then
    prompt="{'text':'text_a'}{'mask'}{'mask'}{'text':'text_b'}"
    max_length=64
elif [ $task_name == "bustm" ]; then
    prompt="{'text':'text_a'}{'sep'}{'text':'text_b'}{'sep'}{'soft':'这两句话看起来'}{'mask'}{'soft':'像一个意思。'}"
    max_length=40
elif [ $task_name == "chid" ]; then
    # prompt="{'soft':'已知候选词有'}{'text':'text_b'}{'sep'}{'text':'text_a'}{'soft':'问：这句话的空格处应该填第'}{'mask'}{'soft':'个词'}"
    prompt="{'mask'}{'soft':'通顺。'}{'text':'text_a'}"
    max_length=256
elif [ $task_name == "cluewsc" ]; then
    prompt="{'mask'}{'soft':'合理。'}{'text':'text_a'}"
    max_length=128
fi

CUDA_VISIBLE_DEVICES=$device python ../train_cls.py \
--output_dir ./checkpoints/ \
--prompt "$prompt" \
--max_seq_length $max_length \
--learning_rate 3e-5 \
--ppt_learning_rate 3e-4 \
--do_train \
--num_train_epochs 20 \
--logging_steps 10 \
--do_predict \
--do_eval \
--do_save True \
--do_test \
--eval_steps 100 \
--save_steps 100 \
--per_device_eval_batch_size 32 \
--per_device_train_batch_size 8 \
--model_name_or_path ernie-3.0-base-zh \
--split_id few_all \
--task_name $task_name \
--metric_for_best_model accuracy \
--load_best_model_at_end \
--disable_tqdm True \
--do_predict \
#--early_stop_patience 10
#--freeze_plm \
#--soft_encoder mlp \
#--save_strategy no
#--evaluation_strategy epoch \
#--save_strategy epoch \
