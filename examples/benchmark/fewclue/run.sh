task_name=$1
device=$2
is_train=$3

batch_size=8

if [ $task_name == "csl" ]; then
    prompt="{'text': 'text_a'}{'hard':'上文中'}{'mask'}{'hard': '这些关键词：'}{'text':'text_b'}"
    max_length=320
elif [ $task_name == "eprstmt" ]; then
    prompt="{'text':'text_a'}{'hard':'这个句话表示我'}{'mask'}{'hard':'喜欢这个东西'}"
    max_length=128
elif [ $task_name == "csldcp" ]; then
    prompt="{'hard':'阅读下边有关'}{'mask'}{'mask'}{'hard':'的材料'}{'text':'text_a'}"
    max_length=256
elif [ $task_name == "tnews" ]; then
    prompt="{'hard':'下边播报一则'}{'mask'}{'mask'}{'hard':'新闻：'}{'text':'text_a'}"
    max_length=64
elif [ $task_name == "iflytek" ]; then
    prompt="{'text':'text_a'}{'hard':'这款应用是'}{'mask'}{'mask'}{'hard':'类型的。'}"
    max_length=320
elif [ $task_name == "ocnli" ]; then
    prompt="“{'text':'text_a'}”和“{'text':'text_b'}”之间的逻辑关系是{'mask'}{'mask'}。"
    max_length=128
elif [ $task_name == "bustm" ]; then
    prompt="“{'text':'text_a'}”和“{'text':'text_b'}”之间的逻辑关系是{'mask'}{'mask'}。"
    max_length=64
elif [ $task_name == "chid" ]; then
    # 0.3  prompt="{'text':'text_a'}{'sep'}{'hard':'这句话中的成语使用'}{'mask'}{'mask'}"
    # 0.18 prompt="{'text':'text_a'}“选项：”{'text':'text_b', 'shortenable':False}。正确答案是{'mask'}"
    # 0.39 prompt="{'text':'text_a'}{'sep'}{'hard':'这句话中的成语使用'}{'mask'}{'mask'}。"
    prompt="{'text':'text_a'}{'text':'text_b', 'shortenable':False}{'mask'}{'mask'}。"
    max_length=256
elif [ $task_name == "cluewsc" ]; then
    # 0.5  prompt="{'text':'text_a'}{'hard':'其中代词使用'}{'mask'}{'mask'}"
    # 0.48 prompt="{'text':'text_a'}{'sep'}{'hard':'其中代词用'}{'mask'}{'hard':'了。'}"
    prompt="{'text':'text_a'}{'mask'}{'mask'}地指代了{'text':'text_b'}。"
    max_length=128
elif [ $task_name == "cmnli" ]; then
    prompt="“{'text':'text_a'}”和“{'text':'text_b'}”之间的逻辑关系是{'mask'}{'mask'}。"
    max_length=128
    batch_size=32
fi

CUDA_VISIBLE_DEVICES=$device python train_single.py \
--output_dir ./checkpoints/ \
--prompt "$prompt" \
--max_seq_length $max_length \
--per_device_eval_batch_size 32 \
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
--learning_rate 3e-5 \
--ppt_learning_rate 3e-4 \
--load_best_model_at_end $is_train \
--do_train $is_train \
--do_predict $is_train \
--do_eval $is_train \
--do_save $is_train \
#--early_stop_patience 10
#--soft_encoder mlp \
#--save_strategy no
#--evaluation_strategy epoch \
#--save_strategy epoch \
--freeze_plm \
