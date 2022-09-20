task_name=$1
device=$2

if [ $task_name == "csl" ]; then
    # A7.PET-csl: prompt="{'mask'}{'mask'}用{'text':'text_b'}概括{'text':'text_a'}"
    prompt="{'text': 'text_a'}{'hard':'上文中找'}{'mask'}{'hard': '出这些关键词：'}{'text':'text_b'}"
    max_length=320
elif [ $task_name == "eprstmt" ]; then
    prompt="{'text':'text_a'}{'hard':'我感觉'}{'mask'}{'hard':'喜欢。'}"
    max_length=128
elif [ $task_name == "csldcp" ]; then
    prompt="{'hard':'阅读下边'}{'mask'}{'mask'}{'hard':'相关的材料'}{'text':'text_a'}"
    max_length=256
elif [ $task_name == "tnews" ]; then
    prompt="{'hard':'下边播报一则'}{'mask'}{'mask'}{'hard':'新闻：'}{'text':'text_a'}"
    max_length=64
elif [ $task_name == "iflytek" ]; then
    prompt="{'mask'}{'mask'}{'hard':'APP更新日志：'}{'text':'text_a'}"
    max_length=320
elif [ $task_name == "ocnli" ]; then
    prompt="{'hard':'请用正确的连接词填空：'}{'text':'text_a'}{'mask'}{'mask'}{'text':'text_b'}"
    max_length=64
elif [ $task_name == "bustm" ]; then
    prompt="{'text':'text_a'}{'sep'}{'text':'text_b'}{'sep'}{'hard':'前两句话'}{'mask'}{'hard':'像'}"
    max_length=40
elif [ $task_name == "chid" ]; then
    prompt="{'text':'text_a'}{'sep'}{'hard':'这句话'}{'mask'}{'hard':'通顺。'}"
    max_length=256
elif [ $task_name == "cluewsc" ]; then
    # A7-cluewsc: 
    prompt="{'text':'text_a'}{'sep'}{'hard':'其中代词用'}{'mask'}{'hard':'了。'}"
    # A7.PET: prompt="下面句子的指代关系正确吗？{'mask'}{'mask'}{'text':'text_a'}"
    max_length=128
fi


CUDA_VISIBLE_DEVICES=$device python train_single.py \
--output_dir ./checkpoints/ \
--prompt "$prompt" \
--max_seq_length $max_length \
--learning_rate 3e-5 \
--ppt_learning_rate 3e-4 \
--num_train_epochs 100 \
--logging_steps 10 \
--do_predict \
--do_save True \
--do_test \
--eval_steps 100 \
--save_steps 100 \
--per_device_eval_batch_size 32 \
--per_device_train_batch_size 8 \
--model_name_or_path hfl/roberta-wwm-ext-large \
--split_id few_all \
--task_name $task_name \
--metric_for_best_model accuracy \
--disable_tqdm True \
--do_predict \
#--early_stop_patience 10
#--freeze_plm \
#--soft_encoder mlp \
#--save_strategy no
#--evaluation_strategy epoch \
#--save_strategy epoch \
--do_train \
--do_eval \
--load_best_model_at_end \
