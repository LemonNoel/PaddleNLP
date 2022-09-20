task_name=$1
device=$2

batch_size=8

if [ $task_name == "csl" ]; then
    prompt="“{'text': 'text_a'}”{'soft':'上文中找'}{'mask'}{'soft': '出这些关键词：'}{'text':'text_b'}"
    max_length=320
elif [ $task_name == "eprstmt" ]; then
    prompt="{'text':'text_a'}{'hard':'这个句话表示我'}{'mask'}{'hard':'喜欢这个东西'}" # BEST: 0.9016
    max_length=128
elif [ $task_name == "csldcp" ]; then
    prompt="“{'text':'text_a'}”{'soft':'这属于'}{'mask'}{'soft':'的知识。'}"
    max_length=256
elif [ $task_name == "tnews" ]; then
    prompt="{'soft':'下边播报一则'}{'mask'}{'mask'}{'soft':'新闻：'}“{'text':'text_a'}”"
    max_length=64
elif [ $task_name == "iflytek" ]; then
    prompt="“{'text':'text_a'}”{'soft':'这款应用属于'}{'mask'}{'mask'}{'soft':'类别'}"
    max_length=320
elif [ $task_name == "ocnli" ]; then
    prompt="{'soft':'请用正确的连接词填空：'}“{'text':'text_a'}”{'mask'}{'mask'}“{'text':'text_b'}”"
    max_length=128
elif [ $task_name == "bustm" ]; then
    prompt="“{'text':'text_a'}”和“{'text':'text_b'}”{'sep'}{'soft':'这两句话看起来'}{'mask'}{'soft':'像一个意思。'}"
    max_length=64
elif [ $task_name == "chid" ]; then
    prompt="{'text':'text_a'}{'soft':'这句话中的成语使用'}{'mask'}{'mask'}{'soft':'。选项：正确/错误'}"
    # prompt="{'text':'text_a'}{'hard':'这句话中的成语使用'}{'mask'}{'mask'}。选项：正确/错误" # 0.2572
    # prompt="{'text':'text_a'}{'sep'}{'hard':'这句话'}{'mask'}{'hard':'通顺。'}" # 0.5504
    max_length=256
elif [ $task_name == "cluewsc" ]; then
    prompt="{'soft':'这句话中指代关系合理吗？'}{'text':'text_a'}{'mask'}"
    max_length=128
elif [ $task_name == "cmnli" ]; then
    # FT.a 
    # prompt="{'hard':'请用正确的连接词填空：'}{'text':'text_a'}（{'mask'}{'mask'}）{'text':'text_b'}"
    # FT.b
    prompt="“{'text':'text_a'}”和“{'text':'text_b'}”之间的逻辑关系是{'mask'}{'mask'}。"
    max_length=128
    batch_size=32
fi

CUDA_VISIBLE_DEVICES=$device python train_single.py \
--output_dir ./checkpoints_iflytek/ \
--prompt "$prompt" \
--max_seq_length $max_length \
--learning_rate 3e-6 \
--ppt_learning_rate 3e-4 \
--num_train_epochs 50 \
--logging_steps 10 \
--do_save True \
--eval_steps 100 \
--save_steps 100 \
--per_device_eval_batch_size 32 \
--per_device_train_batch_size $batch_size \
--model_name_or_path ernie-3.0-xbase-zh \
--split_id few_all \
--task_name $task_name \
--metric_for_best_model accuracy \
--disable_tqdm True \
--do_train \
--do_test \
--do_predict \
--do_eval \
--load_best_model_at_end \
#--early_stop_patience 10
#--soft_encoder mlp \
#--save_strategy no
#--evaluation_strategy epoch \
#--save_strategy epoch \
--freeze_plm \
