task_name=$1
device=$2

if [ $task_name == "csl" ]; then
    prompt="“{'text':'text_a'}”本文的内容{'mask'}{'mask'}“{'text':'text_b'}”"
    # prompt="{'text':'text_a'}其中{'text':'text_b'}{'mask'}{'mask'}是关键词" # 0.6142
    max_length=320
elif [ $task_name == "eprstmt" ]; then
    prompt="“{'text':'text_a'}”这条评论的情感倾向是{'mask'}{'mask'}的。"
    max_length=128
elif [ $task_name == "csldcp" ]; then
    prompt="“{'text':'text_a'}”这篇文献的类别是{'mask'}{'mask'}。"
    max_length=256
elif [ $task_name == "tnews" ]; then
    prompt="“{'text':'text_a'}”上述新闻选自{'mask'}{'mask'}专栏。"
    max_length=64
elif [ $task_name == "iflytek" ]; then
    prompt="“{'text':'text_a'}”因此，应用类别是{'mask'}{'mask'}。"
    max_length=320
elif [ $task_name == "ocnli" ]; then
    prompt="“{'text':'text_a'}”和“{'text':'text_b'}”之间的逻辑关系是{'mask'}{'mask'}。" # 0.4167
    # prompt="“{'text':'text_a'}”{'mask'}{'mask'}，“{'text':'text_b'}”" # 0.7071
    max_length=128
elif [ $task_name == "bustm" ]; then
    # prompt="“{'text':'text_a'}”和“{'text':'text_b'}”之间的逻辑关系是{'mask'}{'mask'}。" # 0.7556
    prompt="“{'text':'text_a'}”和“{'text':'text_b'}”描述的是{'mask'}{'mask'}的事情。" # 0.7624
    max_length=64
elif [ $task_name == "chid" ]; then
    prompt="“{'text':'text_a'}”这句话中成语[{'text':'text_b'}]的理解正确吗？{'mask'}{'mask'}。"
    #prompt="{'text':'text_a'}{'text':'text_b'}用在这里对吗？{'mask'}{'mask'}。"
    max_length=256
elif [ $task_name == "cluewsc" ]; then
    prompt="“{'text':'text_a'}”{'text':'text_b'}这里代词使用正确吗？{'mask'}{'mask'}"
    max_length=128
elif [ $task_name == "cmnli" ]; then
    prompt="“{'text':'text_a'}”和“{'text':'text_b'}”之间的逻辑关系是{'mask'}{'mask'}。"
    max_length=128
fi

CUDA_VISIBLE_DEVICES=$device python train_single.py \
--output_dir ./checkpoints_$task_name/ \
--prompt "$prompt" \
--max_seq_length $max_length \
--learning_rate 3e-6 \
--ppt_learning_rate 3e-5 \
--num_train_epochs 20 \
--logging_steps 10 \
--do_train \
--do_eval \
--do_test \
--disable_tqdm True \
--do_save True \
--eval_steps 200 \
--save_steps 200 \
--per_device_eval_batch_size 16 \
--per_device_train_batch_size 16 \
--model_name_or_path ernie-1.0-large-zh-cw \
--split_id few_all \
--task_name $task_name \
--pretrained "../checkpoints_cmnli/checkpoint-22000/model_state.pdparams" \
--metric_for_best_model accuracy \
--do_predict \
--load_best_model_at_end \
--evaluation_strategy epoch \
--save_strategy epoch 
#--early_stop_patience 10
#--soft_encoder mlp \
#--save_strategy no
--pretrained "/ssd2/wanghuijuan03/data/zero-shot/checkpoints_09191451/checkpoint-43000/model_state.pdparams" \
#--pretrained "/ssd2/wanghuijuan03/data/zero-shot/0919_model_state.pdparams" \
#--pretrained "/ssd2/wanghuijuan03/data/zero-shot/checkpoints/checkpoint-5000/model_state.pdparams" \
#--freeze_plm \

rm -rf ./checkpoints_$task_name/
