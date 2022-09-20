task_name=$1
device=$2

if [ $task_name == "csl" ]; then
    prompt="“{'text':'text_a'}”其中“{'text':'text_b'}”{'mask'}{'mask'}这句话的关键词。选项：不是/就是"
    max_length=320
elif [ $task_name == "eprstmt" ]; then
    prompt="文本：“{'text':'text_a'}”这段文本的情感倾向为{'mask'}{'mask'}。选项：正向/负向"
    max_length=128
elif [ $task_name == "csldcp" ]; then
    prompt="“{'text':'text_a'}”该文本类别为{'mask'}{'mask'}。选项：{'text':'text_b', 'shortenable':False}"
    max_length=1024
elif [ $task_name == "tnews" ]; then
    prompt="“{'text':'text_a'}”上述新闻选自{'mask'}{'mask'}专栏。选项：{'text':'text_b', 'shortenable':False}"
    max_length=512
elif [ $task_name == "iflytek" ]; then
    prompt="“{'text':'text_a'}”这个应用的类别是{'mask'}{'mask'}。选项：{'text':'text_b', 'shortenable':False}"
    max_length=1024
elif [ $task_name == "ocnli" ]; then
    prompt="“{'text':'text_a'}”和“{'text':'text_b'}”之间的逻辑关系是{'mask'}{'mask'}。选项：中立/蕴含/矛盾"
    max_length=128
elif [ $task_name == "bustm" ]; then
    prompt="“{'text':'text_a'}”和“{'text':'text_b'}”描述的是{'mask'}{'mask'}的事情。选项：相同/不同"
    max_length=64
elif [ $task_name == "chid" ]; then
    prompt="“{'text':'text_a'}”这句话中成语[{'text':'text_b'}]的理解正确吗？选项：正确/错误。答：{'mask'}{'mask'}。"
    max_length=256
elif [ $task_name == "cluewsc" ]; then
    prompt="“{'text':'text_a'}”这句话中代词{'text':'text_b'}的使用正确吗？选项：正确/错误。答：{'mask'}{'mask'}"
    max_length=128
elif [ $task_name == "cmnli" ]; then
    prompt="“{'text':'text_a'}”和“{'text':'text_b'}”之间的逻辑关系是{'mask'}{'mask'}。"
    max_length=128
fi

CUDA_VISIBLE_DEVICES=$device python train_single.py \
--pretrained "/ssd2/wanghuijuan03/data/zero-shot/checkpoints_80w/checkpoint-260000/model_state.pdparams" \
--output_dir ./checkpoints/ \
--prompt "$prompt" \
--max_seq_length $max_length \
--learning_rate 3e-6 \
--ppt_learning_rate 3e-5 \
--num_train_epochs 50 \
--logging_steps 10 \
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
--disable_tqdm True \
--do_train \
--do_predict \
--do_eval \
--load_best_model_at_end \
#--early_stop_patience 10
#--soft_encoder mlp \
#--save_strategy no
#--evaluation_strategy epoch \
#--save_strategy epoch \
--freeze_plm \
