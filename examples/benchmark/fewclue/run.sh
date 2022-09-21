task_name=$1
device=$2

if [ $task_name == "csl" ]; then
    prompt="“{'text':'text_a'}”和“{'text':'text_b'}”的观点{'mask'}{'mask'}选项：相同/不同。" 
    max_length=360
elif [ $task_name == "eprstmt" ]; then
    prompt="“{'text':'text_a'}”这个句子说明他很{'mask'}{'mask'}。选项：生气/高兴"
    max_length=128
elif [ $task_name == "csldcp" ]; then
    prompt="“{'text':'text_a'}”这段话选自{'mask'}{'mask'}的课本。选项：{'text':'text_b'}。"
    max_length=512
elif [ $task_name == "tnews" ]; then
    prompt="从新闻标题可以推断出主题为{'mask'}{'mask'}新闻标题：“{'text':'text_a'}”选项：{'text':'text_b'}。"
    max_length=256
elif [ $task_name == "iflytek" ]; then
    prompt="“{'text':'text_a'}”说的是关于{'mask'}{'mask'}的内容。选项：{'text':'text_b'}。"
    max_length=512
elif [ $task_name == "ocnli" ]; then
    prompt="“{'text':'text_a'}”{'mask'}{'mask'}，“{'text':'text_b'}”选项：因而/另外/然而" # 0.7071
    max_length=128
elif [ $task_name == "bustm" ]; then
    prompt="“{'text':'text_a'}”和“{'text':'text_b'}”两个问题语义{'mask'}{'mask'}区别。选项：略有/毫无"
    max_length=128
elif [ $task_name == "chid" ]; then
    prompt="“{'text':'text_a'}”成语{'text':'text_b'}放在句子括号的位置很{'mask'}{'mask'}。空格处可填：恰当/奇怪"
    max_length=256
elif [ $task_name == "cluewsc" ]; then
    prompt="“{'text':'text_a'}”有人认为{'text':'text_b'}，说明他{'mask'}{'mask'}理解了这个句子。选项：已经/没有"
    max_length=256
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
--num_train_epochs 30 \
--logging_steps 10 \
--do_save True \
--do_test \
--eval_steps 200 \
--save_steps 200 \
--per_device_eval_batch_size 4 \
--per_device_train_batch_size 4 \
--model_name_or_path ernie-1.0-large-zh-cw \
--split_id few_all \
--task_name $task_name \
--metric_for_best_model accuracy \
--disable_tqdm True \
--do_train \
--do_eval \
--load_best_model_at_end \
--evaluation_strategy epoch \
--save_strategy epoch \
--save_total_limit 1 \
--pretrained "/ssd2/wanghuijuan03/data/zero-shot/checkpoints_0915/checkpoint-90000/model_state.pdparams" \
--gradient_accumulation_steps 4

#--do_predict \
#--early_stop_patience 10
#--soft_encoder mlp \
#--save_strategy no
#--pretrained "/ssd2/wanghuijuan03/data/zero-shot/checkpoints/checkpoint-5000/model_state.pdparams" \
#--pretrained "./checkpoints_cmnli/checkpoint-6000/model_state.pdparams" \
#--freeze_plm \
rm -rf ./checkpoints_$task_name/
