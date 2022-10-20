# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

task_name=$1
device=$2
fake=$3

if [ $task_name == "csl" ]; then
    prompt="“{'text':'text_a'}”本文的内容{'mask'}{'mask'}“{'text':'text_b'}”"
    # prompt="{'text':'text_a'}其中{'text':'text_b'}{'mask'}{'mask'}是关键词" # 0.6142
    max_length=512
elif [ $task_name == "eprstmt" ]; then
    prompt="“{'text':'text_a'}”这条评论的情感倾向是{'mask'}{'mask'}的。"
    max_length=128
elif [ $task_name == "csldcp" ]; then
    prompt="“{'text':'text_a'}”这篇文献的类别是{'mask'}{'mask'}。"
    max_length=256
elif [ $task_name == "tnews" ]; then
    prompt="“{'text':'text_a'}”上述新闻选自{'mask'}{'mask'}专栏。"
    max_length=128 #64
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
    max_length=128
elif [ $task_name == "chid" ]; then
    prompt="“{'text':'text_a'}”这句话中成语[{'text':'text_b'}]的理解正确吗？{'mask'}{'mask'}。"
    #prompt="{'text':'text_a'}{'text':'text_b'}用在这里对吗？{'mask'}{'mask'}。"
    max_length=384
elif [ $task_name == "cluewsc" ]; then
    prompt="“{'text':'text_a'}”{'text':'text_b'}这里代词使用正确吗？{'mask'}{'mask'}"
    max_length=384
elif [ $task_name == "cmnli" ]; then
    prompt="“{'text':'text_a'}”和“{'text':'text_b'}”之间的逻辑关系是{'mask'}{'mask'}。"
    max_length=128
fi


lrs=(3e-6)
seeds=(42) # 21 42 87) #42 1024 

#lrs=(3e-5)
#pptlrs=(3e-6 3e-5 3e-3 3e-4)

for lr in ${lrs[@]}
do
    for seed in ${seeds[@]}
    do
        out_dir=./checkpoints/ckpt-1w-$task_name
        echo " "
        CUDA_VISIBLE_DEVICES=$device python train_single.py \
        --output_dir $out_dir \
        --max_seq_length $max_length \
        --task_name $task_name \
        --t_index 0 \
        --t_type "auto" \
        --v_type "multi" \
        --learning_rate $lr \
        --ppt_learning_rate $lr \
        --max_steps 3000 \
        --logging_steps 10 \
        --do_train \
        --do_eval \
        --do_test \
        --do_predict \
        --do_label True \
        --do_save True \
        --disable_tqdm True \
        --eval_steps 100 \
        --save_steps 100 \
        --warmup_ratio 0.01 \
        --per_device_eval_batch_size 8 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 2 \
        --model_name_or_path ernie-1.0-large-zh-cw \
        --split_id few_all \
        --task_name $task_name \
        --metric_for_best_model accuracy \
        --load_best_model_at_end \
        --seed $seed \
        --ckpt_plm "/ssd2/wanghuijuan03/prompt/PaddleNLP/model_zoo/ernie-1.0/checkpoints/model_10000/model_state.pdparams"
        #--ckpt_model "results/e1cw/cmnli/checkpoint-24000/model_state.pdparams" 
        echo " "
        rm -rf $out_dir/checkpoint-*
    done
done
#--evaluation_strategy epoch \
#--save_strategy epoch 
#--ckpt_model None \
        --fake_file $fake \
        --use_rdrop True \
        --alpha_rdrop 1.0 \
        --soft_encoder lstm \
        --aug_type substitute \
        --gradient_accumulation_steps 2 \
        --dropout 0.3 \
#--freeze_plm \
#--soft_encoder mlp \

