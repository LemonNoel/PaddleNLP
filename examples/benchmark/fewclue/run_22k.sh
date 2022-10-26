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
    max_length=512
elif [ $task_name == "eprstmt" ]; then
    max_length=128
elif [ $task_name == "csldcp" ]; then
    max_length=256
elif [ $task_name == "tnews" ]; then
    max_length=128 #64
elif [ $task_name == "iflytek" ]; then
    max_length=320
elif [ $task_name == "ocnli" ]; then
    max_length=128
elif [ $task_name == "bustm" ]; then
    max_length=128
elif [ $task_name == "chid" ]; then
    max_length=384
elif [ $task_name == "cluewsc" ]; then
    max_length=384
elif [ $task_name == "cmnli" ]; then
    max_length=128
fi


lrs=(3e-6)
accsteps=(2)


for lr in ${lrs[@]}
do
    for step in ${accsteps[@]}
    do
        out_dir=./checkpoints/ckpt-cu-$task_name
        echo " "
        CUDA_VISIBLE_DEVICES=$device python train_single.py \
        --output_dir $out_dir \
        --max_seq_length $max_length \
        --task_name $task_name \
        --t_index 0 \
        --t_type "auto" \
        --v_type "cls" \
        --learning_rate $lr \
        --ppt_learning_rate $lr \
        --max_steps 3000 \
        --logging_steps 16 \
        --do_predict \
        --do_train \
        --do_eval \
        --do_test \
        --do_label True \
        --do_save True \
        --aug_type substitute \
        --load_best_model_at_end \
        --disable_tqdm True \
        --eval_steps 100 \
        --save_steps 100 \
        --warmup_ratio 0.01 \
        --save_total_limit 1 \
        --per_device_eval_batch_size 16 \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps $step \
        --model_name_or_path ernie-1.0-large-zh-cw \
        --split_id few_all \
        --task_name $task_name \
        --metric_for_best_model accuracy \
        --seed 42 \
        --ckpt_plm "/ssd2/wanghuijuan03/prompt/PaddleNLP/model_zoo/ernie-1.0/checkpoints-mix/model_36000/model_state.pdparams"
        #--ckpt_model cmnli_36k_ckpt_22k.pdparams 
        #--ckpt_model strategy_supervised/model_4600_89.92.pdparams 
        #--ckpt_model "results/e1cw/cmnli/checkpoint-24000/model_state.pdparams" 
        echo " "
        #rm -rf $out_dir/checkpoint-*
    done
done

#--ckpt_model None \
        --freeze_plm \
        --soft_encoder mlp \
        --evaluation_strategy epoch \
        --save_strategy epoch 
       --fake_file $fake \
       --use_rdrop True \
       --alpha_rdrop 1.0 \
       --soft_encoder lstm \
       --gradient_accumulation_steps 2 \
       --dropout 0.3 \

