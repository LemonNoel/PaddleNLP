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
    max_length=512
elif [ $task_name == "csldcp" ]; then
    max_length=384
elif [ $task_name == "tnews" ]; then
    max_length=128 #64
elif [ $task_name == "iflytek" ]; then
    max_length=320
elif [ $task_name == "ocnli" ]; then
    max_length=128
elif [ $task_name == "bustm" ]; then
    max_length=256
elif [ $task_name == "chid" ]; then
    max_length=384
elif [ $task_name == "cluewsc" ]; then
    max_length=384
elif [ $task_name == "cluewsc2020" ]; then
    max_length=384
elif [ $task_name == "cmnli" ]; then
    max_length=128
fi


lrs=(0)
augs=(similar substitute delete swap insert)
#augs=(0)

for lr in ${lrs[@]}
do
    for aug in ${augs[@]}
    do
        out_dir=./checkpoints/ckpt-24k-$device-$task_name
        echo " "
        CUDA_VISIBLE_DEVICES=$device python3 train_single.py \
        --output_dir $out_dir \
        --aug_type $aug \
        --max_seq_length $max_length \
        --task_name $task_name \
        --t_index $lr \
        --t_type "auto" \
        --v_type "multi" \
        --learning_rate 3e-6 \
        --ppt_learning_rate 3e-5 \
        --max_steps 2000 \
        --logging_steps 10 \
        --load_best_model_at_end \
        --do_train \
        --do_eval \
        --do_save True \
        --do_test \
        --do_predict \
        --do_label True \
        --disable_tqdm True \
        --config_path "template1" \
        --lr_scheduler_type 'constant' \
        --eval_steps 50 \
        --save_steps 50 \
        --warmup_ratio 0.01 \
        --save_total_limit 1 \
        --per_device_eval_batch_size 16 \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --model_name_or_path ernie-1.0-large-zh-cw \
        --split_id few_all \
        --task_name $task_name \
        --metric_for_best_model accuracy \
        --seed 21 \
        --ckpt_model ckpt_cmnli_24k.pdparams
        #--ckpt_model ckpt_36k.pdparams # 无监督阅读理解
        
        #--ckpt_model ckpt_36k_wsc20_4h.pdparams # 无监督阅读理解 + 有监督指代
        #--ckpt_plm "/ssd2/wanghuijuan03/prompt/PaddleNLP/model_zoo/ernie-1.0/checkpoints-mix/model_36000/model_state.pdparams" # 无监督阅读理解
        --ckpt_model ckpt_36k_wsc20_21h.pdparams # 无监督阅读理解 + 有监督CLS FT
        #--ckpt_model ckpt-36k-wsc20-9h.pdparams

        #--ckpt_model ckpt_36k_wsc20_4h_cmnli_19k.pdparams # 无监督阅读理解 + 有监督指代 + 句间推理
        #--ckpt_model checkpoints/ckpt-36k-wsc-cluewsc/checkpoint-500/model_state.pdparams
        #--ckpt_model cmnli_36k_ckpt_22k.pdparams # 无监督阅读理解 + 句间推理
        #--ckpt_model cmnli_36k_ckpt_357h.pdparams # 无监督阅读理解 + 句间推理
        #--ckpt_model strategy_supervised/model_4600_89.92.pdparams 
        #--ckpt_model "results/e1cw/cmnli/checkpoint-24000/model_state.pdparams" 
        echo " "
        #rm -rf $out_dir/checkpoint-*
    done
done

#--ckpt_model None \
        --fake_file $fake \
        --use_rdrop True \
        --alpha_rdrop 1.0 \
        --dropout 0.3 \
        --freeze_plm \
        --soft_encoder mlp \
        --evaluation_strategy epoch \
        --save_strategy epoch 
       --soft_encoder lstm \
       --gradient_accumulation_steps 2 \

