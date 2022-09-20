DEVICE=$1

for prom in "{'soft':'当有人问“'}{'text':'text_a'}{'soft':'”时，他想知道的是'}{'mask'}。" #"{'soft':'已知'}{'text':'text_a'}{'soft':'问的是'}{'mask'}。"
do
    for plm in '/ssd2/wanghuijuan03/tmp/ernie-3.0-large' 'ernie-3.0-xbase-zh' 'ernie-3.0-base-zh' 'ernie-3.0-medium-zh' 'ernie-3.0-micro-zh' 'ernie-3.0-nano-zh'
    do
        echo "\n\n"
        CUDA_VISIBLE_DEVICES=$DEVICE python train.py \
        --output_dir ./ckpt/ \
        --prompt "$prom" \
        --max_seq_length 128  \
        --learning_rate 3e-5 \
        --do_eval \
        --ppt_learning_rate 3e-4 \
        --data_dir ../data/ \
        --do_train \
        --logging_steps 10 \
        --eval_steps 100 \
        --max_steps 1000 \
        --per_device_eval_batch_size 32  \
        --train_sample_per_label 16 \
        --save_steps 100000 \
        --model_name_or_path $plm \
        --freeze_dropout
        #--disable_tqdm True
        #--model_name_or_path /ssd2/wanghuijuan03/.paddlenlp/models/ernie-3.0-large
        #--freeze_dropout \
    done
done
