INPUT=./data

unset CUDA_VISIBLE_DEVICES

python -m paddle.distributed.launch --gpus "0,1,2,3" run_pretrain.py \
    --batch_size 4 \
    --input_dir $INPUT \
    --output_dir ./output_1m_test/ \
    --learning_rate 1.25e-5 \
    --max_seq_length 512 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-8 \
    --warmup_steps 10000 \
    --num_epochs 1 \
    --max_steps 1000000 \
    --save_steps 10000 \
    --logging_steps 1 \
    --seed 1000 
#    --use_amp
