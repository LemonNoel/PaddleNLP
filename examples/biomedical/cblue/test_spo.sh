NAME=CMeIE
BATCH_SIZE=12
LR=6e-5
LENGTH=300
EPOCHS=(100)

unset CUDA_VISIBLE_DEVICES

for EPOCH in ${EPOCHS[*]}
do
    python -m paddle.distributed.launch --gpus "0,1,2,3" train_spo.py \
    --batch_size ${BATCH_SIZE} \
    --max_seq_length ${LENGTH} \
    --learning_rate ${LR} \
    --epochs ${EPOCH} \
    --save_steps 10000000 \
    --logging_steps 10 \
    --use_amp True \
    --valid_steps 1000 \
    --save_dir ./CMeIE_result/ \
    #--init_from_ckpt /ssd2/wanghuijuan03/PaddleNLP/examples/biomedical/cblue/checkpoint/model_20000/model_state.pdparams

done

