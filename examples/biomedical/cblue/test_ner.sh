NAME=CMeEE
BATCH_SIZE=32
LEARNING_RATE=(6e-5) #1e-4)
LENGTH=128


for LR in ${LEARNING_RATE[*]}
do
    CUDA_VISIBLE_DEVICES=3 python train_ner.py \
    --batch_size ${BATCH_SIZE} \
    --max_seq_length ${LENGTH} \
    --learning_rate ${LR} \
    --epochs 2 \
    --seed 1000 \
    --logging_steps 10 \
    --valid_steps 100 \
    --save_steps 910 \
    --max_steps 910 \
    --save_dir ./CMeEE_result/ \
    --init_from_ckpt /ssd2/wanghuijuan03/githubs/ernie-health/PaddleNLP/examples/biomedical/ernie_health/output_1m/model_400000.pdparams/model_state.pdparams 
    #--init_from_ckpt output/init_param/model_state.pdparams #~/.paddlenlp/models/chinese-electra-base/chinese-electra-base.pdparams 

done

