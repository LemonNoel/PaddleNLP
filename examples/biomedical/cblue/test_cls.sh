DATA=(CHIP-STS CHIP-CTC KUAKE-QQR KUAKE-QTR KUAKE-QIC)

for NAME in ${DATA[*]}
do
    BATCH_SIZE=32
    LENGTH=64
    LR=6e-5
    EPOCH=4
    MAX_STEPS=-1
    VALIDS=100
    if [[ "${NAME}" == "KUAKE-QIC" ]]; then
        LENGTH=128
    elif [[ "${NAME}" == "KUAKE-QTR" ]]; then
        MAX_STEPS=3010
    elif [[ "${NAME}" == "KUAKE-QQR" ]]; then
        EPOCH=2
        MAX_STEPS=1810
    elif [[ "${NAME}" == "CHIP-STS" ]]; then
        BATCH_SIZE=16
        LR=3e-5 #6e-5 #1e-4
        LENGTH=96
        #MAX_STEPS=4000
    elif [[ "${NAME}" == "CHIP-CTC" ]]; then
        LR=6e-5 #3e-5
        LENGTH=160
        MAX_STEPS=2810
    elif [[ "${NAME}" == "CHIP-CDN-2C" ]]; then
        LR=3e-5
        BATCH_SIZE=256
        LENGTH=32
        EPOCH=16
        VALIDS=1000
        MAX_STEPS=9910
    fi

    #python -m paddle.distributed.launch --gpus "0,1,2,3" train_classification.py \
    CUDA_VISIBLE_DEVICES=4 python train_classification.py \
    --dataset ${NAME} \
    --batch_size ${BATCH_SIZE} \
    --max_seq_length ${LENGTH} \
    --learning_rate ${LR} \
    --epochs ${EPOCH} \
    --init_from_ckpt /ssd2/wanghuijuan03/githubs/ernie-health/PaddleNLP/examples/biomedical/ernie_health/output_1m/model_400000.pdparams/model_state.pdparams \
    --seed 1000 \
    --valid_steps ${VALIDS} \
    --save_steps 200000 \
    --max_steps ${MAX_STEPS} \
    --save_dir ./output/${NAME}
done 

#FLAGS_check_nan_inf=1
python -m paddle.distributed.launch --gpus "4,5,6,7" train_classification.py --dataset CHIP-CDN-2C  --batch_size 256  --max_seq_length 32  --learning_rate 6e-5 --epochs 16 --init_from_ckpt /ssd2/wanghuijuan03/githubs/ernie-health/PaddleNLP/examples/biomedical/ernie_health/output_1m/model_400000.pdparams/model_state.pdparams 
