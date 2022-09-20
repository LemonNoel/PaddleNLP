device=$1
num_test=10

count=0

while (($count < $num_test))
do
    echo " "
    echo "## "$count
    echo " "
    CUDA_VISIBLE_DEVICES=$device python train.py \
    --dataset_dir "./data/" \
    --save_dir "./checkpoints" \
    --max_seq_length 128 \
    --model_name "ernie-3.0-base-zh" \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --epochs 100 \
    --logging_steps 5 \
    --early_stop
    count=$[count+1]
done
