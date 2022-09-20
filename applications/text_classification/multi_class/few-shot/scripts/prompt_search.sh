device=$1
num_test=10

for prompt in {"这条新闻写的是","这是一条{'mask'}{'mask'}新闻","我觉得这篇文章写的是"}
do
    echo "------------------"
    echo $prompt
    echo "------------------"
    count=0
    while (($count < $num_test))
    do
        echo " "
        echo "## "$count
        echo " "
        CUDA_VISIBLE_DEVICES=$device python train.py \
        --data_dir ./data/ \
        --output_dir ./checkpoints/ \
        --prompt $prompt \
        --model_name_or_path ernie-3.0-base-zh \
        --max_seq_length 128  \
        --learning_rate 3e-5 \
        --ppt_learning_rate 3e-4 \
        --do_train \
        --do_eval \
        --num_train_epochs 100 \
        --logging_steps 5 \
        --per_device_eval_batch_size 32 \
        --per_device_train_batch_size 8 \
        --do_predict \
        --metric_for_best_model accuracy \
        --load_best_model_at_end \
        --evaluation_strategy epoch \
        --save_strategy epoch
        count=$[count+1]
    done
done

