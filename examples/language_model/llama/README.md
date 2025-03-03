# LLaMA inplementation

**目录**

- [1. 微调](#1)
- [2. 模型预测](#2)
- [3. 动转静](#3)
- [4. 模型推理](#4)

## 协议

Llama 模型的权重的使用则需要遵循[License](../../../paddlenlp/transformers/llama/LICENSE)。

<a name="1"></a>

## 微调

```shell
python -u  -m paddle.distributed.fleet.launch \
    --gpus "0,1,2,3" finetune_generation.py \
    --model_name_or_path facebook/llama-7b \
    --do_train \
    --do_eval \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --tensor_parallel_degree 4 \
    --overwrite_output_dir \
    --output_dir ./checkpoints/ \
    --logging_steps 10 \
    --fp16 \
    --fp16_opt_level O2 \
    --gradient_accumulation_steps 32 \
    --recompute \
    --learning_rate 3e-5 \
    --lr_scheduler_type linear \
    --max_grad_norm 1.0 \
    --warmup_steps 20
```

### 单卡LoRA微调

```shell
python finetune_generation.py \
    --model_name_or_path facebook/llama-7b \
    --do_train \
    --do_eval \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --overwrite_output_dir \
    --output_dir ./checkpoints/ \
    --logging_steps 10 \
    --fp16 \
    --fp16_opt_level O2 \
    --gradient_accumulation_steps 4 \
    --recompute \
    --learning_rate 3e-4 \
    --lr_scheduler_type linear \
    --max_grad_norm 1.0 \
    --warmup_steps 20 \
    --lora True \
    --r 8
```

### 单卡Prefix微调

```shell
python finetune_generation.py \
    --model_name_or_path facebook/llama-7b \
    --do_train \
    --do_eval \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --overwrite_output_dir \
    --output_dir ./checkpoints/ \
    --logging_steps 10 \
    --fp16 \
    --fp16_opt_level O2 \
    --gradient_accumulation_steps 4 \
    --recompute \
    --learning_rate 3e-2 \
    --lr_scheduler_type linear \
    --max_grad_norm 1.0 \
    --warmup_steps 20 \
    --prefix_tuning True \
    --num_prefix_tokens 64
```

其中参数释义如下：

- `model_name_or_path`: 预训练模型内置名称或者模型所在目录，默认为`facebook/llama-7b`。
- `num_train_epochs`: 要执行的训练 epoch 总数（如果不是整数，将在停止训练之前执行最后一个 epoch
的小数部分百分比）。
- `max_steps`: 模型训练步数。
- `learning_rate`: 参数更新的学习率。
- `warmup_steps`: 学习率热启的步数。
- `eval_steps`: 模型评估的间隔步数。
- `logging_steps`: 训练日志打印的间隔步数。
- `save_steps`: 模型参数保存的间隔步数。
- `save_total_limit`: 模型 checkpoint 保存的份数。
- `output_dir`: 模型参数保存目录。
- `src_length`: 上下文的最大输入长度，默认为128.
- `tgt_length`: 生成文本的最大长度，默认为160.
- `gradient_accumulation_steps`: 模型参数梯度累积的步数，可用于扩大 batch size。实际的 batch_size = per_device_train_batch_size * gradient_accumulation_steps。
- `fp16`: 使用 float16 精度进行模型训练和推理。
- `fp16_opt_level`: float16 精度训练模式，`O2`表示纯 float16 训练。
- `recompute`: 使用重计算策略，开启后可节省训练显存。
- `do_train`: 是否训练模型。
- `do_eval`: 是否评估模型。
- `tensor_parallel_degree`: 模型并行数量。
- `do_generation`: 在评估的时候是否调用model.generate,默认为False。
- `lora`: 是否使用LoRA技术。
- `prefix_tuning`: 是否使用Prefix技术。
- `merge_weights`: 是否合并原始模型和Lora模型的权重。
- `r`: lora 算法中rank（秩）的值。
- `num_prefix_tokens`: prefix tuning算法中前缀token数量。

## 流水线并行
```shell
python -u  -m paddle.distributed.launch \
    --gpus "4,5,6,7"   finetune_generation.py \
    --model_name_or_path facebook/tiny-random-llama \
    --do_train \
    --do_eval \
    --num_train_epochs 1 \
    --dataloader_num_workers 1 \
    --gradient_accumulation_steps 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 16 \
    --tensor_parallel_degree 2 \
    --pipeline_parallel_degree 2 \
    --pipeline_parallel_config "disable_p2p_cache_shape" \
    --overwrite_output_dir \
    --output_dir ./checkpoints/ \
    --logging_steps 1 \
    --disable_tqdm 1 \
    --eval_steps 100 \
    --eval_with_do_generation 0 \
    --fp16 0\
    --fp16_opt_level O2 \
    --recompute 0 \
    --learning_rate 3e-5 \
    --lr_scheduler_type linear \
    --max_grad_norm 1.0 \
    --warmup_steps 20
```

## 指令微调

```shell
python -u  -m paddle.distributed.fleet.launch \
    --gpus "0,1,2,3" finetune_instruction_generation.py \
    --model_name_or_path facebook/llama-7b \
    --do_train \
    --do_eval \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --tensor_parallel_degree 4 \
    --overwrite_output_dir \
    --output_dir ./checkpoints/ \
    --logging_steps 10 \
    --fp16 \
    --fp16_opt_level O2 \
    --recompute \
    --learning_rate 3e-5 \
    --lr_scheduler_type linear \
    --max_grad_norm 1.0 \
    --warmup_steps 20 \
    --gradient_accumulation_steps 32 \
    --logging_steps 1 \
    --eval_steps 1000
```

<a name="2"></a>

## 模型预测

```shell
python predict_generation.py \
    --model_name_or_path ./checkpoints/
```

当ckpt为使用的tensor parallel存储为多分片格式时，也可使用此脚本预测，或者合并为一个单分片权重 例如下面4分片的例子（此模型为glm-10b-chinese）

```shell
-rw-r--r-- 1 root root  523 Apr 13 11:46 config.json
-rw-r--r-- 1 root root 3.2G Apr 13 11:46 model_state.tp00.pdparams
-rw-r--r-- 1 root root 3.2G Apr 13 11:46 model_state.tp01.pdparams
-rw-r--r-- 1 root root 3.2G Apr 13 11:46 model_state.tp02.pdparams
-rw-r--r-- 1 root root 3.2G Apr 13 11:46 model_state.tp03.pdparams
```

设置 merge_tensor_parallel_path，可以将merge好的参数存储到对应位置。不过不设置此参数，将只跑前向预测。

```shell
python -m paddle.distributed.launch --gpus 0,1,2,3 predict_generation.py \
    --model_name_or_path  ./checkpoints/checkpoint-100/ \
    --merge_tensor_parallel_path  ./checkpoints/llama-merged
```

### LoRA微调模型预测
对merge后的单分片模型也可以进行直接预测，脚本如下
```shell
 python predict_generation.py
    --model_name_or_path facebook/llama-7b \
    --lora_path ./checkpoints
```

### Prefix微调模型预测
对merge后的单分片模型也可以进行直接预测，脚本如下
```shell
 python predict_generation.py
    --model_name_or_path facebook/llama-7b \
    --prefix_path ./checkpoints
```

<a name="3"></a>

## 动转静

```shell
python export_generation_model.py \
    --model_path checkpoints/ \
    --output_path inference/llama
```

当在指定数据集上进行 LoRA finetune 后的导出脚本：


```shell
python export_generation_model.py
    --model_name_or_path facebook/llama-7b
    --output_path inference/llama
    --lora_path ./checkpoints
```

<a name="4"></a>

## 模型推理

```shell
python infer_generation.py \
    --model_dir inference \
    --model_prefix llama
```

结果：

```text
answer: linebacker context: The Broncos took an early lead in Super Bowl 50 and never trailed. Newton was limited by Denver's defense, which sacked him seven times and forced him into three turnovers, including a fumble which they recovered for a touchdown. Denver linebacker Von Miller was named Super Bowl MVP, recording five solo tackles, 2½ sacks, and two forced fumbles. </s>

question: What was von Miller's job title?
--------------------
answer: five context: The Broncos took an early lead in Super Bowl 50 and never trailed. Newton was limited by Denver's defense, which sacked him seven times and forced him into three turnovers, including a fumble which they recovered for a touchdown. Denver linebacker Von Miller was named Super Bowl MVP, recording five solo tackles, 2½ sacks, and two forced fumbles. </s>

question: How many total tackles did von Miller get in the Super Bowl?
--------------------
```
