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

from dataclasses import dataclass, field
import os
from functools import partial
from collections import Counter
from tqdm import tqdm
from paddle.io import BatchSampler, DataLoader
from paddlenlp.data import Pad, Tuple
import numpy as np

import paddle
from paddle.static import InputSpec
from paddle.metric import Accuracy
from paddlenlp.utils.log import logger
from paddlenlp.transformers import T5Tokenizer, T5ForConditionalGeneration
from paddlenlp.trainer import PdArgumentParser, EarlyStoppingCallback
from paddlenlp.prompt import (
    AutoTemplate,
    ManualTemplate,
    SoftTemplate,
    PrefixTemplate,
    PromptTuningArguments,
    PromptTrainer,
    PromptModelForGeneration,
)

from utils import load_fewclue, LABEL_LIST, LABEL_MAP
from postprocess import postprocess, save_to_file


# yapf: disable
@dataclass
class DataArguments:
    task_name: str = field(default="tnews", metadata={"help": "Task name in FewCLUE."})
    split_id: str = field(default="0", metadata={"help": "The postfix of subdataset."})
    prompt: str = field(default=None, metadata={"help": "The input prompt for tuning."})
    soft_encoder: str = field(default="lstm", metadata={"help": "The encoder type of soft template, `lstm`, `mlp` or None."})
    encoder_hidden_size: int = field(default=200, metadata={"help": "The dimension of soft embeddings."})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="La", metadata={"help": "Build-in pretrained model name or the path to local model."})
    export_type: str = field(default='paddle', metadata={"help": "The type to export. Support `paddle` and `onnx`."})
    do_test: bool = field(default=False, metadata={"help": "Evaluate the model on test_public dataset."})
    do_save: bool = field(default=False, metadata={"help": "Whether to save checkpoints during training."})
    early_stop_patience: int = field(default=4, metadata={"help": "The descent steps before the training stops."})
# yapf: enable


def main():
    # Parse the arguments.
    parser = PdArgumentParser(
        (ModelArguments, DataArguments, PromptTuningArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)

    # Load the pretrained language model.
    labels = LABEL_LIST[data_args.task_name]
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path)
    tokenizer = T5Tokenizer.from_pretrained(model_args.model_name_or_path)
    model.eval()

    # Define the template for preprocess.
    template = ManualTemplate(tokenizer, training_args.max_seq_length,
                              data_args.prompt)
    logger.info("Using template: {}".format(template.template))

    # Load the few-shot datasets.
    label_dict = {k: v for k, v in LABEL_MAP[data_args.task_name].items()}
    #label_dict = None
    train_ds, dev_ds, public_test_ds, test_ds = load_fewclue(
        task_name=data_args.task_name,
        split_id=data_args.split_id,
        label_list=label_dict)

    def trans_func(example, tokenizer, args):
        inputs = tokenizer(example.text_a,
                           max_seq_len=args.max_seq_length,
                           return_token_type_ids=False,
                           return_attention_mask=True)
        return inputs["input_ids"], inputs["attention_mask"], tokenizer(
            example.labels)["input_ids"]
        #return inputs["input_ids"], inputs["attention_mask"], example.labels

    public_test_ds = public_test_ds.map(
        partial(trans_func, tokenizer=tokenizer, args=training_args))

    batch_sampler = BatchSampler(public_test_ds, batch_size=32, shuffle=False)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),
        Pad(axis=0, pad_val=-100, dtype="int64"),
    ): fn(samples)

    data_loader = DataLoader(dataset=public_test_ds,
                             batch_sampler=batch_sampler,
                             collate_fn=batchify_fn,
                             num_workers=2,
                             return_list=True)

    def compute_metrics(eval_preds, labels):
        # C2-mengzi
        total, correct = 0, 0
        words = []
        for pred, label in zip(eval_preds, labels):
            #print(tokenizer.decode(Counter(pred).most_common(1)[0][0]))
            #print(tokenizer.decode(Counter(label).most_common(1)[0][0]))
            pred = tokenizer.decode(pred, skip_special_tokens=True)
            for idx in range(len(label)):
                if label[idx] < 0:
                    label[idx] = 1
            label = tokenizer.decode(label, skip_special_tokens=True)
            print(pred)
            print(label)
            print('-' * 10)
            words.append(pred)
            total += 1
            if pred == label:
                correct += 1
        print(Counter(words).most_common())
        return {'accuracy': correct / total}

    eval_preds = []
    labels = []
    for binputs, bmask, blabels in tqdm(data_loader):
        outputs = model.generate(input_ids=binputs,
                                 attention_mask=bmask,
                                 max_length=training_args.max_seq_length)[0]
        eval_preds.extend(outputs.numpy().tolist())
        labels.extend(blabels.numpy().tolist())
    print(compute_metrics(eval_preds, labels))


if __name__ == '__main__':
    main()
