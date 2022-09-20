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

import os
from dataclasses import dataclass, field
from functools import partial

import numpy as np
from tqdm import tqdm

import paddle
from paddle.metric import Accuracy
from paddle.static import InputSpec
from paddlenlp.datasets import load_dataset
from paddlenlp.utils.log import logger
from paddlenlp.transformers import ErnieTokenizer, ErnieForSequenceClassification, export_model
from paddlenlp.trainer import PdArgumentParser, Trainer, TrainingArguments


@dataclass
class DataArguments:
    data_dir: str = field(
        metadata={
            "help":
            "The dataset dictionary includes train.txt, dev.txt and label.txt files."
        })
    max_seq_length: int = field(
        metadata={
            "help":
            "The maximum length of input sequence after tokenization. Sequences longer "
            "than it will be truncated, sequences shorter will be padded."
        })
    num_labels: int = field(metadata={"help": "The original number of labels."})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="ernie-3.0-medium-zh",
        metadata={
            "help": "The build-in pretrained model or the path to local model."
        })
    export_type: str = field(
        default='paddle',
        metadata={"help": "The type to export. Support `paddle` and `onnx`."})


@paddle.no_grad()
def evaluate(model, dataloader, num_labels):
    model.eval()
    preds_list = []
    label_list = []
    for inputs in tqdm(dataloader):
        label = inputs.pop("labels")
        label_list.append(label)
        preds = model(**inputs)
        preds_list.append(paddle.argmax(preds, axis=1).numpy())
    preds_list = np.concatenate(preds_list)
    label_list = np.concatenate(label_list)

    total_num, correct_num = 0, 0
    for idx in range(0, len(label_list), num_labels):
        preds = np.argmax(preds_list[idx:idx + num_labels])
        label = np.argmax(label_list[idx:idx + num_labels])
        total_num += 1
        if (preds == label).all():
            correct_num += 1
    acc = correct_num / total_num
    logger.info("Accuracy for original labels: %.4f" % acc)


def main():
    # Parse the arguments.
    parser = PdArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)

    # Load the pretrained language model.
    model = ErnieForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_classes=2)
    tokenizer = ErnieTokenizer.from_pretrained(model_args.model_name_or_path)

    # Load the datasets for binary classificartion.
    def reader(data_dir, split):
        data_file = os.path.join(data_dir, split + ".txt")
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                text_a, text_b, label = line.strip().split("\t")
                yield {"text_a": text_a, "text_b": text_b, "label": int(label)}

    train_ds = load_dataset(reader,
                            data_dir=data_args.data_dir,
                            split="train",
                            lazy=False)
    dev_ds = load_dataset(reader,
                          data_dir=data_args.data_dir,
                          split="dev",
                          lazy=False)

    def convert_example(example, tokenizer, max_seq_length):
        sentence = example["text_a"]
        text_label = example["text_b"]
        encoded_inputs = tokenizer(text=sentence,
                                   text_pair=text_label,
                                   max_seq_len=max_seq_length,
                                   truncation_strategy="only_first")

        return {
            "input_ids": encoded_inputs["input_ids"],
            "token_type_ids": encoded_inputs["token_type_ids"],
            "labels": example["label"]
        }

    trans_fn = partial(convert_example,
                       tokenizer=tokenizer,
                       max_seq_length=data_args.max_seq_length)
    train_ds = train_ds.map(trans_fn)
    dev_ds = dev_ds.map(trans_fn)

    # Define the criterion.
    criterion = paddle.nn.CrossEntropyLoss()

    # Define the metric function.
    def compute_metrics(eval_preds):
        metric = Accuracy()
        correct = metric.compute(paddle.to_tensor(eval_preds.predictions),
                                 paddle.to_tensor(eval_preds.label_ids))
        metric.update(correct)
        acc = metric.accumulate()
        return {'accuracy': acc}

    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      criterion=criterion,
                      train_dataset=train_ds,
                      eval_dataset=dev_ds,
                      compute_metrics=compute_metrics)

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=None)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_predict:
        eval_dataloader = trainer.get_eval_dataloader(dev_ds)
        evaluate(trainer.model, eval_dataloader, data_args.num_labels)

    if training_args.do_export:
        input_spec = [
            InputSpec(shape=[None, None], dtype="int64"),  # input_ids
            InputSpec(shape=[None, None], dtype="int64"),  # token_type_ids
        ]
        export_path = os.path.join(training_args.output_dir, 'export')
        os.makedirs(export_path, exist_ok=True)
        export_model(prompt_model, input_spec, export_path,
                     model_args.export_type)


if __name__ == '__main__':
    main()
