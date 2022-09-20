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
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import InputSpec
from paddle.metric import Accuracy
from visualdl import LogWriter

from paddlenlp.utils.log import logger
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoTokenizer, AutoModelForMaskedLM, export_model
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.prompt import (PromptTuningArguments, PromptTrainer,
                              InputExample, AutoTemplate, ManualVerbalizer,
                              MLMTokenizerWrapper, PromptModelForClassification)


def load_local_dataset(data_path, splits, label_list):

    def reader(data_file):
        import csv
        with open(data_file, 'r', encoding='utf-8-sig') as f:
            data = [x for x in csv.reader(f, delimiter='\t')][1:]
            for idx, line in enumerate(data):
                yield InputExample(str(idx),
                                   text_a=line[0],
                                   text_b=None,
                                   labels=line[1])

    split_map = split_map = {
        "train": "train.tsv",
        "dev": "dev.tsv",
        "test": "test.tsv"
    }
    datasets = []
    for split in splits:
        data_file = os.path.join(data_path, split_map[split])
        datasets.append(
            load_dataset(reader,
                         data_file=data_file,
                         label_list=label_list,
                         lazy=False))
    return datasets


@dataclass
class DataArguments:
    """
    The arguments' subset to formalize the input data.
    """
    dataset: str = field(
        default="SST-2",
        metadata={"help": "The name of the build-in or customized dataset."})
    data_path: str = field(
        default=
        "/ssd2/wanghuijuan03/githubs/PaddleNLP/examples/few_shot/RGL/data/k-shot/SST-2/16-13/",
        metadata={"help": "The path to the customized dataset."})

    prompt: str = field(default=None,
                        metadata={"help": "The input prompt for tuning."})
    verbalizer: str = field(
        default=None, metadata={"help": "The mapping from labels to words."})

    def __post_init__(self):
        self.dataset = self.dataset.lower()


@dataclass
class ModelArguments:
    """
    The arguments' subset for pretrained model/tokenizer/config.
    """

    model_name_or_path: str = field(
        default="ernie-2.0-large-en",
        metadata={
            "help": "The build-in pretrained LM or the path to local model."
        })
    export_type: str = field(
        default='paddle',
        metadata={
            "help": "The type of model to export. Support `paddle` and `onnx`."
        })


def main():
    parser = PdArgumentParser(
        (ModelArguments, DataArguments, PromptTuningArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)

    logger.warning(
        f"Process rank: {training_args.local_rank}, " +
        f"device: {training_args.device}, " +
        f"world_size: {training_args.world_size}, " +
        f"distributed training: {bool(training_args.local_rank != -1)}, " +
        f"16-bits training: {training_args.fp16}")

    model = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    template = AutoTemplate.create_from(data_args.prompt,
                                        tokenizer,
                                        model=model,
                                        prompt_encoder='lstm')
    logger.info("Set template: {}".format(template.template))
    verbalizer = ManualVerbalizer.from_file(tokenizer,
                                            os.path.join(
                                                data_args.data_path,
                                                'label.txt'),
                                            prefix=' ')
    logger.info("Set verbalizer: {}".format(data_args.verbalizer))

    train_ds, dev_ds, test_ds = load_local_dataset(
        data_path=data_args.data_path,
        splits=['train', 'dev', 'test'],
        label_list=verbalizer.labels_to_ids)

    criterion = nn.CrossEntropyLoss()

    prompt_model = PromptModelForClassification(
        model,
        template,
        verbalizer,
        freeze_plm=training_args.freeze_plm,
        freeze_dropout=training_args.freeze_dropout)

    trainer = PromptTrainer(model=prompt_model,
                            tokenizer=tokenizer,
                            args=training_args,
                            criterion=criterion,
                            train_dataset=train_ds,
                            eval_dataset=dev_ds)

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=None)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_predict:
        test_result = trainer.predict(test_ds)
        trainer.log_metrics("test", test_result.metrics)
        if test_result.label_ids is None:
            paddle.save(
                test_result.predictions,
                os.path.join(training_args.output_dir, "test_results.pdtensor"),
            )

    if training_args.do_export:
        input_spec = [
            InputSpec(shape=[None, None], dtype="int64"),  # input_ids
            InputSpec(shape=[None, None], dtype="float32")  # attention_mask
        ]
        export_path = os.path.join(training_args.output_dir, 'export')
        os.makedirs(export_path, exist_ok=True)
        export_model(prompt_model, input_spec, export_path,
                     model_args.export_type)


if __name__ == '__main__':
    main()
