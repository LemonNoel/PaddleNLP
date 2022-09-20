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

    # Define the template for preprocess.
    template = ManualTemplate(tokenizer, training_args.max_seq_length,
                              data_args.prompt)
    logger.info("Using template: {}".format(template.template))

    # Load the few-shot datasets.
    label_dict = {
        k: tokenizer(v)["input_ids"]
        for k, v in LABEL_MAP[data_args.task_name].items()
    }
    train_ds, dev_ds, public_test_ds, test_ds = load_fewclue(
        task_name=data_args.task_name,
        split_id=data_args.split_id,
        label_list=label_dict)

    # Define the criterion.
    criterion = paddle.nn.CrossEntropyLoss()
    criterion = None

    # Initialize the prompt model with the above variables.
    prompt_model = PromptModelForGeneration(
        model,
        template,
        training_args.max_seq_length,
        freeze_plm=training_args.freeze_plm,
        freeze_dropout=training_args.freeze_dropout)

    def compute_metrics(eval_preds):
        # C2-mengzi
        total, correct = 0, 0
        for pred, label in zip(eval_preds.predictions, eval_preds.label_ids):
            total += 1
            if (pred == label).all():
                correct += 1
        return {'accuracy': correct / total}

    used_metrics = chid_compute_metrics if data_args.task_name == "chid" else compute_metrics

    # Deine the early-stopping callback.
    if model_args.do_save:
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=model_args.early_stop_patience,
                early_stopping_threshold=0.)
        ]
    else:
        callbacks = None

    # Initialize the trainer.
    trainer = PromptTrainer(model=prompt_model,
                            tokenizer=tokenizer,
                            args=training_args,
                            criterion=criterion,
                            train_dataset=train_ds,
                            eval_dataset=dev_ds,
                            callbacks=callbacks,
                            compute_metrics=None)  #used_metrics)

    # Traininig.
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=None)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Test.
    if model_args.do_test:
        test_ret = trainer.predict(public_test_ds)
        trainer.log_metrics("test", test_ret.metrics)

    # Prediction.
    label_map = {i: k for i, k in enumerate(labels)}
    if training_args.do_predict:
        test_ret = trainer.predict(test_ds)
        test_ret = postprocess(test_ret, test_ds, data_args.task_name,
                               label_map)
        save_to_file(test_ret, data_args.task_name)


if __name__ == '__main__':
    main()
