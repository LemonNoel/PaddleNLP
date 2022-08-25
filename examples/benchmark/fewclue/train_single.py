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
from paddlenlp.transformers import ErnieTokenizer, ErnieForMaskedLM
from paddlenlp.trainer import PdArgumentParser, EarlyStoppingCallback
from paddlenlp.prompt import (
    AutoTemplate,
    ManualTemplate,
    SoftTemplate,
    PrefixTemplate,
    ManualVerbalizer,
    SoftVerbalizer,
    PromptTuningArguments,
    PromptTrainer,
    PromptModelForSequenceClassification,
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
    model_name_or_path: str = field(default="ernie-3.0-base-zh", metadata={"help": "Build-in pretrained model name or the path to local model."})
    export_type: str = field(default='paddle', metadata={"help": "The type to export. Support `paddle` and `onnx`."})
    do_test: bool = field(default=False, metadata={"help": "Evaluate the model on test_public dataset."})
    do_save: bool = field(default=False, metadata={"help": "Whether to save checkpoints during training."})
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
    model = ErnieForMaskedLM.from_pretrained(model_args.model_name_or_path)
    tokenizer = ErnieTokenizer.from_pretrained(model_args.model_name_or_path)

    # Define the template for preprocess and the verbalizer for postprocess.
    template = SoftTemplate(tokenizer,
                            training_args.max_seq_length,
                            model,
                            data_args.prompt,
                            prompt_encoder=data_args.soft_encoder,
                            encoder_hidden_size=data_args.encoder_hidden_size)
    logger.info("Using template: {}".format(template.template))

    labels = LABEL_LIST[data_args.task_name]
    label_words = LABEL_MAP[data_args.task_name]
    verbalizer = SoftVerbalizer(tokenizer, model, labels, label_words)

    # Load the few-shot datasets.
    train_ds, dev_ds, public_test_ds, test_ds = load_fewclue(
        task_name=data_args.task_name,
        split_id=data_args.split_id,
        label_list=verbalizer.labels_to_ids)

    # Define the criterion.
    criterion = paddle.nn.CrossEntropyLoss()

    # Initialize the prompt model with the above variables.
    prompt_model = PromptModelForSequenceClassification(
        model,
        template,
        verbalizer,
        freeze_plm=training_args.freeze_plm,
        freeze_dropout=training_args.freeze_dropout)

    # Define the metric function.
    def _compute_metrics(eval_preds):
        metric = Accuracy()
        correct = metric.compute(paddle.to_tensor(eval_preds.predictions),
                                 paddle.to_tensor(eval_preds.label_ids))
        metric.update(correct)
        acc = metric.accumulate()
        return {'accuracy': acc}

    def compute_metrics(eval_preds):
        # chid IDEA B.1
        from scipy.special import softmax
        preds = softmax(eval_preds.predictions, axis=1)[:, 1]
        preds = np.argmax(preds.reshape(-1, 7), axis=1)
        labels = np.argmax(eval_preds.label_ids.reshape(-1, 7), axis=1)
        acc = sum(preds == labels) / len(preds)
        return {'accuracy': acc}

    # Deine the early-stopping callback.
    if model_args.do_save:
        callbacks = [
            EarlyStoppingCallback(early_stopping_patience=4,
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
                            compute_metrics=compute_metrics)

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
    if training_args.do_predict:
        test_ret = trainer.predict(test_ds)
        test_ret = postprocess(test_ret, test_ds, data_args.task_name,
                               verbalizer.ids_to_labels)
        save_to_file(test_ret, data_args.task_name)


if __name__ == '__main__':
    main()
