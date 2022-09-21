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
from functools import partial
import os
import json

import numpy as np
from scipy.special import softmax

import paddle
from paddle.static import InputSpec
from paddle.metric import Accuracy
from paddlenlp.utils.log import logger
from paddlenlp.transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
from paddlenlp.trainer import PdArgumentParser, EarlyStoppingCallback
from paddlenlp.prompt import (
    AutoTemplate,
    ManualTemplate,
    SoftTemplate,
    PrefixTemplate,
    ManualVerbalizer,
    SoftVerbalizer,
    MultiMaskVerbalizer,
    PromptTuningArguments,
    PromptTrainer,
    PromptModelForSequenceClassification,
)

from utils import load_fewclue, LABEL_LIST, LABEL_MAP
from postprocess import postprocess, save_to_file


# yapf: disable
@dataclass
class DataArguments:
    ckpt_plm: str = field(default=None, metadata={"help": "Path to the pretrained MaskedLM parameters"})
    ckpt_model: str = field(default=None, metadata={"help": "Path to the pretrained PromptForSequenceClassification parameters"})
    task_name: str = field(default="eprstmt", metadata={"help": "Task name."})
    split_id: str = field(default="0", metadata={"help": "The postfix of subdataset."})
    t_type: str = field(default="auto", metadata={"help": "The class used for template"})
    v_type: str = field(default="manual", metadata={"help": "The class used for verbalizer, including manual, multi, soft, cls"})
    soft_encoder: str = field(default="lstm", metadata={"help": "The encoder type of soft template, `lstm`, `mlp` or None."})
    encoder_hidden_size: int = field(default=200, metadata={"help": "The dimension of soft embeddings."})
    do_analyze: bool = field(default=False, metadata={"help": "Whether to save all predictions for analysis"})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="ernie-3.0-base-zh", metadata={"help": "Build-in pretrained model name or the path to local model."})
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
    configs = json.load(open("template/%s.json" % data_args.task_name, "r"))
    data_args.prompt = configs["template"]
    data_args.verbalizer = configs["verbalizer"]

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)

    # Load the pretrained language model.
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if data_args.v_type == "cls":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            num_classes=len(data_args.verbalizer))
    else:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path)
    if data_args.ckpt_plm is not None:
        print("Loading Pretrained Model from %s" % data_args.ckpt_plm)
        state_dict = paddle.load(data_args.ckpt_plm)
        model.set_state_dict(state_dict)
        del state_dict

    # Define the template for preprocess and the verbalizer for postprocess.
    if data_args.t_type == "auto":
        template = AutoTemplate.create_from(data_args.prompt, tokenizer,
                                            training_args.max_seq_length, model,
                                            data_args.soft_encoder,
                                            data_args.encoder_hidden_size)
    elif data_args.t_type == "prefix":
        template = PrefixTemplate(tokenizer, training_args.max_seq_length,
                                  model, data_args.prompt,
                                  data_args.soft_encoder,
                                  data_args.encoder_hidden_size)
    else:
        raise ValueError(
            "Unsupported Template %s. Please use `auto` or `prefix`" %
            data_args.t_type)
    logger.info("Using template: {}".format(template.template))

    if data_args.v_type == "manual":
        verbalizer = ManualVerbalizer(tokenizer,
                                      label_words=data_args.verbalizer)
    elif data_args.v_type == "multi":
        verbalizer = MultiMaskVerbalizer(tokenizer,
                                         label_words=data_args.verbalizer)
    elif data_args.v_type == "soft":
        verbalizer = SoftVerbalizer(tokenizer,
                                    model,
                                    label_words=data_args.verbalizer)
    elif data_args.v_type == "cls":
        verbalizer = ManualVerbalizer(tokenizer,
                                      label_words=data_args.verbalizer)
    else:
        raise ValueError(
            "Unsupported Verbalizer %s. Please use `manual`, `multi` or `soft`"
            % data_args.v_type)

    # Load the few-shot datasets.
    train_ds, dev_ds, public_test_ds, test_ds = load_fewclue(
        task_name=data_args.task_name,
        split_id=data_args.split_id,
        label_list=verbalizer.labels_to_ids)

    if data_args.v_type == "cls":
        verbalizer = None

    for x in train_ds:
        logger.info(
            "Example: " +
            json.dumps([x.text_a, x.text_b, x.labels], ensure_ascii=False))
        break

    # Define the criterion.
    criterion = paddle.nn.CrossEntropyLoss()

    # Initialize the prompt model with the above variables.
    prompt_model = PromptModelForSequenceClassification(
        model,
        template,
        verbalizer,
        freeze_plm=training_args.freeze_plm,
        freeze_dropout=training_args.freeze_dropout)
    if data_args.ckpt_model is not None:
        print("Loading PromptModel from %s..." % data_args.ckpt_model)
        state_dict = paddle.load(data_args.ckpt_model)
        prompt_model.set_state_dict(state_dict)
        del state_dict

    # Define the metric function.
    def cls_compute_metrics(eval_preds):
        metric = Accuracy()
        correct = metric.compute(paddle.to_tensor(eval_preds.predictions),
                                 paddle.to_tensor(eval_preds.label_ids))
        metric.update(correct)
        acc = metric.accumulate()
        return {'accuracy': acc}

    def cls_compute_metrics_chid(eval_preds):
        # chid IDEA B.1
        preds = softmax(eval_preds.predictions, axis=1)[:, 1]
        preds = np.argmax(preds.reshape(-1, 7), axis=1)
        labels = np.argmax(eval_preds.label_ids.reshape(-1, 7), axis=1)
        acc = sum(preds == labels) / len(preds)
        return {'accuracy': acc}

    def multi_uniprediction(preds, token_ids):
        preds_list = []
        for label_id, tokens in enumerate(token_ids):
            token_pred = preds[:, 0, tokens[0][0]]
            if preds.shape[1] > 1:
                for i, x in enumerate(tokens[1:]):
                    token_pred *= preds[:, i + 1, x[0]]
            preds_list.append(token_pred)
        preds_list = np.stack(preds_list).T
        return preds_list

    def multi_compute_metrics(eval_preds):
        predictions = softmax(eval_preds.predictions, axis=-1)
        preds = multi_uniprediction(predictions, verbalizer.token_ids)
        preds = np.argmax(preds, axis=1)
        label = eval_preds.label_ids
        acc = (preds == label).sum() / len(label)
        return {'accuracy': acc}

    def multi_compute_metrics_chid(eval_preds):
        predictions = softmax(eval_preds.predictions, axis=-1)
        preds = multi_uniprediction(predictions, verbalizer.token_ids)
        preds = softmax(preds, axis=1)[:, 1]
        preds = np.argmax(preds.reshape(-1, 7), axis=1)
        labels = np.argmax(eval_preds.label_ids.reshape(-1, 7), axis=1)
        acc = sum(preds == labels) / len(preds)
        return {'accuracy': acc}

    def csl_compute_metrics(eval_preds, data_ds):
        preds = softmax(eval_preds.predictions, axis=1)[:, 1]
        preds = (preds > 0.5)
        result = {}
        for idx, example in enumerate(data_ds):
            if example.uid not in result:
                result[example.uid] = preds[idx]
            else:
                result[example.uid] = preds[idx] and result[example.uid]
        acc = np.mean([float(x) for x in result.values()])
        return {'accuracy': float(acc)}

    if data_args.v_type == "multi":
        compute_metrics = multi_compute_metrics
        if data_args.task_name == "chid":
            compute_metrics = multi_compute_metrics_chid
    else:
        compute_metrics = cls_compute_metrics
        if data_args.task_name == "chid":
            compute_metrics = cls_compute_metrics_chid

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
                            callbacks=None,
                            compute_metrics=compute_metrics)

    # Traininig.
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=None)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        state_dict = trainer.model.plm.state_dict()
        paddle.save(state_dict,
                    os.path.join(training_args.output_dir, "best_plm.pdparams"))

    # if data_args.task_name == "csl":
    #    trainer.compute_metrics = partial(csl_compute_metrics, data_ds=public_test_ds)

    # Test.
    if model_args.do_test:
        test_ret = trainer.predict(public_test_ds)
        trainer.log_metrics("test", test_ret.metrics)
        preds = test_ret.predictions
        preds = np.argmax(np.array(preds), axis=-1)
        id2label = {
            idx: l
            for idx, l in enumerate(
                sorted([x for x in data_args.verbalizer.keys()]))
        }
        labels = [id2label[x] for x in test_ret.label_ids]
        with open(os.path.join(training_args.output_dir, "test_analysis.txt"),
                  "w") as fp:
            if data_args.v_type == "multi":
                mask_preds = multi_uniprediction(
                    softmax(test_ret.predictions, axis=-1),
                    verbalizer.token_ids)
                if data_args.task_name == "chid":
                    mask_preds = softmax(mask_preds, axis=1)[:, 1]
                    mask_preds = np.argmax(mask_preds.reshape(-1, 7), axis=1)
                else:
                    mask_preds = np.argmax(mask_preds, axis=1)
                mask_preds = [id2label[x] for x in mask_preds]
                for pred, mask_pred, lb in zip(preds, mask_preds, labels):
                    fp.write(
                        json.dumps([
                            tokenizer.convert_ids_to_tokens(pred), mask_pred, lb
                        ],
                                   ensure_ascii=False) + "\n")
            else:
                for pred, lb in zip(preds, labels):
                    fp.write(
                        json.dumps([id2label[pred], lb], ensure_ascii=False) +
                        "\n")

    # Prediction.
    # TODO: TO MODIFY FOR NEW VERSION
    if training_args.do_predict:
        test_ret = trainer.predict(test_ds)
        print("Prediction done.")
        test_ret = postprocess(test_ret,
                               test_ds,
                               data_args.task_name,
                               verbalizer.ids_to_labels,
                               verbalizer=verbalizer)
        save_to_file(test_ret, data_args.task_name)


if __name__ == '__main__':
    main()
