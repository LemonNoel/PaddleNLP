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
import time
import json
from functools import partial
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

import paddle
from paddle.static import InputSpec
from paddle.metric import Accuracy
from paddlenlp.datasets import MapDataset
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

from utils import load_fewclue
from postprocess import save_to_file


# yapf: disable
@dataclass
class DataArguments:
    ckpt_plm: str = field(default=None, metadata={"help": "Path to the pretrained MaskedLM parameters"})
    fake_file: str = field(default=None, metadata={"help": "Path to the labeled data of unlabeled source data."})
    ckpt_model: str = field(default=None, metadata={"help": "Path to the pretrained PromptForSequenceClassification parameters"})
    task_name: str = field(default="eprstmt", metadata={"help": "Task name."})
    split_id: str = field(default="few_all", metadata={"help": "The postfix of subdataset."})
    t_type: str = field(default="auto", metadata={"help": "The class used for template"})
    t_index: int = field(default=0, metadata={"help": "The used template id"})
    v_type: str = field(default="manual", metadata={"help": "The class used for verbalizer, including manual, multi, soft, cls"})
    soft_encoder: str = field(default=None, metadata={"help": "The encoder type of soft template, `lstm`, `mlp` or None."})
    encoder_hidden_size: int = field(default=None, metadata={"help": "The dimension of soft embeddings."})
    do_analyze: bool = field(default=True, metadata={"help": "Whether to save all predictions for analysis"})
    do_label: bool = field(default=True, metadata={"help": "Whether to label unlabeled data."})
    label_threshold: float = field(default=0.8, metadata={"help": "When to label unlabeled data."})

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="ernie-1.0-large-zh-cw", metadata={"help": "Build-in pretrained model name or the path to local model."})
    export_type: str = field(default='paddle', metadata={"help": "The type to export. Support `paddle` and `onnx`."})
    do_test: bool = field(default=True, metadata={"help": "Evaluate the model on test_public dataset."})
    do_save: bool = field(default=True, metadata={"help": "Whether to save checkpoints during training."})
    early_stop_patience: int = field(default=4, metadata={"help": "The descent steps before the training stops."})
    dropout: float = field(default=0.1, metadata={"help": "The dropout used for pretrained model."})
# yapf: enable


def main():
    # Parse the arguments.
    parser = PdArgumentParser(
        (ModelArguments, DataArguments, PromptTuningArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    configs = json.load(open("template/%s.json" % data_args.task_name, "r"))
    data_args.prompt = configs["template"][data_args.t_index]["text"]
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
            model_args.model_name_or_path,
            hidden_dropout_prob=model_args.dropout,
            attention_probs_dropout_prob=model_args.dropout)
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
    train_ds, dev_ds, public_test_ds, test_ds, unlabeled_ds = load_fewclue(
        task_name=data_args.task_name,
        split_id=data_args.split_id,
        label_list=verbalizer.labels_to_ids,
        fake_file=data_args.fake_file)

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
        preds = paddle.nn.functional.softmax(paddle.to_tensor(
            eval_preds.predictions),
                                             axis=1)[:, 1]
        preds = paddle.argmax(preds.reshape(-1, 7), axis=1)
        labels = paddle.argmax(paddle.to_tensor(eval_preds.label_ids).reshape(
            -1, 7),
                               axis=1)
        acc = sum(preds == labels) / len(preds)
        return {'accuracy': float(acc.numpy())}

    def multi_uniprediction(preds, token_ids):
        preds = paddle.to_tensor(preds)
        preds_list = []
        for label_id, tokens in enumerate(token_ids):
            token_pred = preds[:, 0, tokens[0][0]]
            if preds.shape[1] > 1:
                for i, x in enumerate(tokens[1:]):
                    token_pred *= preds[:, i + 1, x[0]]
            preds_list.append(token_pred)
        preds_list = paddle.stack(preds_list).T
        return preds_list

    def multi_compute_metrics(eval_preds):
        predictions = paddle.nn.functional.softmax(paddle.to_tensor(
            eval_preds.predictions),
                                                   axis=-1)
        preds = multi_uniprediction(predictions, verbalizer.token_ids)
        preds = paddle.argmax(preds, axis=1)
        label = paddle.to_tensor(eval_preds.label_ids)
        acc = paddle.sum(preds == label) / len(label)
        return {'accuracy': float(acc.numpy())}

    def multi_compute_metrics_chid(eval_preds):
        logger.info("CHID! CHID! CHID!")
        predictions = paddle.nn.functional.softmax(paddle.to_tensor(
            eval_preds.predictions),
                                                   axis=-1)
        preds = multi_uniprediction(predictions, verbalizer.token_ids)
        preds = paddle.nn.functional.softmax(preds, axis=1)[:, 1]
        preds = paddle.argmax(preds.reshape([-1, 7]), axis=1)
        labels = paddle.argmax(paddle.to_tensor(eval_preds.label_ids).reshape(
            [-1, 7]),
                               axis=1)
        acc = paddle.sum(preds == labels) / len(preds)
        return {'accuracy': float(acc.numpy())}

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

    def tolist(data):
        if isinstance(data, paddle.Tensor):
            return data.numpy().tolist()
        elif isinstance(data, np.ndarray):
            return data.tolist()
        return data

    # Test.
    if model_args.do_test:
        test_ret = trainer.predict(public_test_ds)
        trainer.log_metrics("test", test_ret.metrics)

    if model_args.do_test and data_args.do_analyze:
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
                    paddle.nn.functional.softmax(paddle.to_tensor(
                        test_ret.predictions),
                                                 axis=-1), verbalizer.token_ids)
                if data_args.task_name == "chid":
                    mask_preds = paddle.nn.functional.softmax(mask_preds,
                                                              axis=1)[:, 1]
                    mask_preds = paddle.argmax(mask_preds.reshape([-1, 7]),
                                               axis=1).numpy().tolist()
                else:
                    mask_preds = paddle.argmax(mask_preds, axis=1)
                    mask_preds = [
                        id2label[x] for x in mask_preds.numpy().tolist()
                    ]
                for pred, mask_pred, lb in zip(preds, mask_preds, labels):
                    if data_args.task_name != "chid":
                        pred = tokenizer.convert_ids_to_tokens(pred)
                    to_save = [tolist(x) for x in [pred, mask_pred, lb]]
                    fp.write(json.dumps(to_save, ensure_ascii=False) + "\n")
            else:
                for pred, lb in zip(preds, labels):
                    if data_args.task_name != "chid":
                        pred = id2label[pred]
                    to_save = [tolist(x) for x in [pred, lb]]
                    fp.write(json.dumps(to_save, ensure_ascii=False) + "\n")

    def postprocess(data_ret, data_ds):
        # 构造有序的标签id映射
        id_to_label = {
            i: l
            for i, l in enumerate(
                sorted([x for x in configs["verbalizer"].keys()]))
        }
        # 计算标签概率和标签id
        ret_list = []
        pro_list = []
        if data_args.v_type == "multi":
            mask_preds = multi_uniprediction(
                paddle.nn.functional.softmax(paddle.to_tensor(
                    data_ret.predictions),
                                             axis=-1), verbalizer.token_ids)
            if data_args.task_name == "chid":
                mask_preds = paddle.nn.functional.softmax(mask_preds, axis=1)[:,
                                                                              1]
                preds = paddle.argmax(mask_preds.reshape([-1, 7]),
                                      axis=1).numpy()
                probs = paddle.max(mask_preds.reshape([-1, 7]), axis=1).numpy()
            else:
                preds = paddle.argmax(mask_preds, axis=1).numpy()
                probs = paddle.max(mask_preds, axis=1).numpy()
        else:
            preds = paddle.argmax(paddle.to_tensor(data_ret.predictions),
                                  axis=1).numpy()
            probs = paddle.max(paddle.to_tensor(data_ret.predictions),
                               axis=1).numpy()

        preds = tolist(preds)
        probs = tolist(probs)

        # 读取 iflytek 和 tnews 的标签映射
        remap = configs.get("label_ids", None)

        # 按数据集分别构造结果集合和概率集合
        if data_args.task_name == "chid":
            for idx, example in tqdm(enumerate([x for x in data_ds][::7])):
                ret_list.append({
                    "id": getattr(example, "uid", idx),
                    "answer": preds[idx]
                })
                pro_list.append({
                    "id": getattr(example, "uid", idx),
                    "answer": preds[idx],
                    "prob": probs[idx],
                })
        else:
            if data_args.task_name in ["bustm", "csl"]:
                for idx, example in tqdm(enumerate(data_ds)):
                    uid = getattr(example, "uid", idx)
                    ret_list.append({"id": uid, "label": str(preds[idx])})
                    pro_list.append({
                        "id": uid,
                        "label": str(preds[idx]),
                        "prob": probs[idx]
                    })
            elif data_args.task_name in [
                    "cluewsc", "eprstmt", "ocnli", "csldcp"
            ]:
                for idx, example in tqdm(enumerate(data_ds)):
                    uid = getattr(example, "uid", idx)
                    ret_list.append({
                        "id": uid,
                        "label": id_to_label[preds[idx]]
                    })
                    pro_list.append({
                        "id": uid,
                        "label": id_to_label[preds[idx]],
                        "prob": probs[idx]
                    })
            elif data_args.task_name in ["iflytek", "tnews"]:
                for idx, example in tqdm(enumerate(data_ds)):
                    uid = getattr(example, "uid", idx)
                    ret_list.append({
                        "id": uid,
                        "label": str(remap[id_to_label[preds[idx]]])
                    })
                    pro_list.append({
                        "id": uid,
                        "label": str(remap[id_to_label[preds[idx]]]),
                        "prob": probs[idx]
                    })
        return ret_list, pro_list

    time_stamp = time.strftime("%m%d-%H-%M—%S", time.localtime())

    # Tag unlabeled data.
    if data_args.do_label:
        label_ret = trainer.predict(unlabeled_ds)
        print("Prediction done.")
        _, prob_list = postprocess(label_ret, unlabeled_ds)
        save_to_file(prob_list,
                     data_args.task_name,
                     save_path="fake_data_%s" % time_stamp)

    # Prediction.
    # TODO: TO MODIFY FOR NEW VERSION
    if training_args.do_predict:
        test_ret = trainer.predict(test_ds)
        print("Prediction done.")
        test_ret, _ = postprocess(test_ret, test_ds)
        save_to_file(test_ret,
                     data_args.task_name,
                     save_path="fewclue_submit_examples_%s" % time_stamp)


if __name__ == '__main__':
    main()
