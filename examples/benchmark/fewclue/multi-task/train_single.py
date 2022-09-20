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
from collections import defaultdict

import numpy as np

import paddle
from paddle.static import InputSpec
from paddle.metric import Accuracy
from paddlenlp.utils.log import logger
from paddlenlp.transformers import ErnieTokenizer, ErnieForMaskedLM
from paddlenlp.trainer import PdArgumentParser, EarlyStoppingCallback
from paddlenlp.prompt import (
    ManualTemplate,
    SoftTemplate,
    PrefixTemplate,
    ManualVerbalizer,
    MultiMaskVerbalizer,
    SoftVerbalizer,
    PromptTuningArguments,
    PromptTrainer,
    PromptModelForSequenceClassification,
)

from data import load_fewclue, PROMPT
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
    early_stop_patience: int = field(default=4, metadata={"help": "The descent steps before the training stops."})
# yapf: enable


def main(index):
    # Parse the arguments.
    parser = PdArgumentParser(
        (ModelArguments, DataArguments, PromptTuningArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)
    data_args.prompt = PROMPT[data_args.task_name][index][0]

    # Load the pretrained language model.
    model = ErnieForMaskedLM.from_pretrained(model_args.model_name_or_path)
    tokenizer = ErnieTokenizer.from_pretrained(model_args.model_name_or_path)

    # Define the template for preprocess and the verbalizer for postprocess.
    template = ManualTemplate(tokenizer, training_args.max_seq_length,
                              data_args.prompt)
    logger.info("Using template: {}".format(template.template))

    verbalizer = MultiMaskVerbalizer(
        tokenizer, label_words=PROMPT[data_args.task_name][index][1])

    # Load the few-shot datasets.
    train_ds, dev_ds, public_test_ds, test_ds = load_fewclue(
        task_name=data_args.task_name,
        split_id=data_args.split_id,
        verbalizer=verbalizer)

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
    def manual_metrics(eval_preds):
        metric = Accuracy()
        predictions = paddle.argmax(paddle.to_tensor(eval_preds.predictions),
                                    axis=-1)
        correct = metric.compute(predictions,
                                 paddle.to_tensor(eval_preds.label_ids))
        metric.update(correct)
        acc = metric.accumulate()
        return {'accuracy': acc}

    def multimask_metrics(eval_preds):
        predictions = paddle.nn.functional.softmax(paddle.to_tensor(
            eval_preds.predictions),
                                                   axis=-1).numpy()
        preds_list = []

        for label_id, tokens in enumerate(verbalizer.token_ids):
            token_pred = predictions[:, 0, tokens[0][0]]
            if len(tokens) > 1:
                for i, x in enumerate(tokens[1:]):
                    token_pred *= predictions[:, i + 1, x[0]]
            preds_list.append(token_pred)

        preds_list = np.stack(preds_list).T
        preds = np.argmax(preds_list, axis=1)
        label = eval_preds.label_ids
        acc = (preds == label).sum() / len(label)
        return {'accuracy': acc}

    def chid_metrics(eval_preds):
        from scipy.special import softmax
        predictions = paddle.nn.functional.softmax(paddle.to_tensor(
            eval_preds.predictions),
                                                   axis=-1).numpy()
        preds_list = []

        for label_id, tokens in enumerate(verbalizer.token_ids):
            token_pred = predictions[:, 0, tokens[0][0]]
            if len(tokens) > 1:
                for i, x in enumerate(tokens[1:]):
                    token_pred *= predictions[:, i + 1, x[0]]
            preds_list.append(token_pred)

        preds = np.stack(preds_list).T
        preds = softmax(preds, axis=1)[:, 1]
        preds = np.argmax(preds.reshape(-1, 7), axis=1)
        labels = np.argmax(eval_preds.label_ids.reshape(-1, 7), axis=1)

        print(preds)
        print(labels)

        acc = sum(preds == labels) / len(preds)
        return {'accuracy': acc}

    def csl_metrics(eval_preds, test_ds):
        from scipy.special import softmax
        predictions = paddle.nn.functional.softmax(paddle.to_tensor(
            eval_preds.predictions),
                                                   axis=-1).numpy()
        preds_list = []

        for label_id, tokens in enumerate(verbalizer.token_ids):
            token_pred = predictions[:, 0, tokens[0][0]]
            if len(tokens) > 1:
                for i, x in enumerate(tokens[1:]):
                    token_pred *= predictions[:, i + 1, x[0]]
            preds_list.append(token_pred)

        preds = np.stack(preds_list).T
        preds = softmax(preds, axis=1)
        ret_dict = defaultdict(list)
        print(preds.shape)
        print(len(test_ds))
        for idx, example in enumerate(test_ds):
            uid = getattr(example, "uid")
            ret_dict[uid].append(preds[idx])
        for uid, pred in ret_dict.items():
            # any()
            #all_pred = 1
            #for p in pred:
            #    if p[0] > p[1]:
            #        all_pred = 0
            #        break
            #ret_dict[uid] = all_pred

            # all()
            all_pred = [1., 1.]
            for p in pred:
                all_pred[0] *= p[0]
                all_pred[1] *= p[1]
            ret_dict[uid] = int(all_pred[0] < all_pred[1])

        correct = 0
        for x in test_ds:
            if x.labels == ret_dict[x.uid]:
                correct += 1
        return {"accuracy": correct / len(test_ds)}

    used_metrics = chid_metrics if data_args.task_name == "chid" else multimask_metrics
    if data_args.task_name == "csl":
        used_metrics = partial(csl_metrics, test_ds=dev_ds)

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
                            compute_metrics=used_metrics)

    # Traininig.
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=None)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    trainer.verbalizer.preds = None

    if data_args.task_name == "csl":
        trainer.compute_metrics = partial(csl_metrics, test_ds=public_test_ds)

    # Test.
    if model_args.do_test:
        test_ret = trainer.predict(public_test_ds)
        trainer.log_metrics("test", test_ret.metrics)

    if False:
        from collections import Counter
        counts = {}
        preds = trainer.verbalizer.preds

        for idx, example in enumerate(public_test_ds):
            if example.labels not in counts:
                counts[example.labels] = [[[] for _ in range(5)]
                                          for _ in range(len(preds))]
            for mask_id, pred in enumerate(preds):
                for j, x in enumerate(pred[idx]):
                    counts[example.labels][mask_id][j].append(x)

        for key, value in counts.items():
            print(key)
            for idx, x in enumerate(value):
                print('= ' + str(idx) + ' =' + '=' * 18)
                for i in x:
                    print(Counter(i).most_common())
                    print('-' * 10)

    # Prediction.
    if False and training_args.do_predict:
        test_ret = trainer.predict(test_ds)
        test_ret = postprocess(test_ret, test_ds, data_args.task_name,
                               verbalizer.ids_to_labels)
        save_to_file(test_ret, data_args.task_name)


if __name__ == '__main__':
    for i in range(1):
        print("=" * 20)
        main(i)
