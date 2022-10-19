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

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import Counter
import os

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import InputSpec

from ..datasets import MapDataset
from ..utils.log import logger
from ..trainer import Trainer, TrainerCallback
from ..trainer.trainer_utils import EvalPrediction, get_scheduler
from ..data import DataCollator
from ..losses import RDropLoss
from ..transformers import PretrainedTokenizer, export_model

from .template import AutoTemplate
from .verbalizer import SoftVerbalizer, MultiMaskVerbalizer
from .prompt_utils import InputFeatures, signature
from .prompt_args import PromptTuningArguments

__all__ = [
    "PromptTrainer", "PromptModelForSequenceClassification",
    "PromptModelForGeneration"
]

PROMPT_NAME = "prompt.pdparams"


class PromptTrainer(Trainer):
    """
    PromptTrainer is a feature-complete training and eval loop for PaddleNLP
    on prompt-tuning.
    """

    def __init__(self,
                 model: Union[nn.Layer],
                 tokenizer: PretrainedTokenizer,
                 criterion: Union[nn.Layer],
                 args: PromptTuningArguments = None,
                 data_collator: Optional[DataCollator] = None,
                 train_dataset: Optional[MapDataset] = None,
                 eval_dataset: Optional[MapDataset] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction],
                                                    Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[paddle.optimizer.Optimizer,
                                   paddle.optimizer.lr.LRScheduler] = (None,
                                                                       None)):
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(
                "No `TrainingArguments` passed, initialized with "\
                "output_dir={} by default.".format(output_dir)
            )
            args = PromptTuningArguments(output_dir=output_dir)

        if data_collator is None:
            data_collator = InputFeatures.collate_fn

        super().__init__(model=model,
                         criterion=criterion,
                         args=args,
                         data_collator=data_collator,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         tokenizer=tokenizer,
                         compute_metrics=compute_metrics,
                         callbacks=callbacks,
                         optimizers=optimizers)
        self.load_state_dict_from_checkpoint(args.resume_from_checkpoint)

        self.train_dataset = self._map_dataset(self.train_dataset)
        self.eval_dataset = self._map_dataset(self.eval_dataset)

        if self.args.use_rdrop:
            self.rdrop_criterion = RDropLoss()

    @property
    def template(self):
        return self.model.template

    @template.setter
    def template(self, template):
        self.model.template = template

    @property
    def verbalizer(self):
        return getattr(self.model, "verbalizer", None)

    @verbalizer.setter
    def verbalizer(self, verbalizer):
        setattr(self.model, "verbalizer", verbalizer)

    @property
    def plm(self):
        return self.model.plm

    @plm.setter
    def plm(self, model):
        self.model.plm = model

    def _map_dataset(self, dataset):
        if dataset is None:
            return None
        if not isinstance(dataset, MapDataset):
            raise ValueError("Expected `MapDataset` but received {}.".format(
                type(dataset)))
        return dataset.map(self._convert_example)

    def _convert_example(self, example):
        encoded_inputs = self.template.wrap_one_example(example)
        return encoded_inputs

    def _prepare_input(self, inputs: InputFeatures):
        return inputs

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super()._save(output_dir, state_dict)
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if self.template:
            self.template.save_to(output_dir)
        if self.verbalizer:
            self.verbalizer.save_to(output_dir)

    def load_state_dict_from_checkpoint(self, resume_from_checkpoint=None):
        if resume_from_checkpoint is not None:
            self.template = AutoTemplate.load_from(
                resume_from_checkpoint, self.tokenizer,
                self.args.max_seq_length, self.plm,
                getattr(self.template, "_prompt_encoder", None),
                getattr(self.template, "encoder_hidden_size", None))

        super().load_state_dict_from_checkpoint(resume_from_checkpoint)

    def get_eval_dataloader(self, eval_dataset=None):
        """
        Return the evaluation [`~paddle.io.DataLoader`].

        Args:
            eval_dataset (`paddlenlp.datasets.MapDataset`):
                Created by `paddlenlp.prompt.load_dataset`,
                where every item is an InputExample object.
        """
        eval_dataset = self._map_dataset(eval_dataset)
        return super().get_eval_dataloader(eval_dataset)

    def get_test_dataloader(self, test_dataset):
        """
        Return the test [`~paddle.io.DataLoader`].

        Args:
            test_dataset (`paddlenlp.datasets.MapDataset`):
                The test dataset created by `paddlenlp.prompt.load_dataset`,
                where every item is an InputExample object.
        """
        test_dataset = self._map_dataset(test_dataset)
        return super().get_test_dataloader(test_dataset)

    def create_optimizer(self, lr_scheduler=None):
        """
        Setup the optimizer for both model and prompt parameters.
        """
        if self.optimizer is None:
            optim_cls, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args)

            plm_parameters = []
            if not self.model.freeze_plm:
                plm_parameters.extend([
                    p for p in self.model.plm.parameters()
                    if not p.stop_gradient
                ])

            ppt_parameters = []
            if self.template is not None:
                ppt_parameters.extend([
                    x for x in self.template.parameters() if not x.stop_gradient
                ])
            if self.verbalizer is not None:
                if isinstance(self.verbalizer, SoftVerbalizer):
                    if not self.model.freeze_plm:
                        plm_parameters.extend([
                            p for n, p in self.verbalizer.non_head_parameters()
                            if not p.stop_gradient
                        ])
                    ppt_parameters.extend(
                        [p for n, p in self.verbalizer.head_parameters()])
                else:
                    ppt_parameters.extend(
                        [p for n, p in self.verbalizer.parameters()])

            decay_parameters = [
                p.name for n, p in self.model.named_parameters()
                if not any(nd in n for nd in ["bias", "norm"])
            ]
            apply_decay_param_fun = lambda x: x in decay_parameters

            if len(plm_parameters) > 0:
                ppt_lr = self.args.ppt_learning_rate / self.args.learning_rate
                lr = self.lr_scheduler if lr_scheduler is None else lr_scheduler
                if len(ppt_parameters) > 0:
                    params = [{
                        "params": plm_parameters
                    }, {
                        "params": ppt_parameters,
                        "learning_rate": ppt_lr,
                        "weight_decay": self.args.ppt_weight_decay,
                        "beta1": self.args.ppt_adam_beta1,
                        "beta2": self.args.ppt_adam_beta2,
                        "epsilon": self.args.ppt_adam_epsilon
                    }]
                else:
                    params = plm_parameters
            else:
                args = self.init_num_steps(self.args, len(self.train_dataset))
                warmup = args.warmup_steps if args.warmup_steps > 0 else int(
                    args.warmup_ratio * args.num_training_steps)
                self.lr_scheduler = get_scheduler(
                    args.lr_scheduler_type,
                    learning_rate=self.args.ppt_learning_rate,
                    num_warmup_steps=warmup,
                    num_training_steps=args.num_training_steps,
                )
                lr = self.lr_scheduler
                params = ppt_parameters

            self.optimizer = optim_cls(
                learning_rate=lr,
                apply_decay_param_fun=apply_decay_param_fun,
                parameters=params,
                weight_decay=self.args.weight_decay,
                grad_clip=nn.ClipGradByGlobalNorm(self.args.max_grad_norm),
                **optim_kwargs)

        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the total loss for every batch. 
        """
        if "labels" not in inputs:
            raise ValueError("Fail to compute loss as there are no labels "\
                             "in {}.".format(inputs))
        labels = inputs["labels"]
        soft_token_ids = inputs.get("soft_token_ids", None)

        outputs, hidden_states = model(
            input_ids=inputs["input_ids"],
            mask_ids=inputs["mask_ids"],
            soft_token_ids=soft_token_ids,
            # inputs["labels"], # generation
            return_hidden_states=True)
        if self.criterion is not None:
            if isinstance(model.verbalizer, MultiMaskVerbalizer):
                labels = model.verbalizer.token_ids[labels].squeeze(
                    axis=-1)  #.reshape(outputs.shape[:-1])
            if labels.ndim == outputs.ndim:
                loss = 0
                for idx in range(labels.shape[-1]):
                    loss += self.criterion(
                        outputs,
                        paddle.index_select(labels,
                                            paddle.to_tensor(idx),
                                            axis=1).reshape(outputs.shape[:-1]))
            else:
                labels = labels.reshape(outputs.shape[:-1])
                loss = self.criterion(outputs, labels)

            if self.args.use_rdrop:
                loss = self._compute_rdrop_loss(model, inputs, outputs, loss)

            if self.args.use_rgl:
                loss += self._compute_rgl_loss(hidden_states, labels)

            outputs = (loss, outputs)

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def _compute_rdrop_loss(self, model, inputs, outputs, loss):
        re_outputs = model(input_ids=inputs["input_ids"],
                           mask_ids=inputs["mask_ids"],
                           soft_token_ids=inputs.get("soft_token_ids", None))
        labels = inputs["labels"]
        if isinstance(model.verbalizer, MultiMaskVerbalizer):
            labels = model.verbalizer.token_ids[labels].squeeze(
                axis=-1).reshape(re_outputs.shape[:2])
        ce_loss = (self.criterion(re_outputs, labels) + loss) * 0.5
        kl_loss = self.rdrop_criterion(outputs, re_outputs)
        loss = ce_loss + self.args.alpha_rdrop * kl_loss
        return loss

    def _compute_rgl_loss(self, embeddings, labels, equal_type="raw"):
        """
        Compute the label consistency loss of sentence embeddings per batch.
        Please refer to https://aclanthology.org/2022.findings-naacl.81/ 
        for more details.
        """

        def _max_equal(x, y):
            return int(paddle.argmax(x, axis=0) == paddle.argmax(y, axis=0))

        def _raw_equal(x, y):
            return int(x == y)

        if equal_type == "raw":
            equals = _raw_equal
        elif equal_type == "max":
            equals = _max_equal
        else:
            raise ValueError("Unsupported equal type {}.".format(equal_type))
        batch_size = embeddings.shape[0]
        loss = 0
        for i in range(batch_size):
            for j in range(batch_size):
                score = F.cosine_similarity(embeddings[i],
                                            embeddings[j],
                                            axis=0)
                score = score.unsqueeze(0)
                logits = paddle.concat([(1 - score) * 50, (1 + score) * 50],
                                       axis=-1)
                label = paddle.to_tensor([equals(labels[i], labels[j])])
                logits = logits.reshape([-1, logits.shape[-1]])
                loss += F.cross_entropy(logits, label.unsqueeze(0))
        loss = loss / (batch_size * (batch_size - 1))
        loss = loss / 100 * self.args.alpha_rgl

        return loss

    def export_model(self, export_path, input_spec=None, export_type="paddle"):
        os.makedirs(export_path, exist_ok=True)
        self.template.save_to(export_path)
        if self.verbalizer is not None:
            self.verbalizer.save_to(export_path)
        if input_spec is None and hasattr(self.model, "get_input_spec"):
            input_spec = self.model.get_input_spec()
        if input_spec is None:
            raise ValueError("Please define input_spec to export model.")
        export_model(self.model, input_spec, export_path, export_type)


class PromptModelForSequenceClassification(nn.Layer):
    """
    PromptModel for classification tasks.
    """

    def __init__(self,
                 model,
                 template,
                 verbalizer=None,
                 freeze_plm: bool = False,
                 freeze_dropout: bool = False):
        super().__init__()
        self.plm = model
        self.template = template
        self.verbalizer = verbalizer
        self.freeze_plm = freeze_plm
        self.freeze_dropout = freeze_dropout
        if self.verbalizer is not None and hasattr(verbalizer, "process_model"):
            self.plm = self.verbalizer.process_model(self.plm)
        if self.freeze_plm:
            for param in self.plm.parameters():
                param.stop_gradient = True
        if self.freeze_dropout:
            self.plm.eval()
        self.forward_keys = signature(self.plm.forward)
        self._mask_token_id = self.template.tokenizer.mask_token_id
        self._pad_token_id = self.template.tokenizer.pad_token_id

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                mask_ids=None,
                soft_token_ids=None,
                **kwargs):
        return_hidden_states = kwargs.pop('return_hidden_states', False)
        if self.freeze_dropout:
            self.plm.eval()
        attention_mask = (input_ids != self._pad_token_id).astype("int64")
        inputs = InputFeatures(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               mask_ids=mask_ids,
                               attention_mask=attention_mask,
                               soft_token_ids=soft_token_ids)
        if hasattr(self.template, "process_batch"):
            inputs = self.template.process_batch(inputs)
        model_inputs = {
            k: inputs[k]
            for k in inputs.keys(keep_none=True) if k in self.forward_keys
        }
        outputs = self.plm(**model_inputs)
        hidden_states = outputs
        if hasattr(self.template, "post_process_batch"):
            outputs = self.template.post_process_batch(outputs)
        if self.verbalizer and hasattr(self.verbalizer, "process_outputs"):
            outputs = self.verbalizer.process_outputs(outputs, inputs=inputs)

        if return_hidden_states:
            return outputs, hidden_states
        else:
            return outputs

    def prompt_parameters(self):
        """
        Get the parameters of template and verbalizer.
        """
        params = [p for p in self.template.parameters()]
        if self.verbalizer:
            params += [p for p in self.verbalizer.parameters()]
        return params

    def get_input_spec(self):
        input_spec = [
            InputSpec(shape=[None, None], dtype="int64"),  # input_ids
            InputSpec(shape=[None, None], dtype="int64"),  # mask_ids
            InputSpec(shape=[None, None], dtype="int64")  # soft_token_ids
        ]
        return input_spec


class PromptModelForGeneration(nn.Layer):
    """
    PromptModel for generation tasks.
    """

    def __init__(self,
                 model,
                 template,
                 max_seq_length=512,
                 freeze_plm=False,
                 freeze_dropout=False):
        super().__init__()
        self.plm = model
        self.template = template
        self.tokenizer = self.template.tokenizer
        self.max_seq_length = max_seq_length
        self.freeze_plm = freeze_plm
        self.freeze_dropout = freeze_dropout
        if self.freeze_plm:
            for param in self.plm.parameters():
                param.stop_gradient = True
        if self.freeze_dropout:
            self.plm.eval()
        self.forward_keys = signature(self.plm.forward)
        self._mask_token_id = self.template.tokenizer.mask_token_id
        self._pad_token_id = self.template.tokenizer.pad_token_id

    def forward(self,
                input_ids,
                mask_ids=None,
                soft_token_ids=None,
                labels=None,
                **kwargs):
        return_hidden_states = kwargs.pop('return_hidden_states', False)
        if self.freeze_dropout:
            self.plm.eval()
        attention_mask = (input_ids != self._pad_token_id).astype("int64")
        inputs = InputFeatures(input_ids=input_ids,
                               attention_mask=attention_mask,
                               soft_token_ids=soft_token_ids)
        if hasattr(self.template, "process_batch"):
            inputs = self.template.process_batch(inputs)
        model_inputs = {
            k: inputs[k]
            for k in inputs.keys(keep_none=True) if k in self.forward_keys
        }

        outputs = self.plm(input_ids=inputs.input_ids,
                           attention_mask=inputs.attention_mask,
                           labels=labels)[0]
        if return_hidden_states:
            return outputs, None
        return outputs
