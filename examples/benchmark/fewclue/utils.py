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

import json
from functools import partial

import numpy as np

from paddlenlp.datasets import load_dataset, MapDataset
from paddlenlp.prompt import InputExample
from paddlenlp.utils.log import logger
from paddlenlp.dataaug import WordSubstitute, WordInsert, WordDelete, WordSwap


def convert_eprstmt(example):
    # Unlabeled: 19565
    return InputExample(uid=example["id"],
                        text_a=example["sentence"],
                        text_b="",
                        labels=example.get("label", None))


def convert_csldcp(example):
    # Unlabeled: 67
    return InputExample(uid=example["id"],
                        text_a=example["content"],
                        text_b="",
                        labels=example.get("label", None))


def convert_multi_efl(example,
                      label_set,
                      num_neg=None,
                      text_key="content",
                      label_key="label"):
    true_label = example.get(label_key, None)
    example_list = []
    if true_label is None:
        for label in label_set:
            example_list.append(
                InputExample(uid=example["id"],
                             text_a=example[text_key],
                             text_b=label,
                             labels=None))
    else:
        neg_set = list(set(label_set) - set([true_label]))
        example_list.append(
            InputExample(uid=example["id"],
                         text_a=example[text_key],
                         text_b=true_label,
                         labels=1))
        if num_neg is not None:
            neg_set = np.random.permutation(neg_set).tolist()[:num_neg]
        for neg_label in neg_set:
            example_list.append(
                InputExample(uid=example["id"],
                             text_a=example[text_key],
                             text_b=neg_label,
                             labels=0))
    return example_list


def convert_tnews(example):
    return InputExample(uid=example["id"],
                        text_a=example["sentence"],
                        text_b="",
                        labels=example.get("label_desc", None))


def convert_iflytek(example):
    return InputExample(uid=example["id"],
                        text_a=example["sentence"],
                        text_b="",
                        labels=example.get("label_des", None))


def convert_ocnli(example):
    # Unlabeled: 20000
    # IDEA A: Use multi-task learning.
    #         Train genre classificaiton seperately.
    return InputExample(uid=example.get("id", None),
                        text_a=example["sentence1"],
                        text_b=example["sentence2"],
                        labels=example.get("label", None))


def convert_bustm(example):
    # Unlabeled: 4251
    return InputExample(uid=example["id"],
                        text_a=example["sentence1"],
                        text_b=example["sentence2"],
                        labels=example.get("label", None))


def convert_chid(example):
    # Unlabeled: 7585
    # IDEA B.1: Take it as a binary classification.
    #           Replace #idiom# with candicates.
    # IDEA B.2: Take it as a token classification.
    #           Concatenate all sequences.
    return InputExample(uid=example["id"],
                        text_a=example["content"],
                        text_b="，".join(example["candidates"]),
                        labels=example.get("answer", None))


def convert_chid_efl(example):
    # IDEA B.1
    bi_examples = []
    fragments = example["content"].split("#idiom#")
    label = example.get("answer", None)
    for idx, cand in enumerate(example["candidates"]):
        #text = fragments[0] + "[" + cand + "]" + fragments[1] < 1012
        text = fragments[0] + "（" + cand + "）" + fragments[1]
        bi_examples.append(
            InputExample(uid=example["id"],
                         text_a=text,
                         text_b=cand,
                         labels=None if label is None else int(idx == label)))
    return bi_examples


def convert_csl(example):
    # Unlabeled: 19841. Long sentence.
    # IDEA C: Take it as a NER and compare list. Con: error propagation.
    return InputExample(uid=example["id"],
                        text_a=example["abst"],
                        text_b="，".join(example["keyword"]),
                        labels=example.get("label", None))


def convert_csl_efl(example):
    bi_examples = []
    text = example["abst"]
    label = example.get("label", None)
    for keyword in example["keyword"]:
        bi_examples.append(
            InputExample(uid=example["id"],
                         text_a=text,
                         text_b=keyword,
                         labels=int(label) if label is not None else None))
    return bi_examples


def D0_convert_cluewsc(example):
    # IDEA D.1: Use attention between two positions.
    # IDEA D.2: Take it as binary classification. Replace span2 with span1.
    # IDEA D.3: Use special tokens to mark query and pronoun.
    return InputExample(uid=example.get("id", None),
                        text_a=example["text"],
                        text_b="其中" + example["target"]["span2_text"] + "指的是" +
                        example["target"]["span1_text"],
                        labels=example.get("label", None))


def D2_convert_cluewsc(example):
    # IDEA D.2
    target = example["target"]
    text = example["text"][:target["span2_index"]] + "（" + target["span1_text"] + \
        "）" + example["text"][target["span2_index"] + len(target["span2_text"]):]
    return InputExample(uid=example.get("id", None),
                        text_a=text,
                        text_b="",
                        labels=example.get("label", None))


def D3_convert_cluewsc(example):
    # IDEA D.3
    target, text = example["target"], list(example["text"])
    pronoun, p_index = target["span2_text"], target["span2_index"]
    entity, e_index = target["span1_text"], target["span1_index"]
    if p_index > e_index:
        text.insert(p_index, "_")
        text.insert(p_index + len(pronoun) + 1, "_")
        text.insert(e_index, "[")
        text.insert(e_index + len(entity) + 1, "]")
    else:
        text.insert(e_index, "[")
        text.insert(e_index + len(entity) + 1, "]")
        text.insert(p_index, "_")
        text.insert(p_index + len(pronoun) + 1, "_")
    return InputExample(uid=example.get("id", None),
                        text_a="“" + "".join(text) + "”这段小说节选中_" + pronoun +
                        "_指的",
                        text_b="是[" + entity + "]",
                        labels=example.get("label", None))


def convert_cluewsc(example):
    # IDEA D.3
    target, text = example["target"], list(example["text"])
    pronoun, p_index = target["span2_text"], target["span2_index"]
    entity, e_index = target["span1_text"], target["span1_index"]
    if p_index > e_index:
        text.insert(p_index, "_")
        text.insert(p_index + len(pronoun) + 1, "_")
        text.insert(e_index, "[")
        text.insert(e_index + len(entity) + 1, "]")
    else:
        text.insert(e_index, "[")
        text.insert(e_index + len(entity) + 1, "]")
        text.insert(p_index, "_")
        text.insert(p_index + len(pronoun) + 1, "_")
    return InputExample(
        uid=example.get("id", None),
        #text_a="".join(text) + "上文中(" + pronoun + ")是指",
        #text_b="[" + entity + "]",
        text_a="".join(text),
        text_b="其中_" + pronoun + "_指的是[" + entity + "]",
        #text_a="[" + entity + "]" + "".join(text),
        #text_b="下边这段话中_" + pronoun + "_可以替换成",
        # text_b=pronoun + "指的是" + entity, # 1011-1012
        labels=example.get("label", None))


def convert_cluewsc_demons(examples, demons=None, label_dict=None):

    def add_special_token(example):
        target, text = example["target"], list(example["text"])
        pronoun, p_index = target["span2_text"], target["span2_index"]
        entity, e_index = target["span1_text"], target["span1_index"]
        if p_index > e_index:
            text.insert(p_index, "_")
            text.insert(p_index + len(pronoun) + 1, "_")
            text.insert(e_index, "[")
            text.insert(e_index + len(entity) + 1, "]")
        else:
            text.insert(e_index, "[")
            text.insert(e_index + len(entity) + 1, "]")
            text.insert(p_index, "_")
            text.insert(p_index + len(pronoun) + 1, "_")
        text = "".join(text)
        return text

    def create_demons(example):
        target, text = example["target"], list(example["text"])
        pronoun, p_index = target["span2_text"], target["span2_index"]
        entity, e_index = target["span1_text"], target["span1_index"]
        text = add_special_token(example)
        if example["label"] == "true":
            mask = "确实"
        else:
            mask = "不像"
        text += ("其中_" + pronoun + "_指的" + mask + "是[" + entity + "]")
        return text

    if demons is None:
        demons = [create_demons(x) for x in examples]
    new_examples = []
    for idx, example in enumerate(examples):
        target = example["target"]
        pronoun, entity = target["span2_text"], target["span1_text"]
        demon_idx = np.random.randint(0, len(demons))
        while demon_idx == idx:
            demon_idx = np.random.randint(0, len(demons))
        new_examples.append(
            InputExample(uid=example.get("id", None),
                         text_a=demons[demon_idx] + add_special_token(example) +
                         "其中_" + pronoun + "_指的",
                         text_b="是[" + entity + "]",
                         labels=None if "label" not in example else
                         label_dict[example.get("label", None)]))
    return MapDataset(new_examples), demons


def convert_labels_to_ids(example, label_dict):
    if example.labels is not None:
        example.labels = label_dict[example.labels]
    return example


def data_augment(data_ds, aug_type="delete", num_aug=10, percent=0.1):
    if aug_type == "delete":
        aug = WordDelete(create_n=num_aug, aug_percent=percent)
    elif aug_type == "substitute":
        aug = WordSubstitute("mlm", create_n=num_aug, aug_percent=percent)
    elif aug_type == "insert":
        aug = WordInsert("mlm", create_n=num_aug, aug_percent=percent)
    elif aug_type == "swap":
        aug = WordSwap(create_n=num_aug, aug_percent=percent)

    new_data_ds = []
    for example in data_ds:
        new_data_ds.append(example)
        text_a = aug.augment(example.text_a)
        if example.text_b is None:
            for text in text_a:
                example.text_a = text
                new_data_ds.append(example)
        else:
            text_b = aug.augment(example.text_b)
            for ta, tb in zip(text_a, text_b):
                example.text_a = ta
                example.text_b = tb
                new_data_ds.append(example)
    return MapDataset(new_data_ds)


def extend_with_fakes(data_ds, fake_file=None):
    # 将伪标签数据合入训练集
    # Args:
    # - data_ds (list or MapDataset): 原始训练数据
    # - fake_file (str): 与 FewCLUE 格式相同的伪标签数据
    if fake_file is not None:
        data_ds = [x for x in data_ds]
        with open(fake_file, "r") as fp:
            fake_data = [json.loads(x) for x in fp.readlines()]
            data_ds.extend(fake_data)
        data_ds = MapDataset(data_ds)
    return data_ds


# 读取 FewCLUE 数据集
def load_fewclue(task_name,
                 split_id,
                 label_list,
                 fake_file=None,
                 aug_type=None):
    if task_name == "tnews":
        splits = [f"dev_{split_id}", "test_public", "test", "unlabeled"]
        dev_ds, public_test_ds, test_ds, unlabeled_ds = load_dataset(
            "fewclue", name=task_name, splits=splits, label_list=label_list)
        with open("data/tnews_train.json", "r") as fp:
            data = [x for x in fp.readlines() if x[0] != "#"]
            train_ds = MapDataset([json.loads(x.strip()) for x in data])
    elif task_name == "cluewsc":
        splits = [f"train_{split_id}", f"dev_{split_id}", "test_public", "test"]
        train_ds, dev_ds, public_test_ds, test_ds = load_dataset(
            "fewclue", name=task_name, splits=splits, label_list=label_list)
        unlabeled_ds = None
    elif task_name == "cmnli":
        train_ds, dev_ds = load_dataset("clue",
                                        name="cmnli",
                                        splits=["train", "dev"])
        public_test_ds = None
        test_ds = None
        unlabeled_ds = None
    else:
        # Load FewCLUE datasets and convert the samples to InputExample.
        splits = [
            f"train_{split_id}", f"dev_{split_id}", "test_public", "test",
            "unlabeled"
        ]
        train_ds, dev_ds, public_test_ds, test_ds, unlabeled_ds = load_dataset(
            "fewclue", name=task_name, splits=splits, label_list=label_list)

    train_ds = extend_with_fakes(train_ds, fake_file)

    def convert_to_binary(dataset, convert_efl):
        new_data = []
        for example in dataset:
            new_data.extend(convert_efl(example))
        return MapDataset(new_data)

    if task_name == "chid":
        # IDEA B.1
        train_ds = convert_to_binary(train_ds, convert_chid_efl)
        dev_ds = convert_to_binary(dev_ds, convert_chid_efl)
        public_test_ds = convert_to_binary(public_test_ds, convert_chid_efl)
        test_ds = convert_to_binary(test_ds, convert_chid_efl)
        unlabeled_ds = convert_to_binary(unlabeled_ds, convert_chid_efl)
    elif task_name == "_cluewsc":
        train_ds, demons = convert_cluewsc_demons(train_ds, None, label_list)
        dev_ds, _ = convert_cluewsc_demons(dev_ds, demons, label_list)
        public_test_ds, _ = convert_cluewsc_demons(public_test_ds, demons,
                                                   label_list)
        test_ds, _ = convert_cluewsc_demons(test_ds, demons, label_list)
    elif task_name == "_csldcp":
        label_set = set([x for x in label_list.keys()])
        convert_efl_train = partial(convert_multi_efl,
                                    label_set=label_set,
                                    num_neg=5,
                                    text_key="content",
                                    label_key="label")
        convert_efl_test = partial(convert_multi_efl,
                                   label_set=label_set,
                                   text_key="content",
                                   label_key="label")
        train_ds = convert_to_binary(train_ds, convert_efl_train)
        dev_ds = convert_to_binary(dev_ds, convert_efl_test)
        public_test_ds = convert_to_binary(public_test_ds, convert_efl_test)
        test_ds = convert_to_binary(test_ds, convert_efl_test)
        unlabeled_ds = convert_to_binary(unlabeled_ds, convert_efl_test)
    elif task_name == "_iflytek":
        label_set = set([x for x in label_list.keys()])
        convert_efl_train = partial(convert_multi_efl,
                                    label_set=label_set,
                                    num_neg=10,
                                    text_key="sentence",
                                    label_key="label_des")
        convert_efl_test = partial(convert_multi_efl,
                                   label_set=label_set,
                                   text_key="sentence",
                                   label_key="label_des")
        train_ds = convert_to_binary(train_ds, convert_efl_train)
        dev_ds = convert_to_binary(dev_ds, convert_efl_test)
        public_test_ds = convert_to_binary(public_test_ds, convert_efl_test)
        test_ds = convert_to_binary(test_ds, convert_efl_test)
        unlabeled_ds = convert_to_binary(unlabeled_ds, convert_efl_test)
    else:
        convert_fn = {
            "eprstmt": convert_eprstmt,
            "csldcp": convert_csldcp,
            "tnews": convert_tnews,
            "iflytek": convert_iflytek,
            "ocnli": convert_ocnli,
            "cmnli": convert_ocnli,
            "bustm": convert_bustm,
            "chid": convert_chid,
            "csl": convert_csl,
            "cluewsc": convert_cluewsc
        }[task_name]

        train_ds = train_ds.map(convert_fn)
        dev_ds = dev_ds.map(convert_fn)
        if task_name != "cmnli":
            public_test_ds = public_test_ds.map(convert_fn)
            test_ds = test_ds.map(convert_fn)
            if unlabeled_ds is not None:
                unlabeled_ds = unlabeled_ds.map(convert_fn)

        convert_fn = partial(convert_labels_to_ids, label_dict=label_list)
        if task_name != "cmnli":
            train_ds = train_ds.map(convert_fn)
            dev_ds = dev_ds.map(convert_fn)
            public_test_ds = public_test_ds.map(convert_fn)

    if aug_type is not None:
        train_ds = data_augment(train_ds, aug_type=aug_type)

    return train_ds, dev_ds, public_test_ds, test_ds, unlabeled_ds
