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
import argparse

import numpy as np

from paddlenlp.utils.log import logger

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--src_path", type=str, help="Path to data in standard formation.")
parser.add_argument("--trg_path", default="./bidata", type=str, help="Path to data for binary classification.")
parser.add_argument("--task_type", default="multi-class", choices=["multi-class", "mulit-label", "hierachical"], type=str, help="Select the classification type.")
parser.add_argument("--negative_ratio", type=int, default=1, help="Number of negative examples per positive example.")
parser.add_argument("--label_words", default=None, type=list, help="The words list for binary labels.")
parser.add_argument("--seed", default=1000, type=int, help="Random seed for negative sampling.")
args = parser.parse_args()
# yapf: enable


def do_convert():
    tic_time = time.time()
    if not os.path.exists(args.src_path):
        raise ValueError(
            f"Please input a correct data path, {args.src_path} is invalid.")
    os.makedirs(args.trg_path, exist_ok=True)

    def _check_file(file_name, must_exist=False):
        if not os.path.exists(file_name) and must_exist:
            raise ValueError(f"{file_name} is missing, please check it.")
        return os.path.exists(file_name)

    def _save_to_file(data_dir, file_name, instances):
        save_path = os.path.join(data_dir, file_name)
        with open(save_path, "w", encoding="utf-8") as f:
            for instance in instances:
                f.write(instance + "\n")
        logger.info(f"Save {len(instances)} instances to {save_path}.")

    # Load original labels.
    label_file = os.path.join(args.src_path, "label.txt")
    _check_file(label_file, must_exist=True)
    with open(label_file, "r", encoding="utf-8") as f:
        data = [x.strip() for x in f.readlines()]
        label_map = dict([x.split("==") for x in data])
        label_list = [x.strip().split("==")[0] for x in data]
    _save_to_file(args.trg_path, "true_label.txt", label_list)

    def _add_prompt(label):
        global label_map
        return "这则新闻和" + label_map[label] + "相关。"

    # Save new binary labels.
    new_labels = ["0", "1"]
    if args.label_words is not None:
        try:
            assert len(args.label_words) == 2
        except AssertionError:
            raise ValueError(
                f"Expected 2 words for new binary classification, " +
                f"but get {len(args.label_words)} words.")
        for idx, word in enumerate(args.label_words):
            new_labels[idx] = new_labels[idx] + "==" + word
    _save_to_file(args.trg_path, "label.txt", new_labels)

    # Load original datasets and save new datasets.
    dataset_name = ["train.txt", "dev.txt", "test.txt", "data.txt"]
    rng = np.random.RandomState(args.seed)
    for name in dataset_name:
        data_file = os.path.join(args.src_path, name)
        if _check_file(data_file, must_exist=False):
            with open(data_file, "r", encoding="utf-8") as f:
                data = [x.strip().split("\t") for x in f.readlines()]
            new_subset = []
            if len(data) > 0 and len(data[0]) == 1:
                for text in data:
                    text = text[0]
                    for label in label_list:
                        new_subset.append(text + "\t" + _add_prompt(label))
            elif args.task_type == "multi-class":
                for text, true_label in data:
                    neg_subset = []
                    for label in label_list:
                        if label == true_label:
                            new_subset.append(text + "\t" + _add_prompt(label) +
                                              "\t1")
                        else:
                            if name == "train.txt":
                                neg_subset.append(text + "\t" +
                                                  _add_prompt(label) + "\t0")
                            else:
                                new_subset.append(text + "\t" +
                                                  _add_prompt(label) + "\t0")
                    if name == "train.txt":
                        rng.shuffle(neg_subset)
                        new_subset.extend(neg_subset[:args.negative_ratio])
            else:
                for text, true_label in data:
                    true_label = true_label.split(",")
                    neg_subset = []
                    for label in label_list:
                        if label in true_label:
                            new_subset.append(text + "\t" + _add_prompt(label) +
                                              "\t1")
                        else:
                            if name == "train.txt":
                                neg_subset.append(text + "\t" +
                                                  _add_prompt(label) + "\t0")
                            else:
                                new_subset.append(text + "\t" +
                                                  _add_prompt(label) + "\t0")
                    if name == "train.txt":
                        rng.shuffle(neg_subset)
                        new_subset.extend(neg_subset[:args.negative_ratio])
            _save_to_file(args.trg_path, name, new_subset)
    logger.info(f"It takes {time.time() - tic_time} seconds to convert data.")


if __name__ == "__main__":
    do_convert()
