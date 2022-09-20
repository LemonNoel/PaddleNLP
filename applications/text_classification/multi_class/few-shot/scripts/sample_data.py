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
from collections import defaultdict

import numpy as np

from paddlenlp.utils.log import logger

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--src_file", type=str, help="Path to the original examples.")
parser.add_argument("--trg_file", default="./few_shot.txt", type=str, help="Path to the sampled examples.")
parser.add_argument("--task_type", default="multi-class", choices=["multi-class", "multi_label", "hierachical"], type=str, help="The classification task type.")
parser.add_argument("--sample_type", default="per_label", choices=["total", "per_label"], type=str, help="How to sample examples.")
parser.add_argument("--num_sample_total", default=32, type=int, help="Number of total sampled examples.")
parser.add_argument("--num_sample_per_label", default=16, type=int, help="Number of sampled examples per label.")
parser.add_argument("--seed", default=42, type=int, help="Random seed for sampling.")
args = parser.parse_args()
# yapf: enable


def do_sample():
    tic_time = time.time()
    if not os.path.exists(args.src_file):
        raise ValueError(
            f"Please input a correct path, {args.src_file} does not exist.")

    def _save_to_file(examples):
        with open(args.trg_file, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(example + "\n")
        logger.info(f"Save {len(examples)} examples to {args.trg_file}.")

    def _sample(examples, num_sample, rng):
        indices = np.arange(len(examples))
        rng.shuffle(indices)
        new_examples = [examples[i] for i in indices[:num_sample]]
        return new_examples

    with open(args.src_file, "r", encoding="utf-8") as f:
        data = [x.strip() for x in f.readlines()]

    rng = np.random.RandomState(args.seed)
    if args.sample_type == "total":
        new_data = _sample(data, args.num_sample_total, rng)
        _save_to_file(new_data)
    else:
        if len(data) > 0 and len(data[0].split("\t")) <= 1:
            raise RuntimeError(
                f"There is no labels in {args.src_file}, please use `sample_type`=`total`."
            )
        label_dict = defaultdict(list)
        if args.task_type == "multi-class":
            for line in data:
                label = line.split("\t")[1]
                label_dict[label].append(line)
        else:
            for line in data:
                labels = line.split("\t")[1].split(",")
                for label in labels:
                    label_dict[label].append(line)

        new_data = []
        for examples in label_dict.values():
            new_data.extend(_sample(examples, args.num_sample_per_label, rng))
        new_data = list(set(new_data))
        _save_to_file(new_data)


if __name__ == "__main__":
    do_sample()
