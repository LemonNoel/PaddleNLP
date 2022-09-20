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
import argparse

import numpy as np

from paddlenlp.utils.log import logger

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--src_file", type=str, help="Path to the original examples.")
parser.add_argument("--trg_file", default="./train.txt", type=str, help="Path to the standard examples.")
parser.add_argument("--seed", default=42, type=int, help="Random seed for sampling.")
args = parser.parse_args()
# yapf: enable


def do_convert():
    tic_time = time.time()
    if not os.path.exists(args.src_file):
        raise ValueError(
            f"Please input a correct path, {args.src_file} does not exist.")

    def _save_to_file(examples):
        with open(args.trg_file, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(example + "\n")
        logger.info(f"Save {len(examples)} examples to {args.trg_file}.")

    with open(args.src_file, "r", encoding="utf-8") as f:
        data = [json.loads(x.strip()) for x in f.readlines()]

    new_examples = []
    for line in data:
        text = line["sentence"]
        if "label_desc" in line:
            text = text + "\t" + line["label_desc"]
        new_examples.append(text)
    _save_to_file(new_examples)


if __name__ == "__main__":
    do_convert()
