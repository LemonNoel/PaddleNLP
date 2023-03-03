# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
"""GLM model configuration"""

from __future__ import annotations

from ..configuration_utils import PretrainedConfig

__all__ = [
    "GLM130BConfig",
    "GLM130B_PRETRAINED_INIT_CONFIGURATION",
    "GLM130B_PRETRAINED_RESOURCE_FILES_MAP",
]


GLM130B_PRETRAINED_INIT_CONFIGURATION = {
    "glm-130b": {
        "hidden_size": 12288,
        "mlp_hidden_size": 12288 * 8 / 3,
        "num_layers": 70,
        "num_heads": 96,
        "length_per_sample": 2000,
        "max_length": 2048,
        "vocab_size_base": 768,
        "activation": "geglu",
        "layernorm-epsilon": 1e-5,
        "paddle_dtype": "bfloat16",
        "attention_dropout_prob": 0.1,
        "attention_scale": True,
        "embedding_dropout_prob": 0.1,
        "initializer_range": 0.0052,
        "output_dropout_prob": 0.1,
        "output_predict": True,
        "vocab_size": 50304,
    }
}

GLM130B_PRETRAINED_RESOURCE_FILES_MAP = {"model_state": {"glm-130b": None}}


class GLM130BConfig(PretrainedConfig):
    model_type = "glm-130b"
    pretrained_init_configuration = GLM130B_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        activation="geglu",
        attention_dropout_prob=0.1,
        attention_scale=1.0,
        embedding_dropout_prob=0.1,
        hidden_size=12288,
        initializer_range=0.0052,
        length_per_sample=2000,
        max_length=2048,
        mlp_hidden_size=12288 * 8 // 3,
        num_heads=96,
        num_layers=70,
        output_dropout_prob=0.1,
        output_predict=True,
        vocab_size=50304,
        vocab_size_base=768,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation = activation
        self.attention_dropout_prob = attention_dropout_prob
        self.attention_scale = attention_scale
        self.embedding_dropout_prob = embedding_dropout_prob
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.length_per_sample = length_per_sample
        self.max_length = max_length
        self.mlp_hidden_size = mlp_hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_dropout_prob = output_dropout_prob
        self.output_predict = output_predict
        self.vocab_size = vocab_size
        self.vocab_size_base = vocab_size_base
