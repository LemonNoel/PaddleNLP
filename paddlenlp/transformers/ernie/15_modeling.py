# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
from paddle.distributed.fleet.utils import recompute
from paddle.nn.layer.transformer import _convert_attention_mask

from paddlenlp.transformers import PretrainedModel, register_base_model

__all__ = [
    'ErnieModel', 'ErniePretrainedModel', 'ErnieForSequenceClassification',
    'ErnieForTokenClassification', 'ErnieForQuestionAnswering',
    'ErnieForPretraining', 'ErniePretrainingCriterion', 'ErnieForMaskedLM',
    'ErnieForMultipleChoice'
]


class TransformerEncoder(nn.Layer):
    """
    TransformerEncoder is a stack of N encoder layers. 
    Parameters:
        encoder_layer (Layer): an instance of the `TransformerEncoderLayer`. It
            would be used as the first layer, and the other layers would be created
            according to the configurations of it.
        num_layers (int): The number of encoder layers to be stacked.
        norm (LayerNorm, optional): the layer normalization component. If provided,
            apply layer normalization on the output of last encoder layer.
    Examples:
        .. code-block:: python
            import paddle
            from paddle.nn import TransformerEncoderLayer, TransformerEncoder
            # encoder input: [batch_size, src_len, d_model]
            enc_input = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, n_head, src_len, src_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            encoder_layer = TransformerEncoderLayer(128, 2, 512)
            encoder = TransformerEncoder(encoder_layer, 2)
            enc_output = encoder(enc_input, attn_mask)  # [2, 4, 128]
    """

    def __init__(self,
                 encoder_layer,
                 num_layers,
                 norm=None,
                 enable_recompute=False,
                 preserve_rng_state=False):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.LayerList([
            (encoder_layer if i == 0 else type(encoder_layer)(
                **encoder_layer._config)) for i in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm
        # NOTE recompute modification
        if enable_recompute:
            print("using recompute.")
        self.enable_recompute = enable_recompute
        self.preserve_rng_state = preserve_rng_state
        if preserve_rng_state:
            assert self.enable_recompute, "preserve_rng_state is True, but enable_recompute is False."

    def forward(self, src, src_mask=None, cache=None):
        r"""
        Applies a stack of N Transformer encoder layers on inputs. If `norm` is
        provided, also applies layer normalization on the output of last encoder
        layer.
        Parameters:
            src (Tensor): The input of Transformer encoder. It is a tensor
                with shape `[batch_size, sequence_length, d_model]`. The data
                type should be float32 or float64.
            src_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False` 
                values and the others have `True` values. When the data type is 
                int, the unwanted positions have 0 values and the others have 1 
                values. When the data type is float, the unwanted positions have 
                `-INF` values and the others have 0 values. It can be None when 
                nothing wanted or needed to be prevented attention to. Default None.
            cache (list, optional): It is a list, and each element in the list
                is `incremental_cache` produced by `TransformerEncoderLayer.gen_cache`. 
                See `TransformerEncoder.gen_cache` for more details. It is only
                used for inference and should be None for training. Default None.
        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
                as `src`, representing the output of Transformer encoder. \
                Or a tuple if `cache` is not None, except for encoder output, \
                the tuple includes the new cache which is same as input `cache` \
                argument but `incremental_cache` in it has an incremental length. \
                See `MultiHeadAttention.gen_cache` and `MultiHeadAttention.forward` \
                for more details.
        """
        src_mask = _convert_attention_mask(src_mask, src.dtype)

        output = src
        new_caches = []
        for i, mod in enumerate(self.layers):
            if cache is None:
                # NOTE recompute modification
                if self.enable_recompute:
                    output = recompute(
                        mod,
                        output,
                        src_mask,
                        preserve_rng_state=self.preserve_rng_state)
                else:
                    output = mod(output, src_mask=src_mask)
            else:
                output, new_cache = mod(output,
                                        src_mask=src_mask,
                                        cache=cache[i])
                new_caches.append(new_cache)

        if self.norm is not None:
            output = self.norm(output)

        return output if cache is None else (output, new_caches)

    def gen_cache(self, src):
        r"""
        Generates cache for `forward` usage. The generated cache is a list, and
        each element in it is `incremental_cache` produced by 
        `TransformerEncoderLayer.gen_cache`. See `TransformerEncoderLayer.gen_cache`
        for more details.
        Parameters:
            src (Tensor): The input of Transformer encoder. It is a tensor
                with shape `[batch_size, source_length, d_model]`. The data type
                should be float32 or float64.
        Returns:
            list: It is a list, and each element in the list is `incremental_cache` 
            produced by `TransformerEncoderLayer.gen_cache`. See 
            `TransformerEncoderLayer.gen_cache` for more details.
        """
        cache = [layer.gen_cache(src) for layer in self.layers]
        return cache


class ErnieEmbeddings(nn.Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 pad_token_id=0):
        super(ErnieEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size,
                                            hidden_size,
                                            padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None):
        if input_ids is not None:
            input_shape = paddle.shape(input_ids)
            input_embeddings = self.word_embeddings(input_ids)
        else:
            input_shape = paddle.shape(inputs_embeds)[:-1]
            input_embeddings = inputs_embeds

        if position_ids is None:
            # maybe need use shape op to unify static graph and dynamic graph
            #seq_length = input_ids.shape[1]
            ones = paddle.ones(input_shape, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=1)
            position_ids = seq_length - ones
            position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype="int64")
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embeddings + position_embeddings + token_type_embeddings
        # embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ErniePooler(nn.Layer):
    """
    """

    def __init__(self, hidden_size):
        super(ErniePooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ErniePretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained ERNIE models. It provides ERNIE related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models. See `PretrainedModel` for more details.
    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "ernie-1.0": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "max_position_embeddings": 513,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 2,
            "vocab_size": 18000,
            "pad_token_id": 0,
        },
        "ernie-tiny": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "max_position_embeddings": 600,
            "num_attention_heads": 16,
            "num_hidden_layers": 3,
            "type_vocab_size": 2,
            "vocab_size": 50006,
            "pad_token_id": 0,
        },
        "ernie-2.0-en": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 4,
            "vocab_size": 30522,
            "pad_token_id": 0,
        },
        "ernie-2.0-large-en": {
            "attention_probs_dropout_prob": 0.1,
            "intermediate_size": 4096,  # special for ernie-2.0-large-en
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "type_vocab_size": 4,
            "vocab_size": 30522,
            "pad_token_id": 0,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "ernie-1.0":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie/ernie_v1_chn_base.pdparams",
            "ernie-tiny":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_tiny/ernie_tiny.pdparams",
            "ernie-2.0-en":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_v2_base/ernie_v2_eng_base.pdparams",
            "ernie-2.0-large-en":
            "https://paddlenlp.bj.bcebos.com/models/transformers/ernie_v2_large/ernie_v2_eng_large.pdparams",
        }
    }
    base_model_prefix = "ernie"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # only support dygraph, use truncated_normal and make it inplace
            # and configurable later
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(mean=0.0,
                                         std=self.initializer_range if hasattr(
                                             self, "initializer_range") else
                                         self.ernie.config["initializer_range"],
                                         shape=layer.weight.shape))


@register_base_model
class ErnieModel(ErniePretrainedModel):
    """
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 pad_token_id=0):
        super(ErnieModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = ErnieEmbeddings(vocab_size, hidden_size,
                                          hidden_dropout_prob,
                                          max_position_embeddings,
                                          type_vocab_size, pad_token_id)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0,
            normalize_before=True)
        self.encoder = TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = ErniePooler(hidden_size)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                inputs_embeds=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time."
            )
        elif input_ids is not None:
            input_shape = paddle.shape(input_ids)
        elif inputs_embeds is not None:
            input_shape = paddle.shape(inputs_embeds)[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id).astype(
                    self.pooler.dense.weight.dtype) * -1e9,
                axis=[1, 2])
        elif attention_mask.ndim == 2:
            attention_mask = paddle.unsqueeze(
                attention_mask, axis=[1, 2]).astype(paddle.get_default_dtype())
            attention_mask = (1.0 - attention_mask) * -1e9
        embedding_output = self.embeddings(input_ids=input_ids,
                                           position_ids=position_ids,
                                           token_type_ids=token_type_ids,
                                           inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        sequence_output = encoder_outputs
        sequence_output = self.embeddings.layer_norm(encoder_outputs)
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output


class ErnieForSequenceClassification(ErniePretrainedModel):
    """
    Model for sentence (pair) classification task with ERNIE.
    Args:
        ernie (ErnieModel): An instance of `ErnieModel`.
        num_classes (int, optional): The number of classes. Default 2
        dropout (float, optional): The dropout probability for output of ERNIE.
            If None, use the same value as `hidden_dropout_prob` of `ErnieModel`
            instance `Ernie`. Default None
    """

    def __init__(self, ernie, num_classes=2, dropout=None):
        super(ErnieForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.ernie = ernie  # allow ernie to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else self.
                                  ernie.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                inputs_embeds=None):
        _, pooled_output = self.ernie(input_ids,
                                      token_type_ids=token_type_ids,
                                      position_ids=position_ids,
                                      attention_mask=attention_mask,
                                      inputs_embeds=inputs_embeds)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class ErnieForQuestionAnswering(ErniePretrainedModel):

    def __init__(self, ernie):
        super(ErnieForQuestionAnswering, self).__init__()
        self.ernie = ernie  # allow ernie to be config
        self.classifier = nn.Linear(self.ernie.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        sequence_output, _ = self.ernie(input_ids,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        attention_mask=attention_mask)

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        return start_logits, end_logits


class ErnieForTokenClassification(ErniePretrainedModel):

    def __init__(self, ernie, num_classes=2, dropout=None):
        super(ErnieForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.ernie = ernie  # allow ernie to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else self.
                                  ernie.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        sequence_output, _ = self.ernie(input_ids,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        attention_mask=attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


class ErnieLMPredictionHead(nn.Layer):

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super(ErnieLMPredictionHead, self).__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.activation = getattr(nn.functional, activation)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder_weight = self.create_parameter(
            shape=[hidden_size, vocab_size],
            dtype=self.transform.weight.dtype,
            is_bias=True) if embedding_weights is None else embedding_weights
        self.decoder_bias = self.create_parameter(
            shape=[vocab_size], dtype=self.decoder_weight.dtype, is_bias=True)

    def forward(self, hidden_states, masked_positions=None):
        if masked_positions is not None:
            hidden_states = paddle.reshape(hidden_states,
                                           [-1, hidden_states.shape[-1]])
            hidden_states = paddle.tensor.gather(hidden_states,
                                                 masked_positions)
        # gather masked tokens might be more quick
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = paddle.tensor.matmul(
            hidden_states, self.decoder_weight,
            transpose_y=True) + self.decoder_bias
        return hidden_states


class ErniePretrainingHeads(nn.Layer):

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 activation,
                 embedding_weights=None):
        super(ErniePretrainingHeads, self).__init__()
        self.predictions = ErnieLMPredictionHead(hidden_size, vocab_size,
                                                 activation, embedding_weights)
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output, pooled_output, masked_positions=None):
        prediction_scores = self.predictions(sequence_output, masked_positions)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class ErnieForPretraining(ErniePretrainedModel):

    def __init__(self, ernie):
        super(ErnieForPretraining, self).__init__()
        self.ernie = ernie
        self.cls = ErniePretrainingHeads(
            self.ernie.config["hidden_size"],
            self.ernie.config["vocab_size"],
            self.ernie.config["hidden_act"],
            embedding_weights=self.ernie.embeddings.word_embeddings.weight)

        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                masked_positions=None,
                inputs_embeds=None):
        with paddle.static.amp.fp16_guard():
            outputs = self.ernie(input_ids,
                                 token_type_ids=token_type_ids,
                                 position_ids=position_ids,
                                 attention_mask=attention_mask,
                                 inputs_embeds=inputs_embeds)
            sequence_output, pooled_output = outputs[:2]
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output, masked_positions)
            return prediction_scores, seq_relationship_score


class ErniePretrainingCriterion(paddle.nn.Layer):

    def __init__(self, vocab_size):
        super(ErniePretrainingCriterion, self).__init__()
        self.loss_fn = paddle.nn.loss.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(self, prediction_scores, seq_relationship_score,
                masked_lm_labels, next_sentence_labels, masked_lm_scale):
        with paddle.static.amp.fp16_guard():
            masked_lm_loss = paddle.nn.functional.softmax_with_cross_entropy(
                prediction_scores, masked_lm_labels, ignore_index=-1)
            masked_lm_loss = masked_lm_loss / masked_lm_scale
            next_sentence_loss = paddle.nn.functional.softmax_with_cross_entropy(
                seq_relationship_score, next_sentence_labels)
            return paddle.sum(masked_lm_loss) + paddle.mean(next_sentence_loss)


class ErnieOnlyMLMHead(nn.Layer):

    def __init__(self, hidden_size, vocab_size, activation, embedding_weights):
        super().__init__()
        self.predictions = ErnieLMPredictionHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            activation=activation,
            embedding_weights=embedding_weights)

    def forward(self, sequence_output, masked_positions=None):
        prediction_scores = self.predictions(sequence_output, masked_positions)
        return prediction_scores


class ErnieForMaskedLM(ErniePretrainedModel):
    """
    Ernie Model with a `masked language modeling` head on top.
    Args:
        ernie (:class:`ErnieModel`):
            An instance of :class:`ErnieModel`.
    """

    def __init__(self, ernie):
        super(ErnieForMaskedLM, self).__init__()
        self.ernie = ernie
        self.cls = ErnieOnlyMLMHead(
            self.ernie.config["hidden_size"],
            self.ernie.config["vocab_size"],
            self.ernie.config["hidden_act"],
            embedding_weights=self.ernie.embeddings.word_embeddings.weight)

        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                masked_positions=None,
                inputs_embeds=None,
                labels=None,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=False):
        outputs = self.ernie(input_ids,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             attention_mask=attention_mask,
                             inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output,
                                     masked_positions=masked_positions)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = paddle.nn.CrossEntropyLoss(
            )  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.reshape(
                    (-1, paddle.shape(prediction_scores)[-1])),
                labels.reshape((-1, )))
        if not return_dict:
            output = (prediction_scores, ) + outputs[2:]
            return ((masked_lm_loss, ) +
                    output) if masked_lm_loss is not None else (
                        output[0] if len(output) == 1 else output)

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ErnieForMultipleChoice(ErniePretrainedModel):
    """
    Ernie Model with a linear layer on top of the hidden-states output layer,
    designed for multiple choice tasks like RocStories/SWAG tasks.
    
    Args:
        ernie (:class:`ErnieModel`):
            An instance of ErnieModel.
        num_choices (int, optional):
            The number of choices. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of Ernie.
            If None, use the same value as `hidden_dropout_prob` of `ErnieModel`
            instance `ernie`. Defaults to None.
    """

    def __init__(self, ernie, num_choices=2, dropout=None):
        super(ErnieForMultipleChoice, self).__init__()
        self.num_choices = num_choices
        self.ernie = ernie
        self.dropout = nn.Dropout(dropout if dropout is not None else self.
                                  ernie.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie.config["hidden_size"], 1)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                inputs_embeds=None,
                labels=None,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=False):
        # input_ids: [bs, num_choice, seq_l]
        input_ids = input_ids.reshape(shape=(
            -1, input_ids.shape[-1]))  # flat_input_ids: [bs*num_choice,seq_l]

        if position_ids is not None:
            position_ids = position_ids.reshape(shape=(-1,
                                                       position_ids.shape[-1]))
        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape(
                shape=(-1, token_type_ids.shape[-1]))

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(
                shape=(-1, attention_mask.shape[-1]))

        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.reshape(
                shape=(-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1]))

        outputs = self.ernie(input_ids,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             attention_mask=attention_mask,
                             inputs_embeds=inputs_embeds,
                             output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states,
                             return_dict=return_dict)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)  # logits: (bs*num_choice,1)
        reshaped_logits = logits.reshape(
            shape=(-1, self.num_choices))  # logits: (bs, num_choice)

        loss = None
        if labels is not None:
            loss_fct = paddle.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        if not return_dict:
            output = (reshaped_logits, ) + outputs[2:]
            return ((loss, ) + output) if loss is not None else (
                output[0] if len(output) == 1 else output)

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
