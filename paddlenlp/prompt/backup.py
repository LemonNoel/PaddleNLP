## prompt_tokenizer.py
__all__ = ["TokenizerWrapper", "MLMTokenizerWrapper"]


class TokenizerWrapper:
    """
    Process examples encoded by template, such as truncating and padding.

    Args:
        max_seq_length (int):
            The maximum length of input data (prompt and text).
        tokenizer (paddlenlp.transformers.PreTrainedTokenizer):
            The tokenizer of pretrained model.
        truncate_method (str):
            How to truncate input data. 
            Choices: ``tail``, ``head``, ``manual``.
        create_token_type_ids (bool):
            Whether to create token_type_ids for inputs.
        seq_length_list (list, optional):
            The list of maximum length for every part in input data.  
    """

    def __init__(self,
                 max_seq_length,
                 tokenizer,
                 truncate_method='tail',
                 create_token_type_ids=False,
                 **kwargs):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        if truncate_method == 'manual':
            assert hasattr(kwargs, 'seq_length_list'), 'seq_length_list '\
                'should be defined for manual truncation.'
            self.seq_length_list = kwargs['seq_length_list']
            self.truncate_fn = partial(self.truncate_from_end, etype='tail')
        elif truncate_method == 'tail' or truncate_method == 'head':
            self.truncate_fn = partial(self.truncate_from_end,
                                       etype=truncate_method)
        else:
            raise NotImplementedError

        self.create_token_type_ids = create_token_type_ids

        self.num_truncated_sentences = 0
        self.total_passed_sentences = 0

    @property
    def special_tokens_maps(self):
        if not hasattr(self, "_special_tokens_map"):
            self._special_tokens_map = {
                '<cls>': getattr(self.tokenizer, 'cls_token', ''),
                '<sep>': getattr(self.tokenizer, 'sep_token', ''),
                '<pad>': getattr(self.tokenizer, 'pad_token', ''),
                '<mask>': getattr(self.tokenizer, 'mask_token', ''),
                '<unk>': getattr(self.tokenizer, 'unk_token', '')
            }
        return self._special_tokens_map

    @property
    def truncate_rate(self):
        if self.total_passed_sentences == 0:
            return None
        else:
            return self.num_truncated_sentences / self.total_passed_sentences

    @staticmethod
    def truncate_by_manual(input_dict, max_len_list=[]):
        """
        Truncate input data by manually defined maximum sequence length.

        Args:
            input_dict (dict):
                The dictionary of an input example.
            max_len_list (list):
                The maximum length of every part in example.
                ``-1`` denotes that there is no limit on length.
        """
        truncated_dict = defaultdict(list)
        shortenable_ids = input_dict['shortenable_ids']
        truncated_dict['shortenable_ids'] = shortenable_ids
        for attr_name, attr_values in input_dict.items():
            text_idx = 0
            for i, value in enumerate(attr_values):
                if shortenable_ids[i][0] == 0:
                    continue
                if text_idx >= len(max_len_list):
                    break
                if len(value) > 0:
                    max_len = max_len_list[text_idx]
                    if max_len < 0:
                        attr_values[i] = value
                    else:
                        attr_values[i] = value[:max_len]
                text_idx += 1
            truncated_dict[attr_name] = attr_values
        return truncated_dict

    @staticmethod
    def truncate_from_end(input_dict, num_tokens_to_truncate=0, etype='tail'):
        assert etype in ['head', 'tail']
        step = 1 if etype == 'head' else -1
        idx_offset = 0 if etype == 'head' else 1
        truncated_dict = defaultdict(list)
        shortenable_ids = input_dict['shortenable_ids']
        for attr_name in input_dict:
            attr_values = input_dict[attr_name]
            count = num_tokens_to_truncate
            for i, value in enumerate(attr_values[::step]):
                index = int(step * (idx_offset + i))
                if len(value) == 0 or shortenable_ids[index][0] == 0:
                    continue
                if count < len(value):
                    attr_values[index] = value[:-count]
                else:
                    attr_values[index] = []
                count -= len(value)
                if count <= 0:
                    break
            truncated_dict[attr_name] = attr_values

        return truncated_dict

    @staticmethod
    def concate_parts(input_dict):
        for key in input_dict:
            input_dict[key] = list(itertools.chain(*input_dict[key]))
        return input_dict

    @staticmethod
    def padding(input_dict,
                max_len,
                pad_id_for_inputs=0,
                pad_id_for_others: int = 0) -> None:
        for key, value in input_dict.items():
            if (len(input_dict[key]) > max_len):
                raise ValueError(
                    f'''Truncated seq length of '{key}' still greater than 
                    max length {max_len}. One possible reason is that 
                    no enough shortenable parts in template. Try adding
                    {{"shortenable": "True"}} property.
                ''')
            if 'input' in key:
                input_dict[key].extend([pad_id_for_inputs] *
                                       (max_len - len(value)))
            else:
                input_dict[key].extend([pad_id_for_others] *
                                       (max_len - len(value)))
        return input_dict

    def truncate(self, inputs):
        if hasattr(self, 'seq_length_list'):
            inputs = self.truncate_by_manual(inputs, self.seq_length_list)
        total_tokens = sum([len(part) for part in inputs['input_ids']])
        num_specials = self.num_special_tokens_to_add
        num_tokens_to_truncate = total_tokens - self.max_seq_length + num_specials
        self.total_passed_sentences += 1
        if num_tokens_to_truncate > 0:
            self.num_truncated_sentences += 1
            inputs = self.truncate_fn(
                input_dict=inputs,
                num_tokens_to_truncate=num_tokens_to_truncate)
        return inputs

    def add_special_tokens(self, encode_inputs):
        for key in encode_inputs:
            if key == "input_ids":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    encode_inputs[
                        key] = self.tokenizer.build_inputs_with_special_tokens(
                            encode_inputs[key])
            else:
                special_tokens_mask = np.array(
                    self.tokenizer.get_special_tokens_mask(encode_inputs[key]))
                with_special_tokens = np.array(
                    self.tokenizer.build_inputs_with_special_tokens(
                        encode_inputs[key]))
                with_special_tokens[special_tokens_mask == 1] = 0
                encode_inputs[key] = with_special_tokens.tolist()
        return encode_inputs


class MLMTokenizerWrapper(TokenizerWrapper):
    input_keys = ['input_ids', 'attention_mask', 'token_type_ids']

    @property
    def mask_token(self):
        return self.tokenizer.mask_token

    @property
    def mask_token_id(self):
        return self.tokenizer.mask_token_id

    @property
    def soft_token(self):
        return self.tokenizer.unk_token

    @property
    def soft_token_id(self):
        return self.tokenizer.unk_token_id

    @property
    def num_special_tokens_to_add(self):
        if not hasattr(self, '_num_specials'):
            self._num_specials = self.tokenizer.num_special_tokens_to_add()
        return self._num_specials

    def get_token_type_ids(self, encoded_inputs):
        token_type_ids = [0] * len(encode_inputs['input_ids'])
        sep_token = getattr(self.tokenizer, 'sep_token', -1)
        if sep_token >= 0:
            sep_index = np.where(
                [x == sep_token for x in encode_inputs['input_ids']])[0]
            for i, x in enumerate(sep_index[1:]):
                pre_x = sep_index[i - 1]
                sep_index[pre_x + 1:x + 1] = [i + 1] * (x - pre_x)
        return token_type_ids

    def tokenize_one_example(self, wrapped_example):
        to_tokenize, not_to_tokenize = wrapped_example

        encode_inputs = defaultdict(list)
        for part in to_tokenize:
            if part['mask_ids'] == 1:
                text = [self.mask_token_id]

            if part['text'] in self.special_tokens_maps.keys():
                to_replace = self.special_tokens_maps[part['text']]
                if to_replace is not None:
                    part['text'] = to_replace
                else:
                    raise KeyError(
                        "This tokenizer doesn't specify {} token.".format(
                            piece['prompt']))

            if 'soft_token_ids' in part and part['soft_token_ids'] == 1:
                text = [self.soft_token_id]
            else:
                text = self.tokenizer.encode(
                    part['text'],
                    add_special_tokens=False,
                    return_token_type_ids=False)['input_ids']

            text_len = len(text)
            encode_inputs['input_ids'].append(text)
            for key in part:
                if key not in ['text']:
                    encode_inputs[key].append([part[key]] * text_len)
        encode_inputs = self.truncate(inputs=encode_inputs)
        encode_inputs.pop('shortenable_ids')
        encode_inputs = self.concate_parts(encode_inputs)
        encode_inputs = self.add_special_tokens(encode_inputs)
        encode_inputs['attention_mask'] = [1] * len(encode_inputs['input_ids'])
        if self.create_token_type_ids:
            encode_inputs['token_type_ids'] = get_token_type_ids(encode_inputs)
        encode_inputs = self.padding(
            encode_inputs,
            max_len=self.max_seq_length,
            pad_id_for_inputs=self.tokenizer.pad_token_id)

        for key in not_to_tokenize:
            encode_inputs[key] = not_to_tokenize[key]

        return InputFeatures(**encode_inputs)

## prompt_args.py
    cls_threshold: float = field(
        default=0.5,
        metadata={"help": "The threshold to select predicted labels for "\
                          "multi-label classification."})
    cls_top_k: int = field(
        default=1,
        metadata={"help": "Select the top k labels as prediction results."})


## prompt_utils.py
from ..datasets import MapDataset


class FewShotSampler(object):
    """
    Sampling from datasets for few-shot learning.
    Args:
        dataset
    """

    def __init__(self,
                 num_sample_per_label=None,
                 num_sample_total=None,
                 eval_num_sample_per_label=None,
                 eval_num_sample_total=None):
        if num_sample_per_label is None and num_sample_total is None:
            raise ValueError("Either `num_sample_per_label` or `num_sample_total`"\
                             " should be set.")
        if num_sample_per_label is not None and num_sample_total is not None:
            logger.info(
                "`num_sample_per_label` will overwrite `num_sample_total`.")
            self.num_sample_per_label = num_sample_per_label
            self.num_sample_total = None
        else:
            self.num_sample_per_label = num_sample_per_label
            self.num_sample_total = num_sample_total

        if eval_num_sample_per_label is None and eval_num_sample_total is None:
            self.eval_num_sample_per_label = self.num_sample_per_label
            self.eval_num_sample_total = self.num_sample_total
        elif eval_num_sample_per_label is not None and eval_num_sample_total is not None:
            logger.info(
                "`eval_num_sample_per_label` will overwrite `eval_num_sample_total`."
            )
            self.eval_num_sample_per_label = eval_num_sample_per_label
            self.eval_num_sample_total = None
        else:
            self.eval_num_sample_per_label = eval_num_sample_per_label
            self.eval_num_sample_total = eval_num_sample_total

    def sample_datasets(self, train_dataset, dev_dataset=None, seed=None):
        """
        Sample from every given dataset seperately.
        """
        self.rng = np.random.RandomState(seed)
        indices = np.arange(len(train_dataset))
        labels = [x.labels for x in train_dataset]
        train_indices = self._sample(indices, labels, self.num_sample_per_label,
                                     self.num_sample_total)
        logger.info(f"{len(train_indices)} examples sampled for train dataset.")
        train_ds = MapDataset([train_dataset[i] for i in train_indices],
                              label_list=train_dataset.label_list)

        if dev_dataset is None:
            return train_ds

        indices = np.arange(len(dev_dataset))
        labels = [x.labels for x in dev_dataset]
        eval_indices = self._sample(indices, labels,
                                    self.eval_num_sample_per_label,
                                    self.eval_num_sample_total)
        logger.info(f"{len(train_indices)} examples sampled for train dataset.")
        dev_ds = MapDataset([dev_dataset[i] for i in eval_indices],
                            label_list=dev_dataset.label_list)
        return train_ds, dev_ds

    def sample_and_partition(self, dataset, seed=None):
        """
        Sample from a single dataset and divide it into train, dev and test.
        """
        self.rng = np.random.RandomState(seed)
        total_indices = np.arange(len(dataset))
        total_labels = [x.labels for x in dataset]
        train_indices = self._sample(total_indices, total_labels,
                                     self.num_sample_per_label,
                                     self.num_sample_total)
        logger.info(f"{len(train_indices)} examples sampled for train dataset.")

        non_train_indices = [
            i for i in total_indices if i not in set(train_indices)
        ]
        non_train_labels = [total_labels[i] for i in non_train_indices]
        eval_indices = self._sample(non_train_indices, non_train_labels,
                                    self.eval_num_sample_per_label,
                                    self.eval_num_sample_total)
        logger.info(f"{len(eval_indices)} examples sampled for dev dataset.")

        test_indices = [
            i for i in non_train_indices if i not in set(eval_indices)
        ]
        logger.info(f"{len(test_indices)} examples left as test dataset.")

        train_ds = MapDataset([dataset.data[i] for i in train_indices],
                              label_list=dataset.label_list)
        dev_ds = MapDataset([dataset.data[i] for i in eval_indices],
                            label_list=dataset.label_list)
        test_ds = MapDataset([dataset.data[i] for i in test_indices],
                             label_list=dataset.label_list)
        return train_ds, dev_ds, test_ds

    def _sample(self, indices, labels, num_per_label, num_total):
        if num_per_label is not None:
            sampled_ids = self._sample_per_label(indices, labels, num_per_label)
        else:
            sampled_ids = self._sample_random(indices, num_total)
        return sampled_ids

    def _sample_random(self, indices, num_sample):
        if num_sample > len(indices):
            logger.info("Number to sample exceeds the number of examples " +
                        f"remaining. Only {len(indices)} sampled.")
        self.rng.shuffle(indices)
        return indices[:num_sample]

    def _sample_per_label(self, indices, labels, num_per_label):
        label_dict = defaultdict(list)
        for idx, label in zip(indices, labels):
            # One-hot labels for multi-label tasks.
            if isinstance(label, list):
                label = np.where(np.array(label) > 0)[0]
                for sublabel in label:
                    label_dict[sublabel].append(idx)
            else:
                label_dict[label].append(idx)

        sampled = []
        for label, index in label_dict.items():
            if num_per_label > len(index):
                logger.info("Number to sample exceeds the number of examples" +
                            f" with label {label}, {len(index)} sampled.")
            self.rng.shuffle(index)
            sampled.extend(index[:num_per_label])

        return sampled
