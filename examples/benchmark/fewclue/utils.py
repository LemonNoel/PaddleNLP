import json
from functools import partial
from paddlenlp.datasets import load_dataset, MapDataset
from paddlenlp.prompt import InputExample
from paddlenlp.utils.log import logger


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
        text = fragments[0] + "[" + cand + "]" + fragments[1]
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
                        text_b="、".join(example["keyword"]),
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


def convert_cluewsc(example):
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
                        text_a="".join(text),
                        text_b="",
                        labels=example.get("label", None))


def convert_labels_to_ids(example, label_dict):
    if example.labels is not None:
        example.labels = label_dict[example.labels]
    return example


# 读取 FewCLUE 数据集
def load_fewclue(task_name, split_id, label_list):
    if task_name == "tnews":
        splits = [f"dev_{split_id}", "test_public", "test", "unlabeled"]
        dev_ds, public_test_ds, test_ds, unlabeled_ds = load_dataset(
            "fewclue", name=task_name, splits=splits, label_list=label_list)
        with open("data/tnews_train.json", "r") as fp:
            data = [x for x in fp.readlines() if x[0] != "#"]
            train_ds = MapDataset([json.loads(x.strip()) for x in data])
    else:
        # Load FewCLUE datasets and convert the samples to InputExample.
        splits = [
            f"train_{split_id}", f"dev_{split_id}", "test_public", "test",
            "unlabeled"
        ]
        train_ds, dev_ds, public_test_ds, test_ds, unlabeled_ds = load_dataset(
            "fewclue", name=task_name, splits=splits, label_list=label_list)

    if task_name == "chid":
        # IDEA B.1
        convert_efl = convert_chid_efl

        def convert_to_binary(dataset):
            new_data = []
            for example in dataset:
                new_data.extend(convert_efl(example))
            return MapDataset(new_data)

        train_ds = convert_to_binary(train_ds)
        dev_ds = convert_to_binary(dev_ds)
        public_test_ds = convert_to_binary(public_test_ds)
        test_ds = convert_to_binary(test_ds)
        unlabeled_ds = convert_to_binary(unlabeled_ds)
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
        public_test_ds = public_test_ds.map(convert_fn)
        test_ds = test_ds.map(convert_fn)
        unlabeled_ds = unlabeled_ds.map(convert_fn)

        convert_fn = partial(convert_labels_to_ids, label_dict=label_list)
        if task_name != "cmnli":
            train_ds = train_ds.map(convert_fn)
            dev_ds = dev_ds.map(convert_fn)
            public_test_ds = public_test_ds.map(convert_fn)

    return train_ds, dev_ds, public_test_ds, test_ds, unlabeled_ds
