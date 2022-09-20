import json
import warnings
from functools import partial
from collections import defaultdict
from paddlenlp.datasets import load_dataset, MapDataset
from paddlenlp.prompt import InputExample
from paddlenlp.utils.log import logger

__all__ = ["load_fewclue"]

PROMPT = {
    "chid": [
        ("“{'text':'text_a'}”这句话中成语[{'text':'text_b'}]的理解正确吗？{'mask'}{'mask'}。",
         {
             0: "错误",
             1: "正确"
         }),
    ],
    "csl":
    [("“{'text':'text_a'}”其中“{'text':'text_b'}”{'mask'}{'mask'}这句话的关键词。", {
        "0": "就是",
        "1": "不是"
    })],
    "cluewsc": [
        ("“{'text':'text_a'}”这句话中代词{'text':'text_b'}的理解正确吗？{'mask'}{'mask'}", {
            "false": "错误",
            "true": "正确"
        }),
    ],
    "eprstmt": [("“{'text':'text_a'}”这条评论的情感倾向是{'mask'}{'mask'}的。", {
        'Negative': '消极',
        'Positive': '积极'
    })],
    "csldcp": [("“{'text':'text_a'}”这篇文献的类别是{'mask'}{'mask'}。",
                json.load(open("label_map/csldcp.json", "r")))],
    "tnews": [("“{'text':'text_a'}”上述新闻选自{'mask'}{'mask'}专栏。",
               json.load(open("label_map/tnews.json", "r")))],
    "iflytek": [("“{'text':'text_a'}”这个句子描述的应用类别是{'mask'}{'mask'}。",
                 json.load(open("label_map/iflytek.json", "r")))],
    "ocnli":
    [("“{'text':'text_a'}”和“{'text':'text_b'}”这两句话之间的逻辑关系是{'mask'}{'mask'}", {
        "entailment": "蕴含",
        "contradiction": "矛盾",
        "neutral": "中立"
    })],
    "bustm":
    [("“{'text':'text_a'}”和“{'text':'text_b'}”描述的是{'mask'}{'mask'}的事情。", {
        "0": "不同",
        "1": "相同"
    })],
}


# Define the label mapping.
# ============================================
def convert_label(x, verbalizer):
    if x.labels is not None:
        if x.labels in verbalizer.labels_to_ids:
            x.labels = verbalizer.labels_to_ids[x.labels]
        # Use all tokens in vocabulary for loss.
        # x.labels = verbalizer.token_ids[x.labels].squeeze(-1)
    return x


# Define the preprocess mapping of InputExample
# ============================================
PREPROCESS_MAP = defaultdict(lambda x: x)


def chid_single(x):
    x.text_b = "｜".join(x.text_b)
    return x


#PREPROCESS_MAP["chid"] = chid_single


def csl_single(x):
    x.text_b = "｜".join(x.text_b)
    return x


#PREPROCESS_MAP["csl"] = csl_single


def cluewsc_single(x):
    pronoun, p_index = x.text_b["span2_text"], x.text_b["span2_index"]
    entity, e_index = x.text_b["span1_text"], x.text_b["span1_index"]
    text_a = [c for c in x.text_a]
    if p_index > e_index:
        text_a.insert(p_index, "_")
        text_a.insert(p_index + len(pronoun) + 1, "_")
        text_a.insert(e_index, "[")
        text_a.insert(e_index + len(entity) + 1, "]")
    else:
        text_a.insert(e_index, "[")
        text_a.insert(e_index + len(entity) + 1, "]")
        text_a.insert(p_index, "_")
        text_a.insert(p_index + len(pronoun) + 1, "_")
    x.text_a = "".join(text_a)
    x.text_b = "_" + x.text_b["span2_text"] + "_指代[" + x.text_b[
        "span1_text"] + "]"
    return x


PREPROCESS_MAP["cluewsc"] = cluewsc_single

# Define the preprocess function of MapDataset
# ============================================
PREPROCESS_FUNC = defaultdict(lambda x: x)


def chid_split(data_ds):
    new_ds = []
    for example in data_ds:
        fragments = example.text_a.split("#idiom#")
        label = example.labels
        for idx, cand in enumerate(example.text_b):
            text = fragments[0] + "[" + cand + "]" + fragments[1]
            new_ds.append(
                InputExample(
                    uid=example.uid,
                    text_a=text,
                    text_b=cand,
                    labels=None if label is None else int(idx == label)))
    return MapDataset(new_ds)


PREPROCESS_FUNC["chid"] = chid_split


def csl_split(data_ds):
    new_ds = []
    for example in data_ds:
        for idx, cand in enumerate(example.text_b):
            new_ds.append(
                InputExample(uid=example.uid,
                             text_a=example.text_a,
                             text_b=cand,
                             labels=example.labels))
    return MapDataset(new_ds)


PREPROCESS_FUNC["csl"] = csl_split

# Default dataset
# ============================================
DEFAULT_CONVERT = {
    "eprstmt":
    lambda x: InputExample(uid=x["id"],
                           text_a=x["sentence"],
                           text_b="",
                           labels=x.get("label", None)),
    "csldcp":
    lambda x: InputExample(uid=x["id"],
                           text_a=x["content"],
                           text_b="",
                           labels=x.get("label", None)),
    "tnews":
    lambda x: InputExample(uid=x["id"],
                           text_a=x["sentence"],
                           text_b="",
                           labels=x.get("label_desc", None)),
    "iflytek":
    lambda x: InputExample(uid=x["id"],
                           text_a=x["sentence"],
                           text_b="",
                           labels=x.get("label_des", None)),
    "ocnli":
    lambda x: InputExample(uid=x.get("id", None),
                           text_a=x["sentence1"],
                           text_b=x["sentence2"],
                           labels=x.get("label", None)),
    "ocnli_genre":
    lambda x: InputExample(uid=x.get("id", None),
                           text_a=x["sentence1"],
                           text_b=x["sentence2"],
                           labels=x.get("genre", None)),
    "bustm":
    lambda x: InputExample(uid=x["id"],
                           text_a=x["sentence1"],
                           text_b=x["sentence2"],
                           labels=x.get("label", None)),
    "chid":
    lambda x: InputExample(uid=x["id"],
                           text_a=x["content"],
                           text_b=x["candidates"],
                           labels=x.get("answer", None)),
    "csl":
    lambda x: InputExample(uid=x["id"],
                           text_a=x["abst"],
                           text_b=",".join(x["keyword"]),
                           labels=x.get("label", None)),
    "cluewsc":
    lambda x: InputExample(uid=x.get("id", None),
                           text_a=x["text"],
                           text_b=x["target"],
                           labels=x.get("label", None))
}


def load_fewclue(task_name, split_id, verbalizer):
    splits = [f"train_{split_id}", f"dev_{split_id}", "test_public", "test"]
    datasets = load_dataset("fewclue", name=task_name,
                            splits=splits)  #, verbalizer=verbalizer)

    datasets = [x for x in datasets]
    for idx, data_ds in enumerate(datasets):
        data_ds = data_ds.map(DEFAULT_CONVERT[task_name])
        if task_name in PREPROCESS_MAP and task_name in PREPROCESS_FUNC:
            warnings.warn("Both mapping and functional processors are set.")
        if task_name in PREPROCESS_MAP:
            data_ds = data_ds.map(PREPROCESS_MAP[task_name])
        if task_name in PREPROCESS_FUNC:
            data_ds = PREPROCESS_FUNC[task_name](data_ds)
        trans_fn = partial(convert_label, verbalizer=verbalizer)
        data_ds.map(trans_fn)
        datasets[idx] = data_ds

    train_ds, dev_ds, public_test_ds, test_ds = datasets
    return train_ds, dev_ds, public_test_ds, test_ds
