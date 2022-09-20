import json
import warnings
from functools import partial
from collections import defaultdict
from paddlenlp.datasets import load_dataset, MapDataset
from paddlenlp.prompt import InputExample
from paddlenlp.utils.log import logger

__all__ = ["load_fewclue"]

PROMPT = {
    "chid": [("{'text':'text_a'}{'sep'}{'hard':'这句话中的成语使用'}{'mask'}{'mask'}", {
        0: "错误",
        1: "正确"
    }), ("{'text':'text_a'}文中成语是否正确？{'mask'}", {
        0: "否",
        1: "是"
    }), ("{'text':'text_a'}成语{'text':'text_b'}使用是否正确？{'mask'}", {
        0: "否",
        1: "是"
    }), ("{'text':'text_a'}括号中成语填{'text':'text_b'}对吗？{'mask'}", {
        0: "错",
        1: "对"
    }),
             ("{'text':'text_a'}这句话中空格处填{'text':'text_b'}{'mask'}合适。", {
                 0: "不",
                 1: "很"
             })],
    "csl": [("标题“{'text': 'text_b'}”正文：“{'text':'text_a'}”题目和文章描述{'mask'}符。", {
        "0": "不",
        "1": "相"
    }),
            ("“{'text': 'text_a'}”这句话中讨论的关键词{'mask'}包括“{'text': 'text_b'}”", {
                "0": "不",
                "1": "已"
            }),
            ("“{'text': 'text_a'}”上文中找{'mask'}出这些关键词：“{'text':'text_b'}”", {
                "0": "不",
                "1": "得"
            }),
            ("“{'text': 'text_a'}”上文中{'mask'}'这些关键词：“{'text':'text_b'}”", {
                "0": "无",
                "1": "有"
            }),
            ("{'text':'text_a'}和{'text':'text_b'}这两句话说的是同一件事吗？{'mask'}", {
                "0": "否",
                "1": "是"
            })],
    "cluewsc": [
        ("{'text':'text_a'}{'hard':'其中代词使用'}{'mask'}{'mask'}", {
            "false": "错误",
            "true": "正确"
        }),
        ("{'text':'text_a'}这句话中代词{'text':'text_b'}对吗？{'mask'}", {
            "false": "错",
            "true": "对"
        }),
        ("{'text':'text_a'}这句话中词语{'text':'text_b'}对吗？{'mask'}", {
            "false": "错",
            "true": "对"
        }),
        ("{'text':'text_a'}其中{'text':'text_b'}。这句话描述正确吗？{'mask'}", {
            "false": "否",
            "true": "是"
        }),
        ("{'text':'text_a'}其中{'text':'text_b'}。这句话描述正确吗？{'mask'}{'mask'}", {
            "false": "错误",
            "true": "正确"
        }),
    ],
    "eprstmt":
    [("{'text':'text_a'}{'hard':'这个句话表示我'}{'mask'}{'hard':'喜欢这个东西'}", {
        'Negative': '不',
        'Positive': '很'
    })],
    "csldcp":
    [("{'hard':'阅读下边有关'}{'mask'}{'mask'}{'hard':'的材料'}{'text':'text_a'}",
      json.load(open("label_map/csldcp.json", "r")))],
    "tnews":
    [("{'hard':'下边播报一则'}{'mask'}{'mask'}{'hard':'新闻：'}{'text':'text_a'}",
      json.load(open("label_map/tnews.json", "r")))],
    "iflytek":
    [("{'text':'text_a'}{'hard':'这款应用是'}{'mask'}{'mask'}{'hard':'类型的。'}",
      json.load(open("label_map/iflytek.json", "r")))],
    "ocnli": [
        ("“{'text':'text_a'}”和“{'text':'text_b'}”之间的逻辑关系是{'mask'}{'mask'}", {
            "entailment": "蕴含",
            "contradiction": "矛盾",
            "neutral": "中立"
        })
    ],
    "bustm": [
        ("“{'text':'text_a'}”和“{'text':'text_b'}”之间的逻辑关系是{'mask'}{'mask'}", {
            "0": "中立",
            "1": "蕴含"
        })
    ],
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
    x.text_b = x.text_b["span2_text"] + "指代" + x.text_b["span1_text"]
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
    "clsdcp":
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
