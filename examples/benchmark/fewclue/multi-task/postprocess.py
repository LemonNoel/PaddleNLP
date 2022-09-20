import os
import json
import pathlib
from collections import defaultdict

import numpy as np
from scipy.special import softmax

from paddlenlp.utils.log import logger


def postprocess(test_ret, test_ds, task_name, id_to_label):

    # Unify the results with the same uid.
    def unify_splited(test_ret, test_ds, n_split=None):
        ret_list = []
        ret_dict = defaultdict(list)
        preds = softmax(test_ret.predictions, axis=1)[:, 1]
        for idx, example in enumerate(test_ds):
            uid = getattr(example, "uid", idx // n_split if n_split else None)
            assert uid is not None
            ret_dict[uid].append(preds[idx])
        for uid, pred in ret_dict.items():
            ret_list.append({"id": uid, "answer": int(np.argmax(pred))})
        return ret_list

    # Result of chid_split.
    if task_name == "chid":
        return unify_splited(test_ret, test_ds, n_split=7)

    # Result of csl_split.
    if task_name == "csl":
        return unify_splited(test_ret, test_ds)

    # Convert to standard ids.
    if task_name == "iflytek":
        remap = json.load(open("label_map/iflytek_ids.json", "r"))
    elif task_name == "tnews":
        remap = {
            'news_story': '100',
            'news_culture': '101',
            'news_entertainment': '102',
            'news_sports': '103',
            'news_finance': '104',
            'news_house': '106',
            'news_car': '107',
            'news_edu': '108',
            'news_tech': '109',
            'news_military': '110',
            'news_travel': '112',
            'news_world': '113',
            'news_stock': '114',
            'news_agriculture': '115',
            'news_game': '116'
        }

    preds = np.argmax(test_ret.predictions, axis=1)
    for idx, example in enumerate(test_ds):
        uid = getattr(example, "uid", idx)
        if task_name in ["bustm", "csl"]:
            ret_list.append({"id": uid, "label": str(preds[idx])})
        elif task_name == "chid":
            ret_list.append({"id": uid, "answer": preds[idx]})
        elif task_name in ["cluewsc", "eprstmt", "ocnli", "csldcp"]:
            ret_list.append({"id": uid, "label": id_to_label[preds[idx]]})
        elif task_name in ["tnews"]:
            ret_list.append({
                "id": uid,
                "label": remap[id_to_label[preds[idx]]]
            })
        elif task_name in ["iflytek"]:
            ret_list.append({
                "id": uid,
                "label": str(remap[id_to_label[preds[idx]]])
            })
    return ret_list


def save_to_file(test_ret, task_name, save_path='./fewclue_submit_examples'):
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    task_name = task_name if task_name in ["bustm", "csldcp", "eprstmt"
                                           ] else task_name + "f"
    with open(os.path.join(save_path, task_name + "_predict.json"), "w") as fp:
        for ret in test_ret:
            fp.write(json.dumps(ret) + "\n")
    logger.info(f"Predictions for {task_name} saved to {save_path}.")
