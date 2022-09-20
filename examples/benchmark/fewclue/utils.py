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
    # meta={"genre": example.get("genre", None)})


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


def A0_convert_chid(example):
    choices = ["一、", "二、", "三、", "四、", "五、", "六、", "七、"]
    text_b = " ".join(
        [choices[i] + x for i, x in enumerate(example["candidates"])])
    return InputExample(uid=example["id"],
                        text_a=example["content"].replace("#idiom", "____", 1),
                        text_b=text_b,
                        labels=example.get("answer", None))


def convert_chid_efl(example):
    # IDEA B.1
    bi_examples = []
    fragments = example["content"].split("#idiom#")
    label = example.get("answer", None)
    for idx, cand in enumerate(example["candidates"]):
        text = fragments[0] + "（" + cand + "）" + fragments[1]
        bi_examples.append(
            InputExample(uid=example["id"],
                         text_a=text,
                         text_b="这句话中" + cand + "使用",
                         labels=None if label is None else int(idx == label)))
    return bi_examples


def convert_csl(example):
    # Unlabeled: 19841. Long sentence.
    # IDEA C: Take it as a NER and compare list. Con: error propagation.
    return InputExample(uid=example["id"],
                        text_a=example["abst"],
                        text_b=",".join(example["keyword"]),
                        labels=example.get("label", None))


def convert_cluewsc(example):
    # TEMPLATE 3
    return InputExample(uid=example.get("id", None),
                        text_a="“" + example["text"] + "”其中" +
                        example["target"]["span2_text"],
                        text_b=example["target"]["span1_text"],
                        labels=example.get("label", None))


def A_convert_cluewsc(example):
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


def convert_labels_to_ids(example, verbalizer):
    if example.labels is not None:
        #example.labels = verbalizer.token_ids[verbalizer.labels_to_ids[
        #    example.labels]].squeeze(-1)
        example.labels = verbalizer.labels_to_ids[example.labels]
    return example


def load_fewclue(task_name, split_id, verbalizer):
    if task_name == "cmnli":
        splits = ['train', 'dev', 'test']
        train_ds, dev_ds, test_ds = load_dataset("clue",
                                                 name=task_name,
                                                 splits=splits,
                                                 label_list=label_list)
        public_test_ds = load_dataset("clue", name=task_name, splits=['test'])
    else:
        # Load FewCLUE datasets and convert the samples to InputExample.
        splits = [f"train_{split_id}", f"dev_{split_id}", "test_public", "test"]
        train_ds, dev_ds, public_test_ds, test_ds = load_dataset(
            "fewclue", name=task_name, splits=splits)  #, label_list=label_list)

    if task_name == "chid":
        # IDEA B.1
        def convert_to_binary(dataset):
            new_data = []
            for example in dataset:
                new_data.extend(convert_chid_efl(example))
            return MapDataset(new_data)

        train_ds = convert_to_binary(train_ds)
        dev_ds = convert_to_binary(dev_ds)
        public_test_ds = convert_to_binary(public_test_ds)
        test_ds = convert_to_binary(test_ds)
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

        convert_fn = partial(convert_labels_to_ids,
                             verbalizer=verbalizer)  # label_dict=label_list)
        if task_name != "cmnli":
            train_ds = train_ds.map(convert_fn)
            dev_ds = dev_ds.map(convert_fn)
            public_test_ds = public_test_ds.map(convert_fn)
            test_ds = test_ds.map(convert_fn)

        public_test_ds = MapDataset([x for x in public_test_ds][:2000])

    return train_ds, dev_ds, public_test_ds, test_ds


LABEL_MAP = {
    "bustm": {
        # "0": "不",
        # "1": "很"
        # "0": "而且",
        # "1": "所以"
        "0": "中立",
        "1": "蕴含"
    },
    "chid_a": {
        # IDEA A.0
        0: "一",
        1: "二",
        2: "三",
        3: "四",
        4: "五",
        5: "六",
        6: "七"
    },
    "chid": {
        # IDEA B.1
        0: "错误",
        1: "正确"
    },
    "cluewsc": {
        # A
        "false": "错误",
        "true": "正确"
        # IDEA D.2
        # "false": "错",
        # "true": "对"
    },
    "csl": {
        "0": "没",
        "1": "有"
        # "0": "中立",
        # "1": "蕴含"
    },
    "csldcp": {
        '材料科学与工程': '材料',
        '作物学': '作物',
        '口腔医学': '口腔',
        '药学': '药学',
        '教育学': '教育',
        '水利工程': '水利',
        '理论经济学': '理经',
        '食品科学与工程': '食品',
        '畜牧学/兽医学': '畜牧',
        '体育学': '体育',
        '核科学与技术': '核科',
        '力学': '力学',
        '园艺学': '园艺',
        '水产': '水产',
        '法学': '法学',
        '地质学/地质资源与地质工程': '地质',
        '石油与天然气工程': '石油',
        '农林经济管理': '农林',
        '信息与通信工程': '通信',
        '图书馆、情报与档案管理': '图书',
        '政治学': '政治',
        '电气工程': '电气',
        '海洋科学': '海洋',
        '民族学': '民族',
        '航空宇航科学与技术': '航空',
        '化学/化学工程与技术': '化学',
        '哲学': '哲学',
        '公共卫生与预防医学': '卫生',
        '艺术学': '艺术',
        '农业工程': '农工',
        '船舶与海洋工程': '船舶',
        '计算机科学与技术': '计科',
        '冶金工程': '冶金',
        '交通运输工程': '交通',
        '动力工程及工程热物理': '动力',
        '纺织科学与工程': '纺织',
        '建筑学': '建筑',
        '环境科学与工程': '环境',
        '公共管理': '公管',
        '数学': '数学',
        '物理学': '物理',
        '林学/林业工程': '林学',
        '心理学': '心理',
        '历史学': '历史',
        '工商管理': '工管',
        '应用经济学': '应经',
        '中医学/中药学': '中医',
        '天文学': '天文',
        '机械工程': '机械',
        '土木工程': '土木',
        '光学工程': '光学',
        '地理学': '地理',
        '农业资源利用': '农业',
        '生物学/生物科学与工程': '生物',
        '兵器科学与技术': '兵器',
        '矿业工程': '矿业',
        '大气科学': '大气',
        '基础医学/临床医学': '基础',
        '电子科学与技术': '电子',
        '测绘科学与技术': '测绘',
        '控制科学与工程': '控制',
        '军事学': '军事',
        '中国语言文学': '中文',
        '新闻传播学': '新闻',
        '社会学': '社会',
        '地球物理学': '地球',
        '植物保护': '植保'
    },
    "eprstmt": {
        'Negative': '不',
        'Positive': '很'
    },
    "iflytek": {
        '银行': '银行',
        '社区服务': '社区',
        '电商': '电商',
        '支付': '支付',
        '经营养成': '经营',
        '卡牌': '卡牌',
        '借贷': '借贷',
        '驾校': '驾校',
        '理财': '理财',
        '职考': '职考',
        '新闻': '新闻',
        '旅游资讯': '旅游',
        '公共交通': '交通',
        '魔幻': '魔幻',
        '医疗服务': '医疗',
        '影像剪辑': '影像',
        '动作类': '动作',
        '工具': '工具',
        '体育竞技': '体育',
        '小说': '小说',
        '运动健身': '运动',
        '相机': '相机',
        '辅助工具': '工具',
        '快递物流': '快递',
        '高等教育': '教育',
        '股票': '股票',
        '菜谱': '菜谱',
        '行车辅助': '行车',
        '仙侠': '仙侠',
        '亲子儿童': '亲子',
        '购物咨询': '购物',
        '射击游戏': '射击',
        '漫画': '漫画',
        '中小学': '小学',
        '同城服务': '同城',
        '成人教育': '成人',
        '求职': '求职',
        '电子产品': '电子',
        '艺术': '艺术',
        '薅羊毛': '赚钱',
        '约会社交': '约会',
        '经营': '经营',
        '兼职': '兼职',
        '短视频': '视频',
        '音乐': '音乐',
        '英语': '英语',
        '棋牌中心': '棋牌',
        '摄影修图': '摄影',
        '养生保健': '养生',
        '办公': '办公',
        '政务': '政务',
        '视频': '视频',
        '论坛圈子': '论坛',
        '彩票': '彩票',
        '直播': '直播',
        '其他': '其他',
        '休闲益智': '休闲',
        '策略': '策略',
        '即时通讯': '通讯',
        '汽车交易': '买车',
        '违章': '违章',
        '地图导航': '地图',
        '民航': '民航',
        '电台': '电台',
        '语言(非英语)': '语言',
        '搞笑': '搞笑',
        '婚恋社交': '婚恋',
        '社区超市': '超市',
        '日常养车': '养车',
        '杂志': '杂志',
        '视频教育': '在线',
        '家政': '家政',
        '影视娱乐': '影视',
        '装修家居': '装修',
        '体育咨讯': '资讯',
        '社交工具': '社交',
        '餐饮店': '餐饮',
        '美颜': '美颜',
        '问诊挂号': '挂号',
        '飞行空战': '飞行',
        '综合预定': '预定',
        '电影票务': '票务',
        '笔记': '笔记',
        '买房': '买房',
        '外卖': '外卖',
        '母婴': '母婴',
        '打车': '打车',
        '情侣社交': '情侣',
        '日程管理': '日程',
        '租车': '租车',
        '微博博客': '博客',
        '百科': '百科',
        '绘画': '绘画',
        '铁路': '铁路',
        '生活社交': '生活',
        '租房': '租房',
        '酒店': '酒店',
        '保险': '保险',
        '问答交流': '问答',
        '收款': '收款',
        'MOBA': '竞技',
        'K歌': '唱歌',
        '技术': '技术',
        '减肥瘦身': '减肥',
        '工作社交': '工作',
        '团购': '团购',
        '记账': '记账',
        '女性': '女性',
        '公务员': '公务',
        '二手': '二手',
        '美妆美业': '美妆',
        '汽车咨询': '汽车',
        '行程管理': '行程',
        '免费WIFI': '免费',
        '教辅': '教辅',
        '成人': '两性',
        '婚庆': '婚庆',
        '民宿短租': '民宿',
        '出国': '出国'
    },
    "tnews": {
        'news_story': '故事',
        'news_culture': '文化',
        'news_entertainment': '娱乐',
        'news_sports': '体育',
        'news_finance': '财经',
        'news_house': '房产',
        'news_car': '汽车',
        'news_edu': '教育',
        'news_tech': '科技',
        'news_military': '军事',
        'news_travel': '旅游',
        'news_world': '国际',
        'news_stock': '股票',
        'news_agriculture': '农业',
        'news_game': '电竞'
    },
    "ocnli": {
        # "entailment": "所以",
        # "contradiction": "但是",
        # "neutral": "而且"
        "entailment": "蕴含",
        "contradiction": "矛盾",
        "neutral": "中立"
    },
    "cmnli": {
        # FT.a
        # "entailment": "所以",
        # "contradiction": "但是",
        # "neutral": "而且"
        # FT.b
        "entailment": "蕴含",
        "contradiction": "矛盾",
        "neutral": "中立"
    }
}
LABEL_LIST = {k: list(v.keys()) for k, v in LABEL_MAP.items()}
