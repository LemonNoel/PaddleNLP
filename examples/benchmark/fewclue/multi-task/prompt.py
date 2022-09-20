PROMPT = {
    "eprstmt": [("{'text':'text_a'}{'hard':'这个句话表示我'}{'mask'}{'hard':'喜欢这个东西'}",
                 {
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
    "chid": [("{'text':'text_a'}{'sep'}{'hard':'这句话中的成语使用'}{'mask'}{'mask'}", {
        0: "错误",
        1: "正确"
    })],
    "csl":
    [("{'text': 'text_a'}{'hard':'上文中'}{'mask'}{'hard': '这些关键词：'}{'text':'text_b'}",
      {
          "0": "没",
          "1": "有"
      })],
    "cluewsc": [("{'text':'text_a'}{'hard':'其中代词使用'}{'mask'}{'mask'}", {
        "false": "错误",
        "true": "正确"
    })],
}
