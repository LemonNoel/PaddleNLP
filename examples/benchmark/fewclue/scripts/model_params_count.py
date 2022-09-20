import paddle

from paddlenlp.transformers import AutoModelForMaskedLM

model_list = ['ernie-1.0-large-zh-cw'
              ]  #['roformer_v2_chinese_char_base'] #['ernie-3.0-base-zh']
for modelname in model_list:
    sd = AutoModelForMaskedLM.from_pretrained(modelname).state_dict()
    # paddle.numelmodel.state_dict()
    print(
        "======================================================================"
    )
    print(modelname,
          sum([paddle.numel(sd[key]) for key in sd.keys()]) / 1000000)
