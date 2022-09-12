import torch
import attention_types_imp as imp
from tqdm.auto import tqdm
import json
import math

def profile_attention_type(cls):
    repeat_cnt=500

    print(f'----------------------{cls}-------------------')

    res = []
    for bs in tqdm(range(8, 17, 8)):
        sl_times = []
        for sl in tqdm(range(64, 2000, 128)):
            rp = max(1, int(repeat_cnt * math.pow(64, 1/3.0) / math.pow(sl, 1/3.0)))
            totals_mh = imp.test_attention_total_time(cls, bs=bs, sl=sl, nh=16, hs=64, repeat_cnt=rp)
            sl_times.append((sl, totals_mh))
        res.append((bs, sl_times))

    return res


# warmup
imp.test_attention_total_time(imp.MultiHead, bs=24, sl=8, nh=16, hs=64, repeat_cnt=100)

if True:
    res = {
        'MultiHead': profile_attention_type(imp.MultiHead),
        'MultiQuery': profile_attention_type(imp.MultiQuery),
        'MultiQuery1': profile_attention_type(imp.MultiQuery1),
    }

    with open('profile_attention_types1.json', 'w') as f:
        json.dump(res, f)

