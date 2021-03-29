# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

df = pd.DataFrame({'key1': ['a', 'a', 'b', 'b', 'a'], 'key2': ['one', 'two', 'one', 'two', 'one'],
                   'data1': np.random.randn(5), 'data2': np.random.randn(5)})

# for name, group in df.groupby('key1'):
#     print(name)
#     print(group)

# 多键的情况
for (k1, k2), group in df.groupby(['key1', 'key2']).count():
    print(k1, k2)
    print(group)