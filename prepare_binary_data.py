import os
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")

names = ['baidu', '书剑恩仇录', '侠客行', '倚天屠龙记', '射雕英雄传', '白马啸西风', '碧血剑', '越女剑', '雪山飞狐', '飞狐外传', '鸳鸯刀', '鹿鼎记', '连城诀', '神雕侠侣', '天龙八部', '笑傲江湖']

with open('single.txt', 'a', encoding='utf-8') as f:
    for name in names:
        location = r'/home/py2022010751/hw/hw2/datasets/%s.txt' % name
        with open(location, 'r', encoding='utf-8') as file:
            content = file.readline()
            while content:
                content = content.strip('\n')
                if content:
                    f.write(content)
                content = file.readline()

with open('single.txt', 'r', encoding='utf-8') as f:
    with open('train.txt', 'a', encoding='utf-8') as train:
        for i in range(10000):
            content = f.read(100000)
            train.write(content)
    with open('val.txt', 'a', encoding='utf-8') as val:
        while True:
            content = f.read(100000)   
            val.write(content)
            if not content:
                break
        
### split data for train(0.9) and valid (0.1)
train_data, val_data = None, None

with open('train.txt', 'r', encoding='utf-8') as train:
    train_data = train.read()
train_middle = enc.encode_ordinary(train_data)

with open('val.txt', 'r', encoding='utf-8') as val:
    val_data = val.read()
val_middle = enc.encode_ordinary(val_data)


### tokenize raw data with tiktoken encoder
### transform to numpy array
train_ids, val_ids = None, None
###
train_ids = np.array(train_middle).astype(np.int64)
val_ids = np.array(val_middle).astype(np.int64)
###

# save numpy array to file [name]/train.bin and [name]/val.bin
train_ids.tofile(os.path.join("processed_pretrain", "train.bin"))
val_ids.tofile(os.path.join("processed_pretrain", 'val.bin'))