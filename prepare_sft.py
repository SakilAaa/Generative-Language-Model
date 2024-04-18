import json
import tiktoken 
import numpy as np

enc = tiktoken.get_encoding("gpt2")

train_middle, val_middle = [], []

with open('datasets\\l.jsonl', 'r', encoding='utf-8') as f:
    q, a = [], []
    for line_ in f:
        line = json.loads(line_)
        q.append(line['Question'])
        a.append(line['Answer'])
    ninety = int(len(q) * 0.9)
    for i in range(len(q)):
        qi_ = enc.encode_ordinary(q[i])
        ai_ = enc.encode_ordinary(a[i])
        qi, ai = np.array(qi_).astype(np.int64), np.array(ai_).astype(np.int64)
        q_size = len(qi)
        a_size = len(ai)
        output = np.concatenate((qi, ai))
        if len(output) > 512:
            output = output[:512]
            a_size = 512 - len(qi)
        else:
            zeros = np.full(512 - len(output), 50256).astype(np.int64)
            output = np.concatenate((output, zeros))
        assert(len(output) == 512)
        output = np.append(output, q_size)
        output = np.append(output, a_size) 
        assert(len(output) == 514)
        train_middle.append(output)
    
    for i in range(ninety, len(q)):
        qi_ = enc.encode_ordinary(q[i])
        ai_ = enc.encode_ordinary(a[i])
        qi, ai = np.array(qi_).astype(np.int64), np.array(ai_).astype(np.int64)
        q_size = len(qi)
        a_size = len(ai)
        output = np.concatenate((qi, ai))
        if len(output) > 512:
            output = output[:512]
        else:
            zeros = np.full(512 - len(output), 50256).astype(np.int64)
            output = np.concatenate((output, zeros))
        output = np.append(output, q_size)
        output = np.append(output, a_size)
        val_middle.append(output)
        
train_ids, val_ids = None, None
train_ids = np.concatenate(train_middle)
val_ids = np.concatenate(val_middle)
# save numpy array to file [name]/train.bin and [name]/val.bin

train_ids.tofile('data\\sft\\train_sft.bin')
val_ids.tofile('data\\sft\\val_sft.bin')