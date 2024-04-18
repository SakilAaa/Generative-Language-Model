import torch
import numpy as np

train_data = None
val_data = None
    
def init_data_pretrain(dataset):
    global train_data, val_data
    train_data = np.memmap(r"/home/py2022010751/hw/hw2/MiniGLM/data/processed_pretrain/train.bin", dtype=np.int64, mode='r')
    val_data = np.memmap(r"/home/py2022010751/hw/hw2/MiniGLM/data/processed_pretrain/val.bin", dtype=np.int64, mode='r')

def init_data_sft(dataset):
    global train_data, val_data
    ### 读取+初始化sft数据
    train_data, val_data = None, None
    train_data = np.memmap(r'/home/py2022010751/hw/hw2/MiniGLM/data/sft/train_sft.bin', dtype=np.int64, mode='r')
    val_data = np.memmap(r'/home/py2022010751/hw/hw2/MiniGLM/data/sft/val_sft.bin', dtype=np.int64, mode='r')

def get_batch_pretrain(split, batch_size, block_size, device):
    global train_data, val_data
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    loss_mask = torch.ones_like(x, dtype=torch.float64)
    
    #device_type = 'cuda' if 'cuda' in device else 'cpu'
    device_type = 'cpu'
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y, loss_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), loss_mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
    return x, y, loss_mask
    
def get_batch_sft(split, batch_size, block_size, device): 
    ### 获取sft数据的批次（batch）+ 构建损失函数掩码（loss_mask）
    x, y, loss_mask = None, None, None
    ###
    block_size = 256
    global train_data, val_data
    data = train_data if split == 'train' else val_data
    block_size_ = block_size + 2
    size = int(len(data) / block_size_)
    ix = torch.randint(size, (batch_size, ))
    x = torch.stack([torch.from_numpy((data[i*block_size_:i*block_size_+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i*block_size_+1:i*block_size_+block_size+1]).astype(np.int64)) for i in ix])
    loss = []
    for i in ix:
        q_size = data[(i+1)*block_size_-2]
        a_size = data[(i+1)*block_size_-1]
        zero_1 = np.zeros(q_size)
        size_2 = block_size - q_size - a_size if block_size - q_size - a_size > 0 else 0
        zero_2 = np.zeros(size_2)
        one_size = a_size if block_size - q_size - a_size > 0 else block_size - q_size
        one = np.ones(one_size)
        loss_ = np.concatenate((zero_1, one, zero_2))
        loss.append(loss_)
    loss_mask = torch.from_numpy(np.array(loss))
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y, loss_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), loss_mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
    return x, y, loss_mask