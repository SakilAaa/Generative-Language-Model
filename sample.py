import gradio as gr
import json
import os
from contextlib import nullcontext
import torch
import tiktoken
import argparse
from model import GLMConfig, MiniGLM

# -----------------------------------------------------------------------------

input_data, output_data = None, None

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
parser.add_argument("--output_data", type=str)
args = parser.parse_args()

input_data, output_data = args.input_data, args.output_data

out_dir = r"/home/py2022010751/hw/hw2/MiniGLM/40000pre" # ignored if init_from is not 'resume'
max_new_tokens = 512 # number of tokens generated in each sample
temperature = 0.9 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 250 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1234
device = 'cuda:4' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster

# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
config = GLMConfig(**checkpoint['model_args'])
model = MiniGLM(config)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# encode the beginning of the prompt

def model_gen(prompt):
    if prompt[-1] != '。':
        prompt = prompt.replace('?', '')
        prompt = prompt.replace('？', '')
        prompt += '？'
    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.int64, device=device)[None, ...]
        # run generation
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            output_tokens = y[0].tolist()
            try:
                end_idx = output_tokens.index(50256)
                output_tokens = output_tokens[:end_idx]
            except:
                pass
            output = decode(output_tokens)
            output = output.lstrip(prompt)
            output = output.replace('�', '')
            output = output.split()
            output = ''.join(output)
    return output

with open(input_data, 'r', encoding='utf-8') as input_f:
    with open(output_data, 'a', encoding='utf-8') as output_f:
        for input__ in input_f:
            input_ = json.loads(input__)
            q = input_['question']
            a = model_gen(q)
            pair = "{\"question\": \"%s\", \"answer\": \"%s\"}\n" % (q, a)
            output_f.write(pair)
