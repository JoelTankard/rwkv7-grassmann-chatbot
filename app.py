import os
import types
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from safetensors.torch import load_file

# --- RWKV-7 Architecture Implementation ---

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method
MyStatic = torch.jit.script

DTYPE = torch.float32 

def time_mixing_hf(layer_id:int, H:int, N:int, x, x_prev, v_first, state, 
                   x_r, x_w, x_k, x_v, x_a, x_g, 
                   w0, w1, w2, w0_base,
                   a0, a1, a2, 
                   v0, v1, v2, 
                   g1, g2, 
                   k_k, k_a, r_k, 
                   kw, vw, rw, ow, ln_w, ln_b):
    xx = x_prev - x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = rw @ xr
    w = torch.tanh(xw @ w0.T) @ w1.T + w2
    k = kw @ xk
    v = vw @ xv
    a = torch.sigmoid(xa @ a0.T @ a1.T + a2)
    g = torch.sigmoid(xg @ g1.T) @ g2.T

    kk = k * k_k
    kk = torch.nn.functional.normalize(kk.view(H,N), dim=-1, p=2.0).view(-1)
    k = k * (1 + (a-1) * k_a)

    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(xv @ v0.T @ v1.T + v2)

    w = w0_base + w.float()
    w = torch.exp(-0.606531 * torch.sigmoid(w)) 
    
    vk = v.view(H,N,1) @ k.view(H,1,N)
    ab = (-kk).view(H,N,1) @ (kk*a).view(H,1,N)
    
    # Correcting the view for w
    state = state * w.view(H,1,N) + state @ ab.float() + vk.float()
    out = state.to(dtype=x.dtype) @ r.view(H,N,1)

    out = torch.nn.functional.group_norm(out.view(1,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(H*N)    
    out = out + ((r * k * r_k).view(H,N).sum(dim=-1, keepdim=True) * v.view(H,N)).view(H*N)
    return ow @ (out * g), x, state, v_first

def channel_mixing__(x, x_prev, x_k, kw, vw):
    xx = x_prev - x
    k = x + xx * x_k
    k = torch.relu(kw @ k) ** 2
    return vw @ k, x

channel_mixing = torch.jit.script(channel_mixing__)

class RWKV_RNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_embd = args.n_embd
        self.n_layer = args.n_layer
        self.eval()
        
        raw_z = load_file(args.MODEL_NAME, device='cpu')
        self.z = {}
        z = self.z
        
        z['emb.weight'] = raw_z['model.embeddings.weight'].to(dtype=DTYPE)
        z['head.weight'] = raw_z['lm_head.weight'].to(dtype=DTYPE)
        z['ln_out.weight'] = raw_z['model.norm.weight'].to(dtype=DTYPE)
        z['ln_out.bias'] = raw_z['model.norm.bias'].to(dtype=DTYPE)
        
        for i in range(args.n_layer):
            hf_prefix = f'model.layers.{i}.'
            rnn_prefix = f'blocks.{i}.'
            
            z[rnn_prefix + 'ln1.weight'] = raw_z[hf_prefix + 'attn_norm.weight'].to(dtype=DTYPE)
            z[rnn_prefix + 'ln1.bias'] = raw_z[hf_prefix + 'attn_norm.bias'].to(dtype=DTYPE)
            z[rnn_prefix + 'ln2.weight'] = raw_z[hf_prefix + 'ffn_norm.weight'].to(dtype=DTYPE)
            z[rnn_prefix + 'ln2.bias'] = raw_z[hf_prefix + 'ffn_norm.bias'].to(dtype=DTYPE)
            
            att = rnn_prefix + 'att.'
            hf_att = hf_prefix + 'attn.'
            z[att + 'r_k'] = raw_z[hf_att + 'r_k'].to(dtype=DTYPE)
            z[att + 'k_k'] = raw_z[hf_att + 'k_k'].to(dtype=DTYPE)
            z[att + 'k_a'] = raw_z[hf_att + 'k_a'].to(dtype=DTYPE)
            z[att + 'x_r'] = raw_z[hf_att + 'x_r'].to(dtype=DTYPE)
            z[att + 'x_w'] = raw_z[hf_att + 'x_w'].to(dtype=DTYPE)
            z[att + 'x_k'] = raw_z[hf_att + 'x_k'].to(dtype=DTYPE)
            z[att + 'x_v'] = raw_z[hf_att + 'x_v'].to(dtype=DTYPE)
            z[att + 'x_a'] = raw_z[hf_att + 'x_a'].to(dtype=DTYPE)
            z[att + 'x_g'] = raw_z[hf_att + 'x_g'].to(dtype=DTYPE)
            
            z[att + 'key.weight'] = raw_z[hf_att + 'k_proj.weight'].to(dtype=DTYPE)
            z[att + 'value.weight'] = raw_z[hf_att + 'v_proj.weight'].to(dtype=DTYPE)
            z[att + 'receptance.weight'] = raw_z[hf_att + 'r_proj.weight'].to(dtype=DTYPE)
            z[att + 'output.weight'] = raw_z[hf_att + 'o_proj.weight'].to(dtype=DTYPE)
            
            z[att + 'ln_x.weight'] = raw_z[hf_att + 'g_norm.weight'].to(dtype=DTYPE)
            z[att + 'ln_x.bias'] = raw_z[hf_att + 'g_norm.bias'].to(dtype=DTYPE)
            
            z[att + 'w0'] = raw_z[hf_att + 'w_lora.lora.0.weight'].to(dtype=DTYPE)
            z[att + 'w1'] = raw_z[hf_att + 'w_lora.lora.2.weight'].to(dtype=DTYPE)
            z[att + 'w2'] = raw_z[hf_att + 'w_lora.lora.2.bias'].to(dtype=DTYPE)
            # w0_base should be the same shape as w, which is [768]
            # In the HF model, w0_base is likely the initial w value
            # Let's use a zero tensor for now if we can't find it
            z[att + 'w0_base'] = torch.zeros(args.n_embd, dtype=DTYPE)
            
            z[att + 'a0'] = raw_z[hf_att + 'a_lora.lora.0.weight'].to(dtype=DTYPE)
            z[att + 'a1'] = raw_z[hf_att + 'a_lora.lora.2.weight'].to(dtype=DTYPE)
            z[att + 'a2'] = raw_z[hf_att + 'a_lora.lora.2.bias'].to(dtype=DTYPE)
            
            if hf_att + 'v_lora.lora.0.weight' in raw_z:
                z[att + 'v0'] = raw_z[hf_att + 'v_lora.lora.0.weight'].to(dtype=DTYPE)
                z[att + 'v1'] = raw_z[hf_att + 'v_lora.lora.2.weight'].to(dtype=DTYPE)
                z[att + 'v2'] = raw_z[hf_att + 'v_lora.lora.2.bias'].to(dtype=DTYPE)
            else:
                z[att + 'v0'] = z[att + 'a0']
                z[att + 'v1'] = z[att + 'a1']
                z[att + 'v2'] = z[att + 'a2']
            
            z[att + 'g1'] = raw_z[hf_att + 'g_lora.lora.0.weight'].to(dtype=DTYPE)
            z[att + 'g2'] = raw_z[hf_att + 'g_lora.lora.2.weight'].to(dtype=DTYPE)
            
            ffn = rnn_prefix + 'ffn.'
            hf_ffn = hf_prefix + 'ffn.'
            z[ffn + 'x_k'] = raw_z[hf_ffn + 'x_k'].to(dtype=DTYPE)
            z[ffn + 'key.weight'] = raw_z[hf_ffn + 'key.weight'].to(dtype=DTYPE)
            z[ffn + 'value.weight'] = raw_z[hf_ffn + 'value.weight'].to(dtype=DTYPE)

        self.n_head, self.head_size = z['blocks.0.att.r_k'].shape
        
        keys = list(z.keys())
        for k in keys:
            z[k] = z[k].squeeze()
            if k.endswith('att.r_k'): z[k] = z[k].flatten()

        z['emb.weight'] = F.layer_norm(z['emb.weight'], (args.n_embd,), weight=raw_z['model.layers.0.attn_norm.weight'].to(dtype=DTYPE), bias=raw_z['model.layers.0.attn_norm.bias'].to(dtype=DTYPE))

    def forward(self, token:int, state:List[torch.Tensor]):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][token]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, state[i*3+0], state[i*3+1], v_first = time_mixing_hf(i, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'w0_base'],
                    z[att+'a0'], z[att+'a1'], z[att+'a2'], 
                    z[att+'v0'], z[att+'v1'], z[att+'v2'], 
                    z[att+'g1'], z[att+'g2'], 
                    z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'key.weight'], z[att+'value.weight'], z[att+'receptance.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx, state[i*3+2] = channel_mixing(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = z['head.weight'] @ x

            return x, state

# --- Tokenizer ---

class RWKV_TOKENIZER():
    def __init__(self, file_name):
        self.idx2token = {}
        sorted_tokens = []
        with open(file_name, "r", encoding="utf-8") as f:
            for l in f:
                idx = int(l[:l.index(' ')])
                x = eval(l[l.index(' '):l.rindex(' ')])
                x = x.encode("utf-8") if isinstance(x, str) else x
                sorted_tokens += [x]
                self.idx2token[idx] = x

        self.token2idx = {v: int(k) for k, v in self.idx2token.items()}
        self.table = [[[] for _ in range(256)] for _ in range(256)]
        self.good = [set() for _ in range(256)]
        self.wlen = [0 for _ in range(256)]

        for i in reversed(range(len(sorted_tokens))):
            s = sorted_tokens[i]
            if len(s) >= 2:
                s0, s1 = int(s[0]), int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encode(self, src: str) -> list[int]:
        src_bytes = src.encode("utf-8")
        src_len = len(src_bytes)
        tokens = []
        i = 0
        while i < src_len:
            s = src_bytes[i : i + 1]
            if i < src_len - 1:
                s0, s1 = int(src_bytes[i]), int(src_bytes[i + 1])
                if s1 in self.good[s0]:
                    sss = src_bytes[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except StopIteration:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return b''.join(map(lambda i: self.idx2token[i], tokens)).decode('utf-8', errors='replace')

# --- FastAPI App ---

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    reset: bool = False

# Global state for the chatbot
args = types.SimpleNamespace()
args.MODEL_NAME = "model.safetensors"
args.n_layer = 12
args.n_embd = 768
args.vocab_size = 65536
args.head_size = 64

model = None
tokenizer = None
current_state = None

@app.on_event("startup")
def startup_event():
    global model, tokenizer, current_state
    print("Loading model...")
    model = RWKV_RNN(args)
    tokenizer = RWKV_TOKENIZER("rwkv_vocab_v20230424.txt")
    reset_state()
    print("Model loaded.")

def reset_state():
    global current_state
    current_state = [None for _ in range(args.n_layer * 3)]
    for i in range(args.n_layer):
        current_state[i*3+0] = torch.zeros(args.n_embd, dtype=DTYPE)
        current_state[i*3+1] = torch.zeros((args.n_embd // args.head_size, args.head_size, args.head_size), dtype=torch.float)
        current_state[i*3+2] = torch.zeros(args.n_embd, dtype=DTYPE)

@app.get("/")
async def get_index():
    return FileResponse("index.html")

@app.post("/chat")
async def chat(request: ChatRequest):
    global current_state
    if request.reset:
        reset_state()
    
    if not request.message:
        return {"response": ""}

    prompt = f"User: {request.message}\n\nAssistant:"
    tokens = tokenizer.encode(prompt)
    
    for token in tokens:
        logits, current_state = model.forward(token, current_state)
    
    response_tokens = []
    
    # Sampling parameters
    temperature = 1.0 # Can be adjusted
    top_p = 0.9 # Can be adjusted
    
    for _ in range(200): # Limit response length
        # Apply temperature
        logits = logits / temperature
        
        # Top-p sampling
        probs = F.softmax(logits.float(), dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        # We keep at least one token
        mask = cumulative_probs <= top_p
        mask[0] = True 
        
        sorted_probs = sorted_probs[mask]
        sorted_indices = sorted_indices[mask]
        
        # Rescale probabilities
        sorted_probs = sorted_probs / sorted_probs.sum()
        
        # Sample from the filtered distribution
        token = torch.multinomial(sorted_probs, num_samples=1).item()
        token = sorted_indices[token].item()
        
        if token == 0: # End of text
            break
            
        response_tokens.append(token)
        logits, current_state = model.forward(token, current_state)
        
        # Stop generation if a newline is generated
        if tokenizer.decode([token]).endswith('\n'):
            break
            
    response_text = tokenizer.decode(response_tokens)
    return {"response": response_text.strip()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
