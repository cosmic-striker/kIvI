import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
block_size = 64
batches = 32
train_loop = 5000
learning_rate = 5e-4
train_split = 0.8
debug_loop = 500
embeds = 512
heads = 4
layers = 8
dropout = 0.2

chars = ""
with open('persion.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(set(text))
print(chars)
vocab_size = len(chars)
print(vocab_size)

str2int = { ch:i for i,ch in enumerate(chars) }
int2str = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [ str2int[c] for c in s ]
decode = lambda l: ''.join([ int2str[i] for i in l ])
data = torch.tensor(encode(text), dtype=torch.long).to(device)
print(data[:100])

n = int(train_split*len(data))
train = data[:n]
valid = data[n:]

def get_batch(split):
    data = train if split=='train' else valid
    ix = torch.randint(len(data) - block_size, (batches,))
    # print(ix)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

x, y = get_batch('train')
print('inputs:')
print(x)
print('target:')
print(y)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embeds, head_size, bias=False)
        self.query = nn.Linear(embeds, head_size, bias=False)
        self.value = nn.Linear(embeds, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-.5
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, embeds)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embeds, 4*embeds),
            nn.ReLU(),
            nn.Linear(4*embeds, embeds),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, embeds, heads):
        super().__init__()
        head_size = embeds // heads
        self.sa = MultiHeadAttention(heads, head_size)
        self.ffwd = FeedForward(embeds)
        self.ln1 = nn.LayerNorm(embeds)
        self.ln2 = nn.LayerNorm(embeds)
    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

class Kiwi(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embeds)
        self.position_embedding = nn.Embedding(block_size, embeds)
        self.blocks = nn.Sequential(*[Block(embeds, heads=heads) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(embeds)
        self.head = nn.Linear(embeds, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape

        tok = self.token_embedding(index)
        pos = self.position_embedding(torch.arange(T, device=device))
        x = tok + pos
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
        
    def generate(self, index, max_new_tokens):
        for iter in range(max_new_tokens):
            index_crop = index[:, -block_size:]
            logits, loss = self.forward(index_crop)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
            print(f"\r<{'*' * int(10 * iter/max_new_tokens)}{' ' * (10 - int(10*iter/max_new_tokens))}>", end='', flush=False)
        print("\r<**********>")
        return index

model = Kiwi(vocab_size).to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(debug_loop)
        for k in range(debug_loop):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train(model, train_loop, debug_loop):
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in range(train_loop):
        if iter % debug_loop == 0:
            losses = estimate_loss(model)
            print(f'step: {iter}, train loss: {losses['train']:.6f}, valid loss: {losses['val']:.6f}')
        xb, yb = get_batch('train')
        logits, loss = model.forward(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    print(loss.item())

def generate(model, prompt=None):
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device) if prompt is None else torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated = decode(model.generate(ctx, max_new_tokens=1000)[0].tolist())
    print(generated)

def export_model(model, file):
    with open(file, 'wb') as f:
        pickle.dump(model, f)

def import_model(model, file):
    with open(file, 'rb'):
        return pickle.load(f)
