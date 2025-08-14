# train_addition_tiny_transformer.py
# Requires: Python 3.8+, torch, numpy
# Tip: conda install pytorch cpuonly -c pytorch (or a GPU build if you have one)

import math, random, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --------------------------
# 1) Vocab and tokenization
# --------------------------
VOCAB = list("0123456789+ =#")
stoi = {ch:i for i,ch in enumerate(VOCAB)}
itos = {i:ch for ch,i in stoi.items()}
vocab_size = len(VOCAB)

def encode(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
def decode(t): return "".join(itos[int(i)] for i in t)

def format_example(a,b):
    s = f"{a}+{b}={a+b}#"
    print(s)
    return s

# --------------------------------
# 2) Synthetic dataset (generator)
# --------------------------------
def make_example(n_digits_a=2, n_digits_b=2):
    # Random A,B with specified digit-width (no leading spaces, allow spaces in between)
    A = random.randint(0, 10**n_digits_a - 1)
    B = random.randint(0, 10**n_digits_b - 1)
    s = format_example(A, B)
    return s

class AdditionDataset(Dataset):
    def __init__(self, n_samples=50_000, max_digits=3, curriculum=False, seed=42):
        rng = random.Random(seed)
        self.samples = []
        for _ in range(n_samples):
            if curriculum:
                # gradually mix 1..max_digits
                dA = rng.randint(1, max_digits)
                dB = rng.randint(1, max_digits)
            else:
                dA = max_digits
                dB = max_digits
            self.samples.append(make_example(dA, dB))
        self.max_len = max(len(s) for s in self.samples)
        # left-pad sequences with spaces so they align (optional)
        self.samples = [s.ljust(self.max_len, ' ') for s in self.samples]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = encode(s[:-1])   # input up to last char
        y = encode(s[1:])    # predict next char
        return x, y
    

class FullDataset(Dataset):
    def __init__(self):
        self.samples = []
        for a in range(1000):
            for b in range(1000):
                self.samples.append(format_example(a, b))
        self.max_len = max(len(s) for s in self.samples)
        # left-pad sequences with spaces so they align (optional)
        self.samples = [s.ljust(self.max_len, ' ') for s in self.samples]
    
    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = encode(s[:-1])   # input up to last char
        y = encode(s[1:])    # predict next char
        return x, y


# ------------------------------------
# 3) Tiny causal Transformer language model
# ------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1,1,block_size,block_size))

    def forward(self, x):
        B,T,C = x.size()
        k = self.key(x).view(B,T,self.n_head,C//self.n_head).transpose(1,2)   # B,heads,T,d
        q = self.query(x).view(B,T,self.n_head,C//self.n_head).transpose(1,2) # B,heads,T,d
        v = self.value(x).view(B,T,self.n_head,C//self.n_head).transpose(1,2) # B,heads,T,d
        att = (q @ k.transpose(-2,-1)) / math.sqrt(k.size(-1))                 # B,heads,T,T
        att = att.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        y = att @ v                                                            # B,heads,T,d
        y = y.transpose(1,2).contiguous().view(B,T,C)
        return self.proj(y)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, n_embd=64, n_head=2, n_layer=2, block_size=32):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B,T = idx.size()
        assert T <= self.block_size
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)                    # (B,T,vocab)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)),
                                              targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=16):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.argmax(probs, dim=-1, keepdim=True)  # greedy
            idx = torch.cat([idx, next_id], dim=1)
            if itos[int(next_id)] == '#':
                break
        return idx

# --------------------------
# 4) Train
# --------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    block_size = 32
    model = TinyTransformerLM(vocab_size, n_embd=64, n_head=2, n_layer=2, block_size=block_size).to(device)

    # Data
    # train_ds = AdditionDataset(n_samples=40_000, max_digits=3, curriculum=True)
    train_ds = FullDataset()
    val_ds   = AdditionDataset(n_samples=2_000,  max_digits=3, curriculum=True, seed=777)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    best_val = float("inf")

    def evaluate(loader):
        model.eval()
        total = 0.0; count = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device); y = y.to(device)
                _, loss = model(x, y)
                total += float(loss) * x.size(0)
                count += x.size(0)
        return total / count

    for epoch in range(10):
        model.train()
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            _, loss = model(x, y)
            opt.zero_grad(); loss.backward(); opt.step()
        val_loss = evaluate(val_loader)
        print(f"epoch {epoch:02d}  val_loss {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "full_addition_lm.pt")

    # Quick demo
    model.eval()
    prompts = ["3+5=", "12+7=", "42+58=", "3+111="]
    for p in prompts:
        ctx = encode(p).unsqueeze(0).to(device)
        out = model.generate(ctx, max_new_tokens=8)
        print(p, "→", decode(out[0].cpu().numpy()))

if __name__ == "__main__":
    main()
