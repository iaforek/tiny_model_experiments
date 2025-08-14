# eval_addition_tiny_transformer.py
import math, argparse, time
import torch
import torch.nn as nn

VOCAB = list("0123456789+ =#")
stoi = {ch:i for i,ch in enumerate(VOCAB)}
itos = {i:ch for ch,i in stoi.items()}
vocab_size = len(VOCAB)
def encode(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
def decode(t): return "".join(itos[int(i)] for i in t)

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd); self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd); self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1,1,block_size,block_size))
    def forward(self, x):
        B,T,C = x.size()
        k = self.key(x).view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q = self.query(x).view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = self.value(x).view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        return self.proj(y)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.GELU(), nn.Linear(4*n_embd, n_embd))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=32):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size
    def forward(self, idx):
        B,T = idx.size()
        pos = torch.arange(0,T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks: x = blk(x)
        x = self.ln_f(x)
        return self.head(x)
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
            if itos[int(next_id)] == '#': break
        return idx

@torch.no_grad()
def exhaustive_eval(model, device, max_a=999, max_b=999, progress_step=50000):
    total = (max_a+1)*(max_b+1)
    correct = 0
    t0 = time.time()
    for a in range(max_a+1):
        for b in range(max_b+1):
            prompt = f"{a}+{b}="
            target = f"{a+b}#"
            out = model.generate(encode(prompt).unsqueeze(0).to(device), max_new_tokens=8)[0].cpu().numpy()
            pred = decode(out)
            if pred == prompt + target:  # NOTE: pred includes the prompt tokens we fed
                correct += 1
        done = (a+1)*(max_b+1)
        if done % progress_step == 0:
            rate = done / max(1,(time.time()-t0))
            print(f"Progress: {done}/{total} ({done/total*100:.2f}%) acc={correct/done*100:.4f}%  ~{rate:.0f} samp/s")
    print(f"\nExhaustive accuracy 0..{max_a} + 0..{max_b}: {correct}/{total} = {correct/total*100:.4f}%")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--state-dict", action="store_true",
                    help="Set if ckpt is weights-only (state_dict). Otherwise loads full model.")
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--max-a", type=int, default=999)
    ap.add_argument("--max-b", type=int, default=999)
    ap.add_argument("--progress-step", type=int, default=50000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if args.state_dict:
        model = TinyTransformerLM(vocab_size, args.embed, args.heads, args.layers, max(args.block_size,13)).to(device)
        sd = torch.load(args.ckpt, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
        model.load_state_dict(sd, strict=True)
    else:
        model = torch.load(args.ckpt, map_location=device).to(device)
    model.eval()

    # sanity demos
    for p in ["3+5=", "12+7=", "42+58=", "3+111=", "999+1="]:
        out = model.generate(encode(p).unsqueeze(0).to(device), max_new_tokens=8)[0].cpu().numpy()
        print(p, "→", decode(out))

    exhaustive_eval(model, device, args.max_a, args.max_b, args.progress_step)

if __name__ == "__main__":
    main()
