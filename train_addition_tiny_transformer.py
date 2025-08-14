# train_addition_tiny_transformer.py
import math, random, argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- Vocab ----------
VOCAB = list("0123456789+ =#")  # includes space for pad
stoi = {ch:i for i,ch in enumerate(VOCAB)}
itos = {i:ch for ch,i in stoi.items()}
vocab_size = len(VOCAB)
PAD_ID = stoi[" "]

def encode(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
def decode(t): return "".join(itos[int(i)] for i in t)

def format_pair(a:int,b:int):
    prompt = f"{a}+{b}="
    target = f"{a+b}#"
    return prompt, target

# ---------- Unified Dataset (random or full) ----------
class AdditionDatasetUnified(Dataset):
    def __init__(self, mode="random", max_a=999, max_b=999, n_samples=80_000, curriculum=True, seed=42):
        assert mode in {"random","full"}
        self.examples = []
        rng = random.Random(seed)
        if mode == "full":
            for a in range(max_a+1):
                for b in range(max_b+1):
                    self.examples.append((a,b))
        else:
            for _ in range(n_samples):
                if curriculum:
                    da = rng.randint(1, len(str(max_a)))
                    db = rng.randint(1, len(str(max_b)))
                    A = rng.randint(0, 10**da - 1)
                    B = rng.randint(0, 10**db - 1)
                else:
                    A = rng.randint(0, max_a); B = rng.randint(0, max_b)
                self.examples.append((A,B))

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        a,b = self.examples[idx]
        x_str, y_str = format_pair(a,b)     # x="A+B=", y="C#"
        return encode(x_str), encode(y_str)

# ---------- Collate: pad per-batch + mask loss to answer only ----------
def collate_conditional(batch):
    xs, ys = zip(*batch)
    B = len(xs)
    len_x = [len(x) for x in xs]; len_y = [len(y) for y in ys]
    max_x = max(len_x); max_y = max(len_y)
    T = max_x + max_y

    Z = torch.full((B,T), PAD_ID, dtype=torch.long)      # input tokens
    labels = torch.full((B,T), -100, dtype=torch.long)   # ignore by default

    # place x then y; loss is only on y positions
    for i,(x,y) in enumerate(zip(xs,ys)):
        lx, ly = len_x[i], len_y[i]
        Z[i, :lx] = x
        Z[i, lx:lx+ly] = y
        # prompt region ignored (-100); answer region filled after shift

    # one-step shift of inputs for next-token targets
    Z_shift = torch.roll(Z, shifts=-1, dims=1)
    # Set labels for answer region = shifted Z, but never learn on PAD
    for i,(x,y) in enumerate(zip(xs,ys)):
        lx, ly = len_x[i], len_y[i]
        tgt_slice = slice(lx, lx+ly)  # answer region
        labels[i, tgt_slice] = Z_shift[i, tgt_slice]
    labels[Z_shift == PAD_ID] = -100
    return Z, labels

# ---------- Tiny causal Transformer ----------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
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
    def forward(self, idx, labels=None):
        B,T = idx.size()
        assert T <= self.block_size
        pos = torch.arange(0,T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks: x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)),
                                              labels.view(-1), ignore_index=-100)
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
            if itos[int(next_id)] == '#': break
        return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["random","full"], default="random")
    ap.add_argument("--max-a", type=int, default=999)
    ap.add_argument("--max-b", type=int, default=999)
    ap.add_argument("--samples", type=int, default=80000)  # used in random mode
    ap.add_argument("--curriculum", action="store_true", default=True)
    ap.add_argument("--no-curriculum", dest="curriculum", action="store_false")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save-prefix", type=str, default="addition_cond")
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ds = AdditionDatasetUnified(args.mode, args.max_a, args.max_b, args.samples, args.curriculum, args.seed)
    # Worst-case prompt+answer length = 13
    block_size = max(args.block_size, 13)

    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0,
                    collate_fn=collate_conditional, pin_memory=True)

    model = TinyTransformerLM(vocab_size, args.embed, args.heads, args.layers, block_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0; count = 0
        for Z, labels in dl:
            Z = Z.to(device); labels = labels.to(device)
            _, loss = model(Z, labels)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            running += float(loss.item()) * Z.size(0); count += Z.size(0)
        print(f"epoch {epoch:02d}  loss {running/max(1,count):.6f}")

    # Save both formats
    wpath = f"{args.save_prefix}_weights.pt"
    fpath = f"{args.save_prefix}_full.pt"
    torch.save(model.state_dict(), wpath)
    torch.save(model, fpath)
    print(f"Saved weights to {wpath}")
    print(f"Saved full model to {fpath}")

    # Quick demo
    model.eval()
    for p in ["3+5=", "12+7=", "42+58=", "3+111=", "999+1="]:
        ctx = encode(p).unsqueeze(0).to(device)
        out = model.generate(ctx, max_new_tokens=8)[0].cpu().numpy()
        print(p, "→", decode(out))

if __name__ == "__main__":
    main()
