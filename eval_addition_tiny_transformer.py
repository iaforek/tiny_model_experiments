# eval_addition_tiny_transformer.py
# Evaluate a trained tiny addition LM on A+B=C# sequences.
# Usage:
#   python eval_addition_tiny_transformer.py --ckpt tiny_addition_lm.pt --tests 5000 --max-digits 3

import math, random, argparse
import torch
import torch.nn as nn

# --------------------------
# Vocab and helpers (must match training)
# --------------------------
VOCAB = list("0123456789+ =#")
stoi = {ch:i for i,ch in enumerate(VOCAB)}
itos = {i:ch for ch,i in stoi.items()}
vocab_size = len(VOCAB)

def encode(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
def decode(t): return "".join(itos[int(i)] for i in t)

def make_example(n_digits_a=3, n_digits_b=3):
    A = random.randint(0, 10**n_digits_a - 1)
    B = random.randint(0, 10**n_digits_b - 1)
    return f"{A}+{B}={A+B}#"

def make_prompt(A, B):
    return f"{A}+{B}="

def make_target(A, B):
    return f"{A}+{B}={A+B}#"

# --------------------------
# Model (identical to training)
# --------------------------
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
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)),
                                              targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=16):
        # Greedy generation until '#' or max_new_tokens
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
# Evaluation
# --------------------------
def count_digits(n: int) -> int:
    return 1 if n == 0 else len(str(abs(n)))

@torch.no_grad()
def eval_random(model, device, tests=5000, max_digits=3):
    model.eval()
    correct = 0
    per_bucket = {(1,1): [0,0], (1,2): [0,0], (1,3): [0,0],
                  (2,1): [0,0], (2,2): [0,0], (2,3): [0,0],
                  (3,1): [0,0], (3,2): [0,0], (3,3): [0,0]}

    for _ in range(tests):
        da = random.randint(1, max_digits)
        db = random.randint(1, max_digits)
        A = random.randint(0, 10**da - 1)
        B = random.randint(0, 10**db - 1)
        prompt = make_prompt(A,B)
        target = make_target(A,B)

        ctx = encode(prompt).unsqueeze(0).to(device)
        out = model.generate(ctx, max_new_tokens=16)[0].cpu().numpy()
        pred = decode(out)

        ok = (pred == target)
        correct += int(ok)

        key = (count_digits(A), count_digits(B))
        if key in per_bucket:
            per_bucket[key][0] += int(ok)
            per_bucket[key][1] += 1

    overall = correct / tests
    per_bucket_acc = {k: (v[0]/v[1] if v[1] else 0.0) for k,v in per_bucket.items()}
    return overall, per_bucket_acc

@torch.no_grad()
def eval_edges(model, device):
    model.eval()
    cases = [
        (0,0), (0,9), (9,0), (9,9),
        (10,0), (0,10), (10,9), (9,10),
        (99,1), (1,99), (99,99),
        (100,0), (0,100), (100,1), (1,100), (100,99), (99,100),
        (999,0), (0,999), (999,1), (1,999), (999,999)
    ]
    results = []
    ok_count = 0
    for A,B in cases:
        prompt = make_prompt(A,B)
        target = make_target(A,B)
        ctx = encode(prompt).unsqueeze(0).to(device)
        out = model.generate(ctx, max_new_tokens=16)[0].cpu().numpy()
        pred = decode(out)
        ok = (pred == target)
        ok_count += int(ok)
        results.append((prompt, pred, target, ok))
    return ok_count, len(cases), results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (e.g., tiny_addition_lm.pt)")
    parser.add_argument("--tests", type=int, default=5000, help="Number of random tests")
    parser.add_argument("--max-digits", type=int, default=3, help="Max digits per operand in random tests")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--embed", type=int, default=64)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Build model with the same hyperparams as training (override via args if needed)
    model = TinyTransformerLM(
        vocab_size=vocab_size,
        n_embd=args.embed,
        n_head=args.heads,
        n_layer=args.layers,
        block_size=args.block_size
    ).to(device)

    # Load weights
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}")

    # Quick sanity demos
    demos = ["3+5=", "12+7=", "42+58=", "3+111=", "999+1="]
    print("\n--- Demo generations ---")
    for p in demos:
        out = model.generate(encode(p).unsqueeze(0).to(device), max_new_tokens=12)[0].cpu().numpy()
        print(p, "→", decode(out))

    # Random evaluation
    overall, per_bucket = eval_random(model, device, tests=args.tests, max_digits=args.max_digits)
    print(f"\n--- Random test accuracy ({args.tests} samples, up to {args.max_digits} digits) ---")
    print(f"Overall exact-string accuracy: {overall*100:.2f}%")
    print("By operand digit count (A_digits, B_digits):")
    for k in sorted(per_bucket.keys()):
        print(f"  {k}: {per_bucket[k]*100:.2f}%")

    # Edge cases
    ok, total, results = eval_edges(model, device)
    print(f"\n--- Edge-case set ---")
    print(f"Edge-case exact-string accuracy: {ok}/{total} = {ok/total*100:.2f}%")
    for prompt, pred, target, okflag in results:
        status = "OK " if okflag else "ERR"
        print(f"[{status}] {prompt}  ->  {pred}  (expected {target})")

if __name__ == "__main__":
    main()
