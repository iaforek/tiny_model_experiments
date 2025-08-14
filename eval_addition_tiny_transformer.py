# eval_addition_exhaustive.py
# Exhaustively evaluate a trained tiny addition LM on all pairs 0..999 + 0..999.
# Usage examples:
#   python eval_addition_exhaustive.py --ckpt tiny_addition_lm.pt --state-dict
#   python eval_addition_exhaustive.py --ckpt tiny_addition_lm_full.pt

import math, argparse, time
import torch
import torch.nn as nn

# --- Vocab & helpers (must match training) ---
VOCAB = list("0123456789+ =#")
stoi = {ch:i for i,ch in enumerate(VOCAB)}
itos = {i:ch for ch,i in stoi.items()}
vocab_size = len(VOCAB)

def encode(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
def decode(t): return "".join(itos[int(i)] for i in t)

def make_prompt(A, B): return f"{A}+{B}="
def make_target(A, B): return f"{A}+{B}={A+B}#"

# --- Tiny Transformer definition (must match training if using --state-dict) ---
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
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.argmax(probs, dim=-1, keepdim=True)  # greedy
            idx = torch.cat([idx, next_id], dim=1)
            if itos[int(next_id)] == '#':
                break
        return idx

# --- Exhaustive evaluation ---
@torch.no_grad()
def exhaustive_eval(model, device, max_a=999, max_b=999, show_every=50000, save_first_n_errors=50, error_outfile=None):
    model.eval()
    total = (max_a + 1) * (max_b + 1)
    correct = 0
    errors = []
    t0 = time.time()

    # Precompute a safe max_new_tokens: worst-case "999+999=1998#" is 13 chars;
    # prompts like "A+B=" length varies; we cap extra generation at 12 to cover result + '#'.
    MAX_NEW = 12

    for a in range(max_a + 1):
        for b in range(max_b + 1):
            prompt = f"{a}+{b}="
            target = f"{a}+{b}={a+b}#"

            ctx = encode(prompt).unsqueeze(0).to(device)
            out = model.generate(ctx, max_new_tokens=MAX_NEW)[0].cpu().numpy()
            pred = decode(out)

            if pred == target:
                correct += 1
            else:
                if len(errors) < save_first_n_errors:
                    errors.append((prompt, pred, target))

        # progress
        done = (a + 1) * (max_b + 1)
        if show_every and done % show_every == 0:
            rate = done / max(1, (time.time() - t0))
            print(f"Progress: {done}/{total} ({done/total*100:.2f}%)  acc_so_far={correct/done*100:.2f}%  speed ~{rate:.0f} samples/s")

    acc = correct / total
    print(f"\nExhaustive accuracy 0..{max_a} + 0..{max_b}: {correct}/{total} = {acc*100:.4f}%")

    if errors:
        print(f"\nFirst {len(errors)} errors:")
        for p, pred, tgt in errors:
            print(f"  {p} -> {pred}   (expected {tgt})")

        # Optional: save to file
        if error_outfile:
            with open(error_outfile, "w") as f:
                for p, pred, tgt in errors:
                    f.write(f"{p}\t{pred}\t{tgt}\n")
            print(f"Saved first {len(errors)} errors to {error_outfile}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt)")
    ap.add_argument("--state-dict", action="store_true",
                    help="Load as weights-only state_dict; if not set, loads full pickled model.")
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--max-a", type=int, default=999)
    ap.add_argument("--max-b", type=int, default=999)
    ap.add_argument("--save-errors", type=str, default=None,
                    help="Optional path to save first N errors (tab-separated)")
    ap.add_argument("--n-errors", type=int, default=50, help="How many errors to save/show")
    ap.add_argument("--progress-step", type=int, default=50000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if args.state_dict:
        # Recreate architecture exactly as during training
        model = TinyTransformerLM(
            vocab_size=vocab_size,
            n_embd=args.embed,
            n_head=args.heads,
            n_layer=args.layers,
            block_size=args.block_size
        ).to(device)
        sd = torch.load(args.ckpt, map_location=device)
        # Allow plain dict or {"state_dict":...}
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=True)
    else:
        # Load full pickled model
        model = torch.load(args.ckpt, map_location=device)
        model.to(device)
        model.eval()

    # Quick sanity demo
    demos = ["3+5=", "12+7=", "42+58=", "3+111=", "999+1="]
    print("\nSanity demos:")
    for p in demos:
        out = model.generate(encode(p).unsqueeze(0).to(device), max_new_tokens=12)[0].cpu().numpy()
        print(p, "→", decode(out))

    # Run exhaustive eval
    exhaustive_eval(
        model, device,
        max_a=args.max_a, max_b=args.max_b,
        show_every=args.progress_step,
        save_first_n_errors=args.n_errors,
        error_outfile=args.save_errors
    )

if __name__ == "__main__":
    main()