# train_addition_tiny_transformer.py
# Tiny conditional LM for addition: learns "A+B=" -> "C#"
# Usage examples:
#   python train_addition_tiny_transformer.py --mode random --samples 80000 --epochs 6
#   python train_addition_tiny_transformer.py --mode full --max-a 999 --max-b 999 --epochs 4

import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --------------------------
# Vocab and helpers
# --------------------------
VOCAB = list("0123456789+ =#")  # includes space ' ' (used as PAD)
stoi = {ch: i for i, ch in enumerate(VOCAB)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(VOCAB)

PAD_TOKEN = " "
PAD_ID = stoi[PAD_TOKEN]

def encode(s: str) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def decode(t: torch.Tensor) -> str:
    return "".join(itos[int(i)] for i in t)

def format_pair(a: int, b: int) -> Tuple[str, str]:
    """Return (prompt, target) for conditional training."""
    # Prompt is what we feed; target is what the model must generate
    prompt = f"{a}+{b}="
    target = f"{a+b}#"
    return prompt, target

# --------------------------
# Unified dataset (random or full)
# --------------------------
class AdditionDatasetUnified(Dataset):
    """
    mode='full'    -> all pairs 0..max_a × 0..max_b (inclusive)
    mode='random'  -> n_samples random pairs (optionally curriculum by digits)
    Trains conditionally: X="A+B=", Y="C#"
    """
    def __init__(self,
                 mode: str = "random",
                 max_a: int = 999,
                 max_b: int = 999,
                 n_samples: int = 80_000,
                 curriculum: bool = True,
                 seed: int = 42):
        assert mode in {"random", "full"}
        self.mode = mode
        self.max_a = max_a
        self.max_b = max_b
        self.curriculum = curriculum
        self.seed = seed

        self.examples: List[Tuple[int,int]] = []
        rng = random.Random(seed)

        if mode == "full":
            for a in range(max_a + 1):
                for b in range(max_b + 1):
                    self.examples.append((a, b))
        else:
            for _ in range(n_samples):
                if curriculum:
                    # Mix digit lengths (1..digits of max)
                    da = rng.randint(1, len(str(max_a)))
                    db = rng.randint(1, len(str(max_b)))
                    A = rng.randint(0, 10**da - 1)
                    B = rng.randint(0, 10**db - 1)
                else:
                    A = rng.randint(0, max_a)
                    B = rng.randint(0, max_b)
                self.examples.append((A, B))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        a, b = self.examples[idx]
        prompt, target = format_pair(a, b)
        x = encode(prompt)  # tokens for "A+B="
        y = encode(target)  # tokens for "C#"
        return x, y

# --------------------------
# Batch collate (pad + CORRECT label alignment)
# --------------------------
def collate_conditional(batch):
    """
    Each item: (x, y) with
      x = tokens("A+B=")
      y = tokens("C#")

    Build teacher-forced inputs so the LAST prompt token predicts y[0]:
      inp_i = concat(x_i, y_i[:-1])
      lab_i = -100 everywhere except the segment that should predict y:
              lab_i[(len(x_i)-1) : (len(x_i)-1 + len(y_i))] = y_i

    Then pad INP/LAB to the batch max length.
    """
    xs, ys = zip(*batch)
    B = len(xs)

    inps = []
    labs = []
    for x, y in zip(xs, ys):
        # input = prompt + answer[:-1]  (teacher forcing)
        if len(y) > 1:
            inp = torch.cat([x, y[:-1]], dim=0)
        else:
            inp = x.clone()
        # labels default to ignore
        lab = torch.full((inp.shape[0],), -100, dtype=torch.long)
        start = len(x) - 1                     # last prompt token predicts y[0]
        lab[start : start + len(y)] = y        # supervise the entire answer y
        inps.append(inp)
        labs.append(lab)

    maxlen = max(t.shape[0] for t in inps)
    INP = torch.full((B, maxlen), PAD_ID, dtype=torch.long)
    LAB = torch.full((B, maxlen), -100,   dtype=torch.long)

    for i, (inp, lab) in enumerate(zip(inps, labs)):
        INP[i, : inp.shape[0]] = inp
        LAB[i, : lab.shape[0]] = lab

    return INP, LAB

# --------------------------
# Tiny causal Transformer LM
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
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

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
        B, T = idx.size()
        assert T <= self.block_size, f"Sequence length {T} > block_size {self.block_size}"
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B,T,V)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=8):
        # Greedy generation until '#' or max_new_tokens
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
            if itos[int(next_id)] == '#':
                break
        return idx

# --------------------------
# Mini-evaluation utilities
# --------------------------
@torch.no_grad()
def mini_eval_exact(model, device, n_samples=2000, max_a=999, max_b=999) -> float:
    """
    Exact-string accuracy on a random sample of prompts.
    """
    model.eval()
    correct = 0
    for _ in range(n_samples):
        a = random.randint(0, max_a)
        b = random.randint(0, max_b)
        prompt, target = format_pair(a, b)
        ctx = encode(prompt).unsqueeze(0).to(device)
        out = model.generate(ctx, max_new_tokens=8)[0].cpu().numpy()
        pred = decode(out)
        if pred == prompt + target:
            correct += 1
    return correct / n_samples

# --------------------------
# Training
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["random", "full"], default="random")
    ap.add_argument("--max-a", type=int, default=999)
    ap.add_argument("--max-b", type=int, default=999)
    ap.add_argument("--samples", type=int, default=80_000, help="Used in random mode")
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
    ap.add_argument("--eval-every", type=int, default=1, help="Run mini-eval every N epochs (0 to disable)")
    ap.add_argument("--eval-samples", type=int, default=2000, help="Number of random prompts in mini-eval")
    args = ap.parse_args()

    # Determinism-ish
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Data
    ds = AdditionDatasetUnified(
        mode=args.mode,
        max_a=args.max_a,
        max_b=args.max_b,
        n_samples=args.samples,
        curriculum=args.curriculum,
        seed=args.seed,
    )

    # Ensure block size covers max prompt+answer length (<= 13 for 999+999=1998#)
    block_size = args.block_size
    if block_size < 13:
        print("Warning: block_size < 13; increasing to 13")
        block_size = 13

    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_conditional,
        pin_memory=True
    )

    # Model
    model = TinyTransformerLM(
        vocab_size=vocab_size,
        n_embd=args.embed,
        n_head=args.heads,
        n_layer=args.layers,
        block_size=block_size
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Train
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        sup_correct = 0        # token-level correct on supervised (non -100) positions
        sup_total = 0

        for Z, labels in dl:
            Z = Z.to(device)
            labels = labels.to(device)
            logits, loss = model(Z, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            # Accumulate loss
            running_loss += float(loss.item()) * Z.size(0)

            # Token-level accuracy on supervised positions
            with torch.no_grad():
                pred = logits.argmax(dim=-1)      # (B,T)
                mask = labels != -100             # supervised positions only
                sup_correct += (pred[mask] == labels[mask]).sum().item()
                sup_total   += mask.sum().item()

        avg_loss = running_loss / max(1, len(ds))
        tok_acc = (sup_correct / sup_total) * 100.0 if sup_total else 0.0
        print(f"epoch {epoch:02d}  loss {avg_loss:.6f}  token_acc {tok_acc:.2f}%")

        # Mini-eval (exact-string accuracy) every N epochs
        if args.eval_every > 0 and (epoch % args.eval_every == 0):
            acc = mini_eval_exact(model, device, n_samples=args.eval_samples,
                                  max_a=args.max_a, max_b=args.max_b)
            print(f"  mini-eval exact acc ({args.eval_samples} samples): {acc*100:.2f}%")

    # Save both formats
    weights_path = f"{args.save_prefix}_weights.pt"
    full_path    = f"{args.save_prefix}_full.pt"
    torch.save(model.state_dict(), weights_path)
    torch.save(model, full_path)
    print(f"Saved weights to {weights_path}")
    print(f"Saved full model to {full_path}")

    # Quick demo
    model.eval()
    demos = ["3+5=", "12+7=", "42+58=", "3+111=", "999+1="]
    print("\nDemos:")
    for p in demos:
        ctx = encode(p).unsqueeze(0).to(device)
        out = model.generate(ctx, max_new_tokens=8)[0].cpu().numpy()
        print(p, "→", decode(out))

if __name__ == "__main__":
    main()
