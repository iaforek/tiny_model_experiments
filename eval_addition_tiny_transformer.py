# eval_addition_tiny_transformer.py
# Exhaustive, batched, resumable evaluation for the conditional addition LM.
# Usage examples:
#   python eval_addition_tiny_transformer.py --ckpt addition_cond_weights.pt --state-dict
#   python eval_addition_tiny_transformer.py --ckpt addition_cond_full.pt
#   python eval_addition_tiny_transformer.py --ckpt addition_cond_weights.pt --state-dict --save-errors errs.tsv
#   python eval_addition_tiny_transformer.py --ckpt addition_cond_weights.pt --state-dict --resume-a 400 --save-state eval_state.json

import math, argparse, time, json, os, sys
import torch
import torch.nn as nn

# --------------------------
# Vocab & helpers (must match training)
# --------------------------
VOCAB = list("0123456789+ =#")
stoi = {ch:i for i,ch in enumerate(VOCAB)}
itos = {i:ch for ch,i in stoi.items()}
vocab_size = len(VOCAB)

def encode(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
def decode(t): return "".join(itos[int(i)] for i in t)

# --------------------------
# Model (must match training if using --state-dict)
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
        # Greedy generation until '#' or max_new_tokens (row-wise)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
            # Do NOT early-stop the whole batch; rows can finish at different steps
        return idx

# --------------------------
# Batched, resumable exhaustive eval with error printing
# --------------------------
@torch.no_grad()
def exhaustive_eval_print_errors(
    model,
    device,
    max_a=999,
    max_b=999,
    batch_size=8192,
    progress_step=100_000,
    resume_a=0,
    save_state_path=None,
    save_errors=None,
    max_errors_to_save=1000,
    fail_fast=False,
):
    model.eval()
    total = (max_a + 1) * (max_b + 1)
    correct = 0
    seen = 0
    t0 = time.time()
    PAD = stoi[' ']
    err_file = open(save_errors, "w", buffering=1) if save_errors else None
    saved_errs = 0

    def log_error(prompt, pred, expected):
        nonlocal saved_errs
        line = f"ERR  {prompt} -> {pred}   (expected {expected})"
        print(line)
        if err_file and saved_errs < max_errors_to_save:
            err_file.write(f"{prompt}\t{pred}\t{expected}\n")
            saved_errs += 1

    # Iterate row-wise on 'a' so we can checkpoint per-row
    for a in range(resume_a, max_a + 1):
        # Build all prompts for this 'a'
        row_prompts = [f"{a}+{b}=" for b in range(max_b + 1)]
        row_targets = [f"{a+b}#"  for b in range(max_b + 1)]

        # Sub-batch across 'b'
        s = 0
        while s < len(row_prompts):
            e = min(s + batch_size, len(row_prompts))
            batch_prompts = row_prompts[s:e]
            batch_targets = row_targets[s:e]

            # Pad prompts batch
            lens = [len(p) for p in batch_prompts]
            T = max(lens)
            X = torch.full((e - s, T), PAD, dtype=torch.long)
            for r, p in enumerate(batch_prompts):
                X[r, :len(p)] = encode(p)
            X = X.to(device)

            # Generate up to 8 new tokens (max result length is 5 for 1998#)
            out = model.generate(X, max_new_tokens=8).cpu()

            # Compare predictions
            for r in range(out.size(0)):
                pred = decode(out[r])
                expected = batch_prompts[r] + batch_targets[r]
                if pred == expected:
                    correct += 1
                else:
                    log_error(batch_prompts[r], pred, expected)
                    if fail_fast:
                        if err_file: err_file.close()
                        acc = correct / (seen + r + 1)
                        print(f"\nStopped on first error. Acc so far: {acc*100:.4f}%")
                        return
            s = e
            seen += (e - (s - (e - s)))  # increment as we go; simpler: seen = (a - resume_a)*(max_b+1) + e

        # End of row: update progress & checkpoint
        seen = (a - resume_a + 1) * (max_b + 1)
        done_global = (a + 1) * (max_b + 1)
        if done_global % progress_step == 0 or done_global == total:
            elapsed = time.time() - t0
            rate = done_global / max(1e-6, elapsed)
            acc = correct / done_global
            print(f"Progress: {done_global}/{total} ({done_global/total*100:.2f}%) "
                  f"acc={acc*100:.4f}%  ~{int(rate)} samp/s")

        if save_state_path:
            state = {
                "resume_a": a + 1,          # next a to evaluate
                "correct": correct,
                "done": done_global,
                "total": total,
                "acc": (correct / done_global),
                "timestamp": time.time(),
            }
            with open(save_state_path, "w") as f:
                json.dump(state, f)

    if err_file: err_file.close()
    final_acc = correct / total
    print(f"\nFinal accuracy 0..{max_a} + 0..{max_b}: {correct}/{total} = {final_acc*100:.4f}%")

# --------------------------
# CLI
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint (.pt)")
    ap.add_argument("--state-dict", action="store_true",
                    help="Load as weights-only state_dict; otherwise loads full pickled model.")
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--max-a", type=int, default=999)
    ap.add_argument("--max-b", type=int, default=999)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--progress-step", type=int, default=100000)
    ap.add_argument("--resume-a", type=int, default=0, help="Row index 'a' to resume from.")
    ap.add_argument("--save-state", type=str, default=None, help="Path to JSON checkpoint for resumable eval.")
    ap.add_argument("--save-errors", type=str, default=None, help="Optional TSV file to save first N errors.")
    ap.add_argument("--max-errors-to-save", type=int, default=1000)
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load model
    if args.state_dict:
        model = TinyTransformerLM(
            vocab_size=vocab_size,
            n_embd=args.embed,
            n_head=args.heads,
            n_layer=args.layers,
            block_size=max(args.block_size, 13)
        ).to(device)
        sd = torch.load(args.ckpt, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=True)
    else:
        model = torch.load(args.ckpt, map_location=device).to(device)
    model.eval()

    # Auto-resume from save-state (if present)
    resume_a = args.resume_a
    if args.save_state and os.path.exists(args.save_state):
        try:
            with open(args.save_state) as f:
                st = json.load(f)
                if "resume_a" in st:
                    resume_a = max(resume_a, int(st["resume_a"]))
                    print(f"Resuming from a={resume_a}")
        except Exception:
            pass

    # Quick sanity demos
    demos = ["3+5=", "12+7=", "42+58=", "3+111=", "999+1="]
    for p in demos:
        X = encode(p).unsqueeze(0).to(device)
        out = model.generate(X, max_new_tokens=8)[0].cpu().numpy()
        print(p, "→", decode(out))

    exhaustive_eval_print_errors(
        model, device,
        max_a=args.max_a, max_b=args.max_b,
        batch_size=args.batch_size,
        progress_step=args.progress_step,
        resume_a=resume_a,
        save_state_path=args.save_state,
        save_errors=args.save_errors,
        max_errors_to_save=args.max_errors_to_save,
        fail_fast=args.fail_fast,
    )

if __name__ == "__main__":
    main()
