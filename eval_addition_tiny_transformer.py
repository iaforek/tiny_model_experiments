# eval_addition_tiny_transformer.py
# Exhaustive or random evaluation (batched, with error logging and optional resume).
# Examples:
#   FULL (exhaustive, 1,000,000 pairs)
#     python eval_addition_tiny_transformer.py --ckpt addition_cond_weights.pt --state-dict --mode full
#   RANDOM (e.g., 50k random pairs)
#     python eval_addition_tiny_transformer.py --ckpt addition_cond_weights.pt --state-dict --mode random --samples 50000

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
PAD_ID  = stoi[' ']
EOS_ID  = stoi['#']
PLUS_ID = stoi['+']
EQ_ID   = stoi['=']

def encode(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
def decode(t): return "".join(itos[int(i)] for i in t)

def trim_to_eos(ids_row: torch.Tensor) -> torch.Tensor:
    hits = (ids_row == EOS_ID).nonzero(as_tuple=False)
    if hits.numel() == 0:
        return ids_row
    end = int(hits[0]) + 1
    return ids_row[:end]

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
    def generate(self, idx, max_new_tokens=8, stop_id=EOS_ID):
        """
        Greedy generation with simple constraints:
        - Disallow PAD (' '), '+' and '=' in generated tail.
        - Stop early once '#' (EOS) is produced for a row.
        """
        B = idx.size(0)
        finished = torch.zeros(B, dtype=torch.bool, device=idx.device)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)[:, -1, :]  # (B, vocab)

            # Ban tokens that should never appear in the answer tail
            logits[:, PAD_ID]  = -1e9
            logits[:, PLUS_ID] = -1e9
            logits[:, EQ_ID]   = -1e9

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.argmax(probs, dim=-1)  # (B,)

            # For sequences already finished, force EOS to be repeated (no change)
            next_id = torch.where(finished, torch.full_like(next_id, stop_id), next_id)

            # Append
            idx = torch.cat([idx, next_id.unsqueeze(1)], dim=1)

            # Update finished mask
            finished |= (next_id == stop_id)
            if finished.all():
                break

        return idx

# --------------------------
# FULL (exhaustive) eval with error printing + resume
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
    PAD = stoi[' ']
    t0 = time.time()
    err_file = open(save_errors, "w", buffering=1) if save_errors else None
    saved_errs = 0

    def log_error(prompt, pred, expected):
        nonlocal saved_errs
        line = f"ERR  {prompt} -> {pred}   (expected {expected})"
        print(line)
        if err_file and saved_errs < max_errors_to_save:
            err_file.write(f"{prompt}\t{pred}\t{expected}\n")
            saved_errs += 1

    for a in range(resume_a, max_a + 1):
        # Build this row of prompts/targets
        row_prompts = [f"{a}+{b}=" for b in range(max_b + 1)]
        row_targets = [f"{a+b}#" for b in range(max_b + 1)]

        s = 0
        while s < len(row_prompts):
            e = min(s + batch_size, len(row_prompts))
            batch_prompts = row_prompts[s:e]
            batch_targets = row_targets[s:e]

            # --- SAFETY CHECK: no leading spaces in prompts
            for p in batch_prompts:
                assert not p or p[0] != ' ', f"Prompt has leading space: {repr(p)}"

            # Pad prompts
            lens = [len(p) for p in batch_prompts]
            T = max(lens)
            X = torch.full((e - s, T), PAD, dtype=torch.long)
            for r, p in enumerate(batch_prompts):
                t = encode(p)
                X[r, :len(p)] = t
            X = X.to(device)

            # Generate (includes prompt + generated tail)
            out = model.generate(X, max_new_tokens=8).cpu()

            # Compare
            for r in range(out.size(0)):
                pred = decode(trim_to_eos(out[r]))
                expected = batch_prompts[r] + batch_targets[r]
                if pred == expected:
                    correct += 1
                else:
                    log_error(batch_prompts[r], pred, expected)
                    if fail_fast:
                        if err_file: err_file.close()
                        done = a * (max_b + 1) + (s + r + 1)
                        acc = correct / done
                        print(f"\nStopped on first error. Acc so far: {acc*100:.4f}%")
                        return
            s = e

        # Progress + checkpoint per-row
        done_global = (a + 1) * (max_b + 1)
        if done_global % progress_step == 0 or done_global == total:
            elapsed = time.time() - t0
            rate = done_global / max(1e-6, elapsed)
            acc = correct / done_global
            print(f"Progress: {done_global}/{total} ({done_global/total*100:.2f}%) acc={acc*100:.4f}%  ~{int(rate)} samp/s")

        if save_state_path:
            state = {
                "mode": "full",
                "resume_a": a + 1,  # next a
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
# RANDOM eval with error printing (non-resumable)
# --------------------------
@torch.no_grad()
def random_eval_print_errors(
    model,
    device,
    samples=50000,
    max_a=999,
    max_b=999,
    batch_size=8192,
    progress_step=100_000,
    save_errors=None,
    max_errors_to_save=1000,
    fail_fast=False,
    seed=1234,
):
    import random
    model.eval()
    PAD = stoi[' ']
    err_file = open(save_errors, "w", buffering=1) if save_errors else None
    saved_errs = 0
    correct = 0
    t0 = time.time()
    random.seed(seed)

    def log_error(prompt, pred, expected):
        nonlocal saved_errs
        line = f"ERR  {prompt} -> {pred}   (expected {expected})"
        print(line)
        if err_file and saved_errs < max_errors_to_save:
            err_file.write(f"{prompt}\t{pred}\t{expected}\n")
            saved_errs += 1

    # Build all random pairs up-front for determinism / repeatability
    pairs = [(random.randint(0, max_a), random.randint(0, max_b)) for _ in range(samples)]

    # Batched loop
    s = 0
    while s < samples:
        e = min(s + batch_size, samples)
        batch_prompts = [f"{a}+{b}=" for (a,b) in pairs[s:e]]
        batch_targets = [f"{a+b}#" for (a,b) in pairs[s:e]]

        # --- SAFETY CHECK: no leading spaces in prompts
        for p in batch_prompts:
            assert not p or p[0] != ' ', f"Prompt has leading space: {repr(p)}"

        lens = [len(p) for p in batch_prompts]
        T = max(lens)
        X = torch.full((e - s, T), PAD, dtype=torch.long)
        for r, p in enumerate(batch_prompts):
            t = encode(p)
            X[r, :len(p)] = t  # LEFT-ALIGN, RIGHT-PAD (matches training)
        X = X.to(device)

        out = model.generate(X, max_new_tokens=8).cpu()

        for r in range(out.size(0)):
            pred = decode(trim_to_eos(out[r]))
            expected = batch_prompts[r] + batch_targets[r]
            if pred == expected:
                correct += 1
            else:
                log_error(batch_prompts[r], pred, expected)
                if fail_fast:
                    if err_file: err_file.close()
                    acc = correct / (s + r + 1)
                    print(f"\nStopped on first error. Acc so far: {acc*100:.4f}%")
                    return
        s = e

        if s % progress_step == 0 or s == samples:
            elapsed = time.time() - t0
            rate = s / max(1e-6, elapsed)
            acc = correct / s
            print(f"Progress: {s}/{samples} ({s/samples*100:.2f}%) acc={acc*100:.4f}%  ~{int(rate)} samp/s")

    if err_file: err_file.close()
    final_acc = correct / samples
    print(f"\nFinal random-eval accuracy ({samples} samples): {final_acc*100:.4f}%")

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
    ap.add_argument("--mode", choices=["full","random"], default="full",
                    help="full = evaluate all 0..max-a x 0..max-b; random = sample uniformly.")
    ap.add_argument("--max-a", type=int, default=999)
    ap.add_argument("--max-b", type=int, default=999)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--progress-step", type=int, default=100000)
    # full-mode resume
    ap.add_argument("--resume-a", type=int, default=0, help="(full mode) Row index 'a' to resume from.")
    ap.add_argument("--save-state", type=str, default=None, help="(full mode) Path to JSON checkpoint for resumable eval.")
    # errors
    ap.add_argument("--save-errors", type=str, default=None, help="Optional TSV file to save first N errors.")
    ap.add_argument("--max-errors-to-save", type=int, default=1000)
    ap.add_argument("--fail-fast", action="store_true")
    # random-mode options
    ap.add_argument("--samples", type=int, default=50000, help="(random mode) Number of random pairs to evaluate.")
    ap.add_argument("--seed", type=int, default=1234, help="(random mode) RNG seed for reproducibility.")
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

    # Quick sanity demos
    demos = ["3+5=", "12+7=", "42+58=", "3+111=", "999+1="]
    for p in demos:
        X = encode(p).unsqueeze(0).to(device)
        out = model.generate(X, max_new_tokens=8)[0].cpu()
        print(p, "→", decode(trim_to_eos(out)))

    # Auto-resume (full mode only) if save-state present
    resume_a = args.resume_a
    if args.mode == "full" and args.save_state and os.path.exists(args.save_state):
        try:
            with open(args.save_state) as f:
                st = json.load(f)
                if st.get("mode") == "full" and "resume_a" in st:
                    resume_a = max(resume_a, int(st["resume_a"]))
                    print(f"Resuming from a={resume_a}")
        except Exception:
            pass

    # Run selected mode
    if args.mode == "full":
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
    else:
        random_eval_print_errors(
            model, device,
            samples=args.samples,
            max_a=args.max_a, max_b=args.max_b,
            batch_size=args.batch_size,
            progress_step=args.progress_step,
            save_errors=args.save_errors,
            max_errors_to_save=args.max_errors_to_save,
            fail_fast=args.fail_fast,
            seed=args.seed,
        )

if __name__ == "__main__":
    main()
