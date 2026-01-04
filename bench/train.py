# train.py
# nanoGPT-style training loop with:
# - MHA/GQA/MQA support via n_kv_head passed into GPTConfig
# - use_rope flag passed into GPTConfig
# - norm_type flag passed into GPTConfig (layernorm vs rmsnorm)
#
# NEW checkpoint naming policy (your requested canonical form):
#   gqa_h{n_head}_kv{n_kv_head}_{rope|seq}_{layernorm|rmsnorm}.pt
# and best:
#   gqa_h{n_head}_kv{n_kv_head}_{rope|seq}_{layernorm|rmsnorm}_best.pt
#
# Backward compatibility:
# - If an older checkpoint exists using ckpt_name-based naming, we will resume from it
#   (but we will SAVE going forward using the new canonical name).
#
# Usage:
#   python -m bench.train config/train_shakespeare_char.py --out_dir=out-attn --n_head=12 --n_kv_head=12 --norm_type=rmsnorm --use_rope=False

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from nanogpt.model import GPT, GPTConfig

print("RUNNING:", os.path.abspath(__file__))

# -----------------------------
# Default config (overridden by config file + CLI)
# -----------------------------

seed = 1337

# I/O
out_dir = "out"
ckpt_name = "ckpt"  # legacy logical run name (used only for backward-compat resume)
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True

# data
dataset = "shakespeare_char"
batch_size = 12
block_size = 256

# model
n_layer = 12
n_head = 12
n_embd = 768
n_kv_head = 0  # 0 => default to n_head (MHA)
dropout = 0.0
bias = True
use_rope = False

# norm
norm_type = "layernorm"  # "layernorm" | "rmsnorm"
norm_eps = 1e-5

# optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr decay
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# system
device = "cuda"
dtype = "bfloat16"  # 'float32', 'float16', 'bfloat16'
compile = False

# -----------------------------
# config file + CLI overrides (robust)
# - always load config file if provided
# - then apply --key=value overrides
# -----------------------------
if len(sys.argv) > 1 and sys.argv[1].endswith(".py"):
    cfg_path = sys.argv[1]
    exec(open(cfg_path, "r").read(), globals())

# apply CLI overrides like --use_rope=True --norm_type=rmsnorm
for arg in sys.argv[2:]:
    if not arg.startswith("--"):
        continue
    key, eq, val = arg[2:].partition("=")
    if eq != "=":
        continue
    if key not in globals():
        raise ValueError(f"Unknown config key: {key}")

    cur = globals()[key]
    if isinstance(cur, bool):
        globals()[key] = val.lower() in ("1", "true", "yes", "y", "t")
    elif isinstance(cur, int):
        globals()[key] = int(val)
    elif isinstance(cur, float):
        globals()[key] = float(val)
    else:
        globals()[key] = val

# normalize norm_type early
norm_type = str(norm_type).strip().lower()
if norm_type not in ("layernorm", "rmsnorm"):
    raise ValueError("norm_type must be 'layernorm' or 'rmsnorm'")

norm_eps = float(norm_eps)

# -----------------------------
# DDP setup (optional; works fine single-GPU)
# -----------------------------
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl")
    master_process = rank == 0
else:
    master_process = True
    world_size = 1
    local_rank = 0

# -----------------------------
# Seeding (DONE ONCE, AFTER DDP SETUP)
# - In DDP, offset by local_rank to avoid identical sampling across ranks
# -----------------------------
seed = int(seed) + int(local_rank)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

# Optional determinism knobs (can reduce performance; helpful for debugging)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]

# -----------------------------
# normalize + validate KV head grouping after config/CLI has been applied
# -----------------------------
if n_kv_head == 0:
    n_kv_head = n_head
assert isinstance(n_kv_head, int) and n_kv_head > 0
assert n_head % n_kv_head == 0, f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head})"

# -----------------------------
# NEW canonical checkpoint naming
# -----------------------------
rope_tag = "rope" if bool(use_rope) else "seq"
norm_tag = "rmsnorm" if norm_type == "rmsnorm" else "layernorm"
ckpt_tag = f"gqa_h{n_head}_kv{n_kv_head}_{rope_tag}_{norm_tag}"

os.makedirs(out_dir, exist_ok=True)

ckpt_path = os.path.join(out_dir, f"{ckpt_tag}.pt")
best_path = os.path.join(out_dir, f"{ckpt_tag}_best.pt")

# -----------------------------
# Legacy naming (backward compatible resume)
# - old policy: ckpt_name (+ optional _rmsnorm) WITHOUT rope in name
# -----------------------------
legacy_norm_suffix = "_rmsnorm" if norm_type == "rmsnorm" else ""
legacy_tag = f"{ckpt_name}{legacy_norm_suffix}"
legacy_ckpt_path = os.path.join(out_dir, f"{legacy_tag}.pt")

print("DEBUG out_dir          =", out_dir)
print("DEBUG n_head/n_kv_head =", n_head, n_kv_head)
print("DEBUG use_rope         =", use_rope, "->", rope_tag)
print("DEBUG norm_type        =", norm_type, "->", norm_tag)
print("DEBUG ckpt_tag         =", ckpt_tag)
print("DEBUG ckpt_path        =", ckpt_path)
print("DEBUG legacy_tag       =", legacy_tag)
print("DEBUG legacy_ckpt_path =", legacy_ckpt_path)

# -----------------------------
# helpers
# -----------------------------
def get_lr(it: int) -> float:
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model, get_batch_fn, device_type: str):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_fn(split)
            with torch.autocast(
                device_type=device_type,
                dtype=ptdtype,
                enabled=(device_type == "cuda" and dtype != "float32"),
            ):
                logits, loss = model(X, Y)  # <-- FIX: don't assign to "_"
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# -----------------------------
# data loader
# -----------------------------
data_dir = os.path.join("data", dataset)
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")


def get_batch(split: str):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y


# -----------------------------
# init model
# -----------------------------
meta_path = os.path.join(data_dir, "meta.pkl")
vocab_size = None
if os.path.exists(meta_path):
    import pickle

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta.get("vocab_size", None)

if "vocab_size" not in globals() and vocab_size is None:
    raise ValueError(
        "vocab_size not found. Set vocab_size in your config file or provide data/{dataset}/meta.pkl with vocab_size."
    )

if "vocab_size" in globals():
    vocab_size = globals()["vocab_size"]

model_args = dict(
    block_size=block_size,
    vocab_size=vocab_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    n_kv_head=n_kv_head,
    dropout=dropout,
    bias=bias,
    use_rope=use_rope,
    norm_type=norm_type,
    norm_eps=float(norm_eps),
)

gptconf = GPTConfig(**model_args)
model = GPT(gptconf).to(device)

if compile:
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[local_rank])

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    weight_decay=weight_decay,
)

scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda" and dtype == "float16"))

# -----------------------------
# resume logic:
# Prefer NEW canonical checkpoint; else try legacy checkpoint.
# NOTE: regardless of resume source, we SAVE to canonical ckpt_path going forward.
# -----------------------------
iter_num = 0
best_val_loss = 1e9

resume_path = None
if os.path.exists(ckpt_path):
    resume_path = ckpt_path
elif os.path.exists(legacy_ckpt_path):
    resume_path = legacy_ckpt_path

if resume_path is not None:
    if master_process:
        print(f"Resuming from checkpoint: {resume_path}")

    checkpoint = torch.load(resume_path, map_location=device)
    state_dict = checkpoint["model"]
    if ddp:
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    optimizer.load_state_dict(checkpoint["optimizer"])
    iter_num = int(checkpoint.get("iter_num", 0))
    best_val_loss = float(checkpoint.get("best_val_loss", 1e9))

    ckpt_args = checkpoint.get("model_args", {})
    for k in [
        "use_rope",
        "n_kv_head",
        "block_size",
        "n_layer",
        "n_head",
        "n_embd",
        "vocab_size",
        "norm_type",
        "norm_eps",
    ]:
        if k in ckpt_args:
            assert ckpt_args[k] == model_args[k], (
                f"Checkpoint {k}={ckpt_args[k]} but current run {k}={model_args[k]}. "
                "Train a separate variant (rope/seq + norm) or point to the right ckpt."
            )

# -----------------------------
# training loop
# -----------------------------
t0 = time.time()
while True:
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss(model.module if ddp else model, get_batch, device_type=device_type)
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        improved = losses["val"] < best_val_loss
        if improved:
            best_val_loss = losses["val"]

        if always_save_checkpoint or improved:
            raw_model = model.module if ddp else model
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
            }

            # ALWAYS save canonical filenames
            torch.save(checkpoint, ckpt_path)
            if improved:
                torch.save(checkpoint, best_path)

            print(f"saved: {ckpt_path} (best={best_val_loss:.4f})")

        if eval_only:
            break

    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    X, Y = get_batch("train")

    with torch.autocast(
        device_type=device_type,
        dtype=ptdtype,
        enabled=(device_type == "cuda" and dtype != "float32"),
    ):
        logits, loss = model(X, Y) if not ddp else model(X, Y)  # <-- FIX: don't use "_"

    optimizer.zero_grad(set_to_none=True)

    if scaler.is_enabled():
        scaler.scale(loss).backward()
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    if master_process and iter_num % log_interval == 0:
        dt = time.time() - t0
        t0 = time.time()
        print(f"iter {iter_num}: loss {loss.item():.4f}, lr {lr:.3e}, dt {dt*1000:.2f}ms")

    iter_num += 1
    if iter_num > max_iters:
        break

if ddp:
    torch.distributed.destroy_process_group()
