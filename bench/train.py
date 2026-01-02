# train.py
# nanoGPT-style training loop with:
# - MHA/GQA/MQA support via n_kv_head passed into GPTConfig
# - configurable checkpoint base name via ckpt_name
#
# Usage:
#   python train.py config/train_shakespeare_char.py --n_kv_head=1 --ckpt_name=mqa

from __future__ import annotations

import math
import os
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig

# -----------------------------
# Default config (overridden by config file + CLI)
# -----------------------------

# I/O
out_dir = "out"
ckpt_name = "ckpt"            # base name for checkpoint files
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
n_kv_head = 0                 # 0 => default to n_head (MHA)
dropout = 0.0
bias = True
# use_sdpa = True             # REMOVED (train.py no longer passes this into GPTConfig)

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
dtype = "bfloat16"            # 'float32', 'float16', 'bfloat16'
compile = False

# -----------------------------
# config file + CLI overrides
# -----------------------------
import sys

if os.path.exists("configurator.py"):
    exec(open("configurator.py", "r").read(), globals())
else:
    if len(sys.argv) > 1 and sys.argv[1].endswith(".py"):
        exec(open(sys.argv[1], "r").read(), globals())

# normalize + validate KV head grouping after config/CLI has been applied
if n_kv_head == 0:
    n_kv_head = n_head
assert isinstance(n_kv_head, int) and n_kv_head > 0
assert n_head % n_kv_head == 0, f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head})"

# -----------------------------
# helpers
# -----------------------------

def get_lr(it: int) -> float:
    # linear warmup + cosine decay (nanoGPT)
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

@torch.no_grad()
def estimate_loss(model: torch.nn.Module, get_batch_fn, device_type: str):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_fn(split)
            with torch.autocast(device_type=device_type, dtype=ptdtype, enabled=(device_type == "cuda" and dtype != "float32")):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

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

device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]

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
    raise ValueError("vocab_size not found. Set vocab_size in your config file or provide data/{dataset}/meta.pkl with vocab_size.")

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
)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

if compile:
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[local_rank])

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda" and dtype == "float16"))

os.makedirs(out_dir, exist_ok=True)
ckpt_path = os.path.join(out_dir, f"{ckpt_name}.pt")
best_path = os.path.join(out_dir, f"{ckpt_name}_best.pt")

iter_num = 0
best_val_loss = 1e9
if os.path.exists(ckpt_path):
    if master_process:
        print(f"Resuming from checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model"]
    if ddp:
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint["optimizer"])
    iter_num = checkpoint.get("iter_num", 0)
    best_val_loss = checkpoint.get("best_val_loss", 1e9)

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

    with torch.autocast(device_type=device_type, dtype=ptdtype, enabled=(device_type == "cuda" and dtype != "float32")):
        _, loss = (model(X, Y) if not ddp else model(X, Y))

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
