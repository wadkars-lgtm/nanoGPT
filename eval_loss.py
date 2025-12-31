import argparse
import math
import os
import pickle
import numpy as np
import torch

from model import GPT, GPTConfig


def load_memmap(data_dir: str):
    train_bin = os.path.join(data_dir, "train.bin")
    val_bin = os.path.join(data_dir, "val.bin")
    if not os.path.exists(val_bin):
        raise FileNotFoundError(val_bin)
    train = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val = np.memmap(val_bin, dtype=np.uint16, mode="r")
    return train, val


def get_batch(data: np.memmap, block_size: int, batch_size: int, device: str):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_val_loss(model: GPT, val_data: np.memmap, block_size: int, batch_size: int, iters: int, device: str, autocast_dtype):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(val_data, block_size, batch_size, device)
        with torch.autocast(
            device_type="cuda" if "cuda" in device else "cpu",
            dtype=autocast_dtype,
            enabled=("cuda" in device and autocast_dtype is not None),
        ):
            logits, loss = model(x, y, use_cache=False, past_kv=None)
        losses.append(float(loss.item()))
    mean_loss = float(np.mean(losses))
    ppl = math.exp(mean_loss) if mean_loss < 50 else float("inf")
    return mean_loss, ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="shakespeare_char")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_iters", type=int, default=200)
    args = ap.parse_args()

    device = args.device
    data_dir = args.data_dir or os.path.join("data", args.dataset)
    _, val = load_memmap(data_dir)

    ckpt = torch.load(args.ckpt, map_location=device)
    if "model_args" not in ckpt or "model" not in ckpt:
        raise ValueError("Checkpoint missing model_args or model keys; update eval script to match your ckpt format.")

    cfg = GPTConfig(**ckpt["model_args"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    autocast_dtype = None
    if "cuda" in device:
        autocast_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": None,
        }[args.dtype]

    loss, ppl = eval_val_loss(model, val, cfg.block_size, args.batch_size, args.eval_iters, device, autocast_dtype)
    print(f"ckpt={args.ckpt}")
    print(f"val_loss={loss:.6f}")
    print(f"val_ppl={ppl:.4f}")


if __name__ == "__main__":
    main()
