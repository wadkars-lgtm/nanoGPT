"""
Sample from a trained model (supports MHA / GQA / MQA checkpoints)

Key point:
- If you trained a char-level dataset (e.g. shakespeare_char), you MUST use its meta.pkl
  for encoding/decoding. Falling back to GPT-2 tokenizer will produce token IDs > vocab_size
  and crash with CUDA indexing asserts.

PowerShell examples at bottom.
"""

import os
import pickle
from contextlib import nullcontext

import torch
import tiktoken

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Defaults (overridable via configurator.py / CLI)
# -----------------------------------------------------------------------------
init_from = "resume"            # 'resume' (from checkpoint) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = "out"                 # directory containing checkpoints
ckpt_name = "ckpt"              # which checkpoint to load: {out_dir}/{ckpt_name}.pt

dataset = "shakespeare_char"    # used to locate data/{dataset}/meta.pkl for char-level models

start = "\n"                    # prompt string, or FILE:prompt.txt
num_samples = 3
max_new_tokens = 300
temperature = 0.8
top_k = 200
seed = 1337

device = "cuda"                 # 'cpu', 'cuda', 'cuda:0', etc.
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = False                 # torch.compile

# NEW: what to do if prompt contains characters not in stoi (char-level datasets)
# options: "error" | "drop" | "replace"
prompt_unknown_policy = "error"
prompt_unknown_replacement = " "   # used only if policy == "replace"

# -----------------------------------------------------------------------------
# CLI/config overrides (nanoGPT-style)
# -----------------------------------------------------------------------------
exec(open("configurator.py").read())
# -----------------------------------------------------------------------------

def normalize_prompt_text(s: str) -> str:
    """
    Normalize common Unicode punctuation to ASCII so char-level vocab doesn't KeyError.
    """
    replacements = {
        "\u2018": "'",  # ‘
        "\u2019": "'",  # ’
        "\u201c": '"',  # “
        "\u201d": '"',  # ”
        "\u2013": "-",  # –
        "\u2014": "-",  # —
        "\u2212": "-",  # −
        "\xa0": " ",    # non-breaking space
        "\t": " ",      # tabs -> spaces
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s

def summarize_unknown_chars(s: str, stoi: dict) -> list[str]:
    """
    Returns sorted list of distinct chars in s that are not present in stoi.
    """
    unknown = sorted({c for c in s if c not in stoi})
    return unknown

def apply_unknown_policy(s: str, stoi: dict, policy: str, replacement: str) -> str:
    """
    Enforce policy for characters not in stoi.
    """
    unknown = summarize_unknown_chars(s, stoi)
    if not unknown:
        return s

    if policy == "error":
        # show readable repr so you can see whitespace/newlines too
        preview = ", ".join([repr(c) for c in unknown[:50]])
        more = "" if len(unknown) <= 50 else f" ... (+{len(unknown)-50} more)"
        raise KeyError(
            f"Prompt contains {len(unknown)} character(s) not in dataset vocab (meta.pkl stoi). "
            f"Unknown chars: {preview}{more}\n"
            f"Fix: replace smart quotes, remove unicode, or set --prompt_unknown_policy=drop|replace."
        )

    if policy == "drop":
        return "".join([c for c in s if c in stoi])

    if policy == "replace":
        return "".join([c if c in stoi else replacement for c in s])

    raise ValueError(f"Invalid prompt_unknown_policy={policy}. Use error|drop|replace.")

# reproducibility / perf knobs
torch.manual_seed(seed)
if "cuda" in device and torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device_type = "cuda" if ("cuda" in device and torch.cuda.is_available()) else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------------
checkpoint = None
if init_from == "resume":
    ckpt_path = os.path.join(out_dir, f"{ckpt_name}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)

    state_dict = checkpoint["model"]
    # strip torch.compile prefix if present
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)

elif init_from.startswith("gpt2"):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
    gptconf = model.config

else:
    raise ValueError(f"Unknown init_from={init_from}")

model.eval()
model.to(device)

if compile:
    model = torch.compile(model)

print(f"Loaded checkpoint: init_from={init_from}, out_dir={out_dir}, ckpt_name={ckpt_name}")
print(
    f"Model vocab_size={gptconf.vocab_size}, block_size={gptconf.block_size}, "
    f"n_head={gptconf.n_head}, n_kv_head={getattr(gptconf,'n_kv_head',None)}"
)

# -----------------------------------------------------------------------------
# Tokenizer / encoder-decoder
# -----------------------------------------------------------------------------
meta_path = os.path.join("data", dataset, "meta.pkl")
using_char_vocab = False

if init_from == "resume" and os.path.exists(meta_path):
    using_char_vocab = True
    print(f"Loading meta from {meta_path}")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    stoi, itos = meta["stoi"], meta["itos"]

    def encode(s: str):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join([itos[i] for i in ids])

else:
    print("No meta.pkl found, using GPT-2 tokenizer")
    enc = tiktoken.get_encoding("gpt2")

    def encode(s: str):
        return enc.encode(s, allowed_special={"<|endoftext|>"})

    def decode(ids):
        return enc.decode(ids)

# -----------------------------------------------------------------------------
# Prompt
# -----------------------------------------------------------------------------
if isinstance(start, str) and start.startswith("FILE:"):
    with open(start[5:], "r", encoding="utf-8") as f:
        start = f.read()

# NEW: normalize smart quotes/etc so char-level vocabs don't blow up
start = normalize_prompt_text(start)

# NEW: handle unknown chars for char-level vocab
if using_char_vocab:
    start = apply_unknown_policy(
        start,
        stoi=stoi,
        policy=prompt_unknown_policy,
        replacement=prompt_unknown_replacement,
    )

start_ids = encode(start)
if len(start_ids) == 0:
    raise ValueError("Prompt encoded to empty token list.")

# Runtime sanity check to prevent CUDA assert (relevant when using GPT-2 tokenizer with small vocab)
max_id = max(start_ids)
if max_id >= gptconf.vocab_size:
    raise ValueError(
        f"Prompt contains token id {max_id} but model vocab_size is {gptconf.vocab_size}. "
        f"You're almost certainly using the wrong tokenizer. "
        f"If this is shakespeare_char, ensure data/{dataset}/meta.pkl exists and dataset is set correctly."
    )

x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# -----------------------------------------------------------------------------
# Generate
# -----------------------------------------------------------------------------
with torch.no_grad():
    with ctx:
        for i in range(num_samples):
            y = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            print(decode(y[0].tolist()))
            print("-" * 80)

"""
PowerShell examples:

# (1) Sample from a trained run (char-level shakespeare)
python .\sample.py `
  --out_dir=out-attn `
  --ckpt_name=gqa_h12_kv3 `
  --dataset=shakespeare_char `
  --start="FILE:.\prompt.txt" `
  --num_samples=2 `
  --max_new_tokens=200

# (2) If your prompt has weird unicode and you want it to auto-handle:
# Drop unknown chars:
python .\sample.py `
  --out_dir=out-attn `
  --ckpt_name=gqa_h12_kv3 `
  --dataset=shakespeare_char `
  --start="FILE:.\prompt.txt" `
  --prompt_unknown_policy="drop" `
  --num_samples=2 `
  --max_new_tokens=200

# Replace unknown chars with space:
python .\sample.py `
  --out_dir=out-attn `
  --ckpt_name=gqa_h12_kv3 `
  --dataset=shakespeare_char `
  --start="FILE:.\prompt.txt" `
  --prompt_unknown_policy="replace" `
  --prompt_unknown_replacement=" " `
  --num_samples=2 `
  --max_new_tokens=200
"""
