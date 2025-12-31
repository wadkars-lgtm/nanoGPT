# GQA vs MQA vs MHA: Decode Performance, Memory, and Accuracy Tradeoffs

This document summarizes a controlled experimental study comparing **Multi-Head Attention (MHA)**,
**Grouped-Query Attention (GQA)**, and **Multi-Query Attention (MQA)** in nanoGPT.

All experiments were run with identical model size, training setup, dataset, and evaluation protocol.
Only the number of KV heads (`n_kv_head`) was varied.

The goal is to understand the **full systems tradeoff**:
- Decode latency
- GPU memory consumption (KV cache)
- Model quality (validation loss and perplexity)

---

## 1. Decode Latency vs Prompt Length

These plots show **decode latency per generated token** as a function of prompt length, for different batch sizes.

### Batch = 1
![](results/gqa/plots/decode_ms_per_tok_vs_promptlen_B1.png)

### Batch = 4
![](results/gqa/plots/decode_ms_per_tok_vs_promptlen_B4.png)

### Batch = 8
![](results/gqa/plots/decode_ms_per_tok_vs_promptlen_B8.png)

### Batch = 16
![](results/gqa/plots/decode_ms_per_tok_vs_promptlen_B16.png)

### Batch = 32
![](results/gqa/plots/decode_ms_per_tok_vs_promptlen_B32.png)

---

## 2. Decode Latency vs Batch Size

### Prompt Length = 128
![](results/gqa/plots/decode_ms_per_tok_vs_batchsize_T128.png)

### Prompt Length = 512
![](results/gqa/plots/decode_ms_per_tok_vs_batchsize_T512.png)

### Prompt Length = 1024
![](results/gqa/plots/decode_ms_per_tok_vs_batchsize_T1024.png)

### Prompt Length = 2048
![](results/gqa/plots/decode_ms_per_tok_vs_batchsize_T2048.png)

---

## 3. Decode Peak Memory vs Prompt Length

### Batch = 1
![](results/gqa/plots/decode_peakmem_vs_promptlen_B1.png)

### Batch = 4
![](results/gqa/plots/decode_peakmem_vs_promptlen_B4.png)

### Batch = 8
![](results/gqa/plots/decode_peakmem_vs_promptlen_B8.png)

### Batch = 16
![](results/gqa/plots/decode_peakmem_vs_promptlen_B16.png)

### Batch = 32
![](results/gqa/plots/decode_peakmem_vs_promptlen_B32.png)

---

## 4. Validation Loss vs KV Heads

![](results/gqa/plots/val_loss_vs_n_kv_head.png)

---

## 5. Validation Perplexity vs KV Heads

![](results/gqa/plots/val_ppl_vs_n_kv_head.png)

---

## 6. Conclusion

Grouped-Query Attention (GQA) sits on the Pareto frontier for inference:
- Near-MHA quality
- Much lower KV-cache memory
- Better decode throughput

This explains why modern inference engines default to GQA over classical MHA.
