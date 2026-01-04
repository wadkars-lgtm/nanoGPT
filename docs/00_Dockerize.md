# Dockerize


## Create a full dependency snapshot
```powershell
pip install pip-tools
pip-compile pyproject.toml `
  --output-file=requirements.lock.txt `
  --resolver=backtracking
```

## Create docker image

```powershell
docker build -t nanogpt:cu128 -f docker/Dockerfile .
```

```powershell
docker run --rm -it --gpus all `
  -v ${PWD}:/workspace `
  nanogpt:cu128
```

```shell
python -m bench.sample \
  --out_dir=out-attn \
  --ckpt_name=mha_h12_kv12 \
  --dataset=shakespeare_char \
  --start=$'\n' \
  --num_samples=3 \
  --max_new_tokens=300 \
  --temperature=0.8 \
  --top_k=200
  ```
