# GPU and memory on the ILPS cluster

Where each piece runs, what it costs, and why the current settings are what they
are. Every number below was measured on this cluster with `sacct` or read out of
a job log — none are estimates. Reference run: **342904** (most recent completed
training; cn119, 4x L40, 25.3 h, 150 steps).

## Nodes in the `gpu` partition

| node | GPUs | host RAM | fits training *with* checkpoints (400G)? |
|---|---|---|---|
| **cn119** | 8x nvidia_l40 | **756 GiB** | yes, one — plus a 100G retriever beside it |
| **cn120** | 8x nvidia_rtx_a6000 | 504 GiB | yes, one |
| cn117 | 8x nvidia_rtx_a6000 | 312 GiB | **no** — 320,000 MB < 400G; `sbatch` rejects it |
| cn118 | 8x nvidia_rtx_a6000 | 250 GiB | **no** — use for the retriever |
| cn116 | 4x nvidia_rtx_a6000 | 251 GiB | **no** — retriever, or training with `save_freq: 0` |
| cn101–115 | Titan X / Xp / V, P40 | 250 GiB | no — GPUs far too small for a 3B policy |

A6000 and L40 both expose ~47.4 GiB usable VRAM.

**Always name the GPU type** (`--gres=gpu:nvidia_l40:4`). A bare `--gres=gpu:4`
can land you on a Titan X.

## Retrieval — bounded by host RAM, barely touches the GPU

Measured **61.4 GiB** RSS, and `AveRSS == MaxRSS`: the index is read in once and
stays resident for the life of the job.

`faiss.read_index` pulls the 60.1 GiB `e5_Flat.index` (from
`/fnwi_fs/ivi/irlab/projects/chatr1/collection/`) into host RAM, then
`--faiss_gpu` shards it across the allocated GPUs as fp16, about 15 GiB per GPU
on two. The wiki-18 corpus is memory-mapped Arrow and costs almost nothing.

- `--mem=100G` is correct and is what `run/retrieval.job` uses. Don't go below ~80G.
- 2 GPUs is plenty. Put it on **cn118 or cn116** and leave the big-RAM nodes free.
- Startup is 15–30 min before the port opens, so **submit it first**.
- It is only queried during the `gen` phase (~24% duty cycle), so run **one**
  retriever and point several trainers at it rather than one each.

## Training — GPU

Per device, from `log_gpu_memory_usage` in the 342904 log:

```
before vllm rollout :  2.9 GB allocated            (FSDP-sharded 3B policy)
after  vllm rollout : 23.1 GB allocated / 28.9 GB reserved
```

28.9 GB is exactly `gpu_memory_utilization: 0.6` x 48 GB, and vLLM holds it for
the whole step. **That leaves ~19 GB** for the log-prob and actor-update passes,
which is the entire reason for two settings in `subsearch_grpo.yaml`:

- **`log_prob_max_token_len_per_gpu: 8192`** — Qwen2.5-3B has a 151,936 vocab and
  `entropy_from_logits` materialises `softmax(logits)` and `pd*logits` alongside
  `logits`, so peak ≈ 3 x tokens x vocab x 2 B. 8192 tokens → ~7 GB.
  Job 339222 died here: 32 sequences x ~2500 tokens = **22.75 GiB for one logits
  tensor**, ~68 GiB with the entropy terms, against 47.4 GiB of card.
- **`ppo_micro_batch_size: 16`** — divided by world size, so 4 sequences per GPU.

Both are divided by the number of GPUs in `fsdp_workers.py`, so keep them
multiples of the world size or integer division silently truncates.

## Training — host RAM, driven by checkpointing not batch size

`_save_checkpoint` gathers a full fp32 state dict on rank 0 while the
CPU-offloaded params, grads and Adam moments are still resident. That peak scales
with **model size, not batch size**:

| job | batch | checkpoints | node | MaxRSS |
|---|---|---|---|---|
| 341915 | 16 | **none** | cn116 A6000 | **184.2 GiB** |
| 342904 | 128 | 6 saves | cn119 L40 | **248.6 GiB** |
| 339350 | 512 | 8 saves | cn120 A6000 | **252.8 GiB** |

A 4x larger batch (128 → 512) adds ~4 GiB. Turning saving on adds ~65 GiB.

| configuration | request |
|---|---|
| `save_freq > 0` | **`--mem=400G`** |
| `save_freq: 0` | `--mem=250G` (184 GiB measured; 200G leaves only 16 GiB spare) |

Two separate killers enforce this:

- **SLURM cgroup.** Job 339460 was killed by the kernel at 215 GiB against a 200G
  request — `slurmstepd: Detected 1 oom_kill event`.
- **Ray's memory monitor**, which triggers on **node-wide** usage at 95%, not on
  your cgroup. Job 339329 died this way on cn117. Two large jobs on a small node
  can therefore kill each other. cn119's 756 GiB makes it the safe place to put a
  trainer and a retriever together (400G + 100G); it cannot take two 400G trainers
  (773,668 MB < 819,200 MB). Raise with `RAY_memory_usage_threshold=0.97` if
  needed, but prefer more `--mem`.

## Running it

```bash
sbatch run/retrieval.job     # 2 GPUs, 100G, ~30 min index load — submit first
sbatch run/train.job         # 4 GPUs, 400G, polls for the retriever, then trains
```

`run/env.sh` holds every path (data, caches, checkpoints, retriever endpoint) —
edit there, not in the job files.

## Which knob for which failure

| symptom | cause | fix |
|---|---|---|
| `torch.OutOfMemoryError` in `compute_log_prob` / `update_policy` | GPU: logits tensor too large | lower `log_prob_max_token_len_per_gpu`, `ppo_micro_batch_size` |
| SLURM state `OUT_OF_MEMORY`, `oom_kill event` in the log | host RAM: cgroup exceeded, usually at a checkpoint save | raise `--mem` to 400G |
| `ray.exceptions.OutOfMemoryError`, "node running low on memory" | host RAM: node-wide 95% | move to a bigger node, or stop sharing it |
| `allocation failure: Requested node configuration is not available` | `--mem` exceeds every node's RAM | lower `--mem`, or target cn119/cn120 |
| job pending on `(Resources)` for days | `--mem` too close to node total | drop `--mem`, or pick a node with headroom |
