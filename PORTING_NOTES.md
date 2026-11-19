# OLMo 2 support for sgl-jax: porting notes

Branch `add-olmo2-support`. It adds `Olmo2ForCausalLM`, which covers `allenai/OLMo-2-0425-1B`, `OLMo-2-1124-7B`, `-13B`, `-0325-32B` and their Instruct/SFT/DPO variants.

## Why OLMo 2
- It is fully open: Apache-2.0, ungated, with open data and code. It has a 1B variant that is easy to test.
- No sgl-jax issue or PR mentions OLMo (checked 2026-11-18). Gemma 3 already has an open PR (#1100). GPT-OSS has an open PR (#630). Mistral is already routed through the Llama code path. Phi-3 is served by `llama.py`.
- It is a dense model and close enough to Qwen3 to reuse every sgl-jax layer, with no new kernels.

## What changed from the template (`qwen3.py`)
| Area | Qwen3 | OLMo 2 |
|---|---|---|
| Block norm | pre-norm (`input_layernorm`, `post_attention_layernorm` on the input) | **post-norm**: `h = x + post_attention_layernorm(attn(x))`, `y = h + post_feedforward_layernorm(mlp(h))`. There is no `input_layernorm`. |
| Residual plumbing | fused residual carried between layers | plain residual inside each layer. `Olmo2Model` has no trailing `+ residual` before the final `norm`. |
| QK-norm | RMSNorm per head (`head_dim`) after the reshape | RMSNorm over the **whole projected** q (`num_heads*head_dim`) and k (`num_kv_heads*head_dim`), applied **before** the reshape |
| RoPE | NeoX style, `rope_parameters` | the same, with theta 500k. A helper also accepts older `rope_theta`/`rope_scaling` configs. |
| Weights | `input_layernorm` | `post_feedforward_layernorm`. The q_norm/k_norm shapes are full-width. |

## TPU and sharding choices
- q/k/v/gate/up are column-parallel on `"tensor"` and o/down are row-parallel, the same as Qwen3.
- The full-width QK-norm reduces over an axis that is sharded on `"tensor"`. Under the Explicit mesh, XLA inserts a small all-reduce for the variance: one `[tokens]` scalar per q and per k, per layer. The norm scale is replicated (`(None,)`). The TP=2 CPU run checks this path.
- K/V use the standard `kv_head_padding=True` loader path, so a TP degree larger than `num_kv_heads` still works the usual way. OLMo-2 1B/7B use MHA (kv_heads == heads).
- The FlashAttention/RPA kernel needs no changes because head_dim is 128.

## Gotchas found
1. **A batch of several requests is wrong on the CPU `native` backend, and this is not specific to OLMo.** When more than one request is prefilled together, every request after the first diverges from HF. The prompt logprob error is about 0.5 to 1.0 and greedy tokens differ. The upstream `LlamaForCausalLM` shows the same failure with a tiny random config. The same tests pass at batch size 1, so the parity test sends one request at a time. Worth an upstream issue; the likely place is `_apply_extend_mask` / page-aligned `cache_loc` on CPU. It is not expected on TPU, where `fa` is used, but the TPU run below checks for it.
2. A tiny config with head_dim 32 failed with a `dot_general` shape error in `o_proj`, because the attention output is padded to 128 lanes. The tests use head_dim 128, which matches the real models.
3. Engine tests need an `if __name__ == "__main__"` guard. The engine uses spawn multiprocessing, and without the guard the child re-runs the script and hangs.
4. A tiny random-init **Qwen3** config also failed parity, even for the first request (max |dlogprob| about 0.52), using the same harness. This was not investigated because it is outside this change. It is noted here in case it is a harness or transformers-5 config interaction.

## Test results (CPU, JAX 0.11.1, torch 2.15, transformers 5.12.1, fp32)
`test/srt/models/test_olmo2_hf_parity.py`: a random-init Olmo2 model (3 layers, hidden 512, 4 heads, 2 KV heads for GQA, norm weights randomized) served through the full sgl-jax `Engine` (`device=cpu`, `attention_backend=native`).

| Run | worst \|prompt logprob − HF\| | greedy 8 tokens |
|---|---|---|
| TP=1 | 2.86e-6 | identical on all 3 prompts |
| TP=2 (2 simulated CPU devices) | 2.86e-6 | identical on all 3 prompts |

The real-checkpoint test (`OLMO2_REAL_MODEL=allenai/OLMo-2-0425-1B`) was **not run**. The dev machine had less than 1 GB of free disk space and the checkpoint is about 6 GB. The test is written and skipped by default.

```
JAX_PLATFORMS=cpu python -m pytest -s test/srt/models/test_olmo2_hf_parity.py            # tiny, ~40 s
OLMO2_TP=2 JAX_PLATFORMS=cpu python -m pytest -s test/srt/models/test_olmo2_hf_parity.py
OLMO2_REAL_MODEL=allenai/OLMo-2-0425-1B JAX_PLATFORMS=cpu python -m pytest -s test/srt/models/test_olmo2_hf_parity.py -k real
```

## TPU validation (not yet run)
Use one v6e-1 (1 chip, 32 GB HBM). The 1B model in bf16 needs about 3 GB.
```bash
# on the TPU VM
git clone -b add-olmo2-support <this repo> sglang-jax && cd sglang-jax/python
pip install -e ".[tpu]" && cd ..
# 1) correctness vs HF (CPU torch reference on the VM host), bf16 tolerance
OLMO2_DEVICE=tpu OLMO2_ATTN=fa OLMO2_REAL_MODEL=allenai/OLMo-2-0425-1B OLMO2_REAL_ATOL=0.1 \
  python -m pytest -s test/srt/models/test_olmo2_hf_parity.py -k real
# 2) serve + smoke + throughput
python -m sgl_jax.launch_server --model-path allenai/OLMo-2-0425-1B --tp-size 1 --port 30000 &
curl -s localhost:30000/generate -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":16,"temperature":0}}'
python -m sgl_jax.bench_serving --backend sgl-jax --num-prompts 200 --random-input 512 --random-output 128 --dataset-name random
```
On TPU, add `OLMO2_DEVICE=tpu OLMO2_ATTN=fa` to the first command so it runs through the real TPU kernels.
Cost estimate: v6e-1 is about $2.70/hr on demand or about $1.0–1.3/hr spot (us-east5 / europe-west4). Setup, the test and the benchmark take about 45 min, so **about $2 on demand or under $1 spot.** Delete the VM afterwards.

## Not done
- Real-checkpoint parity (blocked by disk space), TPU correctness and performance numbers.
- OLMo 3 (`Olmo3ForCausalLM`) adds sliding window on 3 of every 4 layers. It could be added on top of this model and `RadixAttention`'s sliding window.
- Adding the model to a docs "supported models" list.


## Olmo 3 / 3.1 (`Olmo3ForCausalLM`, branch `add-olmo3-support`)
Olmo 3 is the OLMo 2 block plus two config-driven changes, so it reuses `olmo2.py`:
- Interleaved sliding-window attention: `config.layer_types` (3 `sliding_attention` : 1 `full_attention`), window `config.sliding_window` (4096 for 7B). Passed per layer to `RadixAttention(sliding_window_size=...)`; the model runner then sets up the hybrid SWA KV pool.
- YaRN RoPE (`factor` 8, `original_max_position_embeddings` 8192). The existing `YarnRotaryEmbedding` reproduces HF `attention_factor = 0.1*ln(factor)+1` through its mscale.

CPU parity (`test/srt/models/test_olmo3_hf_parity.py`, tiny random fp32, window 3 so it really masks, YaRN on): worst |prompt logprob - HF| 1.9e-6, identical greedy tokens. The CPU `native` backend has no SWA KV pool, so that test runs with `OLMO_DISABLE_HYBRID_SWA=1` (window still applied in the mask). OLMo 2 test unchanged: 2.9e-6.

    OLMO_DISABLE_HYBRID_SWA=1 JAX_PLATFORMS=cpu python -m pytest -s test/srt/models/test_olmo3_hf_parity.py
