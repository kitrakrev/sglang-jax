"""OLMo 2 numerical parity vs HF transformers, runnable on CPU.

Builds a tiny random-init Olmo2ForCausalLM in PyTorch (fp32), saves it as
safetensors, serves it with the sgl-jax Engine (JAX CPU, native attention,
float32) and compares per-token prompt logprobs and greedy continuations.

    JAX_PLATFORMS=cpu python -m pytest -s test/srt/models/test_olmo2_hf_parity.py

Set OLMO2_REAL_MODEL=allenai/OLMo-2-0425-1B to also check a real checkpoint.
"""

import os
import tempfile
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
TP_SIZE = int(os.environ.get("OLMO2_TP", "1"))
if TP_SIZE > 1:  # simulate a TP mesh with fake CPU devices
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={TP_SIZE}"

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Olmo2Config,
    Olmo2ForCausalLM,
)

TOKENIZER = os.environ.get("OLMO2_TOKENIZER", "allenai/OLMo-2-0425-1B")
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Tensor processing units are",
]


def _hf_logprobs(model, ids):
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0].float()
    lp = torch.log_softmax(logits, -1)
    # logprob of token t given tokens < t, for t = 1..n-1
    return np.array([lp[i - 1, ids[i]].item() for i in range(1, len(ids))]), logits


def _hf_greedy(model, ids, n):
    with torch.no_grad():
        out = model.generate(torch.tensor([ids]), max_new_tokens=n, do_sample=False)
    return out[0, len(ids) :].tolist()


def _run_engine(model_path, prompts_ids, n_new, dtype):
    from sgl_jax.srt.entrypoints.engine import Engine

    engine = Engine(
        model_path=model_path,
        trust_remote_code=False,
        device=os.environ.get("OLMO2_DEVICE", "cpu"),
        dtype=dtype,
        attention_backend=os.environ.get("OLMO2_ATTN", "native"),
        tp_size=TP_SIZE,
        mem_fraction_static=0.3,
        max_total_tokens=4096,
        max_running_requests=4,
        chunked_prefill_size=-1,
        disable_radix_cache=True,
        skip_server_warmup=True,
        precompile_token_paddings=[64],
        precompile_bs_paddings=[4],
        log_level="error",
        # CPU `native` backend has no SWA KV pool; the sliding window is still applied in its mask.
        disable_hybrid_swa_memory=os.environ.get("OLMO_DISABLE_HYBRID_SWA") == "1",
    )
    try:
        # One request at a time: on the CPU `native` attention backend, a
        # multi-request EXTEND batch gives wrong results for every request
        # after the first, for the upstream Llama model too (see
        # PORTING_NOTES.md), so the comparison runs at batch size 1.
        outs = [
            engine.generate(
                input_ids=ids,
                sampling_params={"temperature": 0.0, "max_new_tokens": n_new},
                return_logprob=True,
                logprob_start_len=0,
            )
            for ids in prompts_ids
        ]
    finally:
        engine.shutdown()
    return outs


def _compare(test, hf_model, model_path, tok, n_new, atol, dtype, require_greedy):
    prompts_ids = [tok(p)["input_ids"] for p in PROMPTS]
    outs = _run_engine(model_path, prompts_ids, n_new, dtype)
    worst = 0.0
    for p, ids, out in zip(PROMPTS, prompts_ids, outs):
        hf_lp, _ = _hf_logprobs(hf_model, ids)
        jax_lp = np.array([x[0] for x in out["meta_info"]["input_token_logprobs"][1:]])
        diff = float(np.max(np.abs(hf_lp - jax_lp)))
        worst = max(worst, diff)
        hf_new = _hf_greedy(hf_model, ids, n_new)
        jx_new = [x[1] for x in out["meta_info"]["output_token_logprobs"]]
        print(f"[{p!r}] max|dlogprob|={diff:.2e}  hf={hf_new}  jax={jx_new}")
        print(f"    hf : {tok.decode(hf_new)!r}\n    jax: {tok.decode(jx_new)!r}")
        test.assertLess(diff, atol)
        if require_greedy:
            test.assertEqual(hf_new, jx_new)
        else:
            test.assertEqual(hf_new[0], jx_new[0])
    print(f"worst prompt-logprob abs diff: {worst:.3e}")


class TestOlmo2Parity(unittest.TestCase):
    def test_tiny_random_fp32(self):
        torch.manual_seed(0)
        tok = AutoTokenizer.from_pretrained(TOKENIZER)
        cfg = Olmo2Config(
            vocab_size=len(tok),
            hidden_size=512,  # head_dim=128 (matches real OLMo 2; kernels pad smaller heads)
            intermediate_size=1024,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,  # exercise GQA
            max_position_embeddings=512,
            rms_norm_eps=1e-6,
            rope_theta=500000.0,
            tie_word_embeddings=False,
            pad_token_id=tok.pad_token_id,
            bos_token_id=tok.bos_token_id,
            eos_token_id=tok.eos_token_id,
        )
        model = Olmo2ForCausalLM(cfg).float().eval()
        # randomize norm weights so a wrong norm placement can't hide behind ones
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "norm" in n:
                    p.copy_(1.0 + 0.2 * torch.randn_like(p))
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d, safe_serialization=True)
            tok.save_pretrained(d)
            _compare(self, model, d, tok, 8, 1e-3, "float32", require_greedy=True)

    @unittest.skipUnless(os.environ.get("OLMO2_REAL_MODEL"), "set OLMO2_REAL_MODEL")
    def test_real_checkpoint(self):
        path = os.environ["OLMO2_REAL_MODEL"]
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()
        atol = float(os.environ.get("OLMO2_REAL_ATOL", "2e-2"))
        _compare(self, model, path, tok, 8, atol, "float32", require_greedy=False)


if __name__ == "__main__":
    unittest.main()
