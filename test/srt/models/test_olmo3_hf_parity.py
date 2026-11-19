"""Olmo 3 numerical parity vs HF transformers, runnable on CPU.

Same harness as test_olmo2_hf_parity.py, with the two Olmo 3 additions turned on:
interleaved sliding-window attention (layer_types, window shorter than the prompt so it
actually masks) and YaRN RoPE.

    JAX_PLATFORMS=cpu python -m pytest -s test/srt/models/test_olmo3_hf_parity.py

Set OLMO3_REAL_MODEL=allenai/Olmo-3-7B-Instruct to also check a real checkpoint.
"""

import math
import os
import tempfile
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import torch
from test_olmo2_hf_parity import TOKENIZER, _compare
from transformers import AutoModelForCausalLM, AutoTokenizer, Olmo3Config, Olmo3ForCausalLM


class TestOlmo3Parity(unittest.TestCase):
    def test_tiny_random_fp32(self):
        torch.manual_seed(0)
        tok = AutoTokenizer.from_pretrained(TOKENIZER)
        factor = 8.0
        cfg = Olmo3Config(
            vocab_size=len(tok),
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            rms_norm_eps=1e-6,
            sliding_window=3,  # shorter than every prompt, so the window really masks
            layer_types=["sliding_attention"] * 3 + ["full_attention"],
            rope_parameters={
                "rope_type": "yarn",
                "rope_theta": 500000.0,
                "factor": factor,
                "original_max_position_embeddings": 64,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "attention_factor": 0.1 * math.log(factor) + 1.0,
            },
            tie_word_embeddings=False,
            pad_token_id=tok.pad_token_id,
            bos_token_id=tok.bos_token_id,
            eos_token_id=tok.eos_token_id,
        )
        model = Olmo3ForCausalLM(cfg).float().eval()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "norm" in n:
                    p.copy_(1.0 + 0.2 * torch.randn_like(p))
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d, safe_serialization=True)
            tok.save_pretrained(d)
            _compare(self, model, d, tok, 8, 1e-3, "float32", require_greedy=True)

    @unittest.skipUnless(os.environ.get("OLMO3_REAL_MODEL"), "set OLMO3_REAL_MODEL")
    def test_real_checkpoint(self):
        path = os.environ["OLMO3_REAL_MODEL"]
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()
        atol = float(os.environ.get("OLMO3_REAL_ATOL", "2e-2"))
        _compare(self, model, path, tok, 8, atol, "float32", require_greedy=False)


if __name__ == "__main__":
    unittest.main()
