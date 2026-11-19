"""CPU: Gemma 3 text matches HF transformers on tiny random checkpoints.

Covers the 1B layout (default RoPE on both layer types) and the 4B+ layout
(linear RoPE scaling on global layers), each with a sliding window shorter
than the longest prompt. Requires network access to fetch configs/tokenizers.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


class TestGemma3TinyParity(unittest.TestCase):
    def _run(self, arch):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, arch)
            subprocess.run(
                [sys.executable, os.path.join(HERE, "tiny_hf_parity.py"), arch, out],
                check=True,
                env={**os.environ, "JAX_PLATFORMS": "cpu"},
            )
            report = json.load(open(os.path.join(out, "parity.json")))
        self.assertTrue(report["all_match"], report)

    def test_gemma3_1b_layout(self):
        self._run("gemma3")

    def test_gemma3_4b_layout_linear_rope(self):
        self._run("gemma3lin")


if __name__ == "__main__":
    unittest.main()
