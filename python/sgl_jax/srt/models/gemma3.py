"""Inference-only Gemma 3 text model.

Gemma 3 vs Gemma 2:
- q/k RMSNorm (Gemma-style ``1 + w``) on each head before RoPE,
- no attention / final logit soft-capping,
- per-layer-type RoPE: local (sliding) layers use ``rope_local_base_freq``
  (10k), global layers use ``rope_theta`` (1M) with optional linear scaling,
- sliding window on 5 of every 6 layers (``layer_types``).

``Gemma3ForConditionalGeneration`` checkpoints (4B/12B/27B) are served
text-only here: the language-model weights are loaded and the vision tower /
projector are skipped.
"""

import logging

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, get_rope
from sgl_jax.srt.layers.layernorm import GemmaRMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.model_loader.weights import WeightLoader, WeightSpec
from sgl_jax.srt.models.gemma2 import Gemma2MLP

logger = logging.getLogger(__name__)


def _layer_type(config: PretrainedConfig, layer_id: int) -> str:
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        return layer_types[layer_id]
    pattern = getattr(config, "sliding_window_pattern", 6)
    return "full_attention" if (layer_id + 1) % pattern == 0 else "sliding_attention"


def _rope_for_layer(config: PretrainedConfig, layer_type: str) -> tuple[float, dict | None]:
    """Return (rope_theta, rope_scaling) for a layer type.

    Handles both the transformers>=5 per-layer-type ``rope_parameters`` and
    the older flat ``rope_theta`` / ``rope_local_base_freq`` / ``rope_scaling``.
    """
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    if layer_type in rope_parameters:
        params = dict(rope_parameters[layer_type])
    elif layer_type == "sliding_attention":
        params = {
            "rope_type": "default",
            "rope_theta": getattr(config, "rope_local_base_freq", 10000.0),
        }
    else:
        params = dict(getattr(config, "rope_scaling", None) or {"rope_type": "default"})
        params.setdefault("rope_theta", getattr(config, "rope_theta", 1000000.0))
    theta = params.pop("rope_theta")
    if "rope_type" not in params and "type" not in params:
        params["rope_type"] = "default"
    return theta, params


class Gemma3Attention(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        attention_bias = getattr(config, "attention_bias", False)

        layer_type = _layer_type(config, layer_id)
        is_sliding = layer_type == "sliding_attention"
        rope_theta, rope_scaling = _rope_for_layer(config, layer_type)

        self.q_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.num_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.k_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.num_kv_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.v_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.num_kv_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.o_proj = LinearBase(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            use_bias=attention_bias,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, epsilon=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, epsilon=config.rms_norm_eps)

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            rope_scaling=rope_scaling,
            dtype=dtype,
        )
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=config.query_pre_attn_scalar**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=config.sliding_window if is_sliding else 0,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(forward_batch.positions, q, k)
        attn_output, kv_fused = self.attn(
            q, k, v, forward_batch=forward_batch, token_to_kv_pool=token_to_kv_pool
        )
        output, _ = self.o_proj(attn_output)
        return output, kv_fused


class Gemma3DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.self_attn = Gemma3Attention(config, layer_id=layer_id, mesh=mesh, dtype=dtype)
        self.mlp = Gemma2MLP(
            config.hidden_size,
            config.intermediate_size,
            layer_id=layer_id,
            dtype=dtype,
            mesh=mesh,
        )

        def norm():
            return GemmaRMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

        self.input_layernorm = norm()
        self.post_attention_layernorm = norm()
        self.pre_feedforward_layernorm = norm()
        self.post_feedforward_layernorm = norm()

    def __call__(
        self,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, kv_fused = self.self_attn(hidden_states, forward_batch, token_to_kv_pool)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, kv_fused


class Gemma3TextModel(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.embed_tokens = Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )
        self.layers = nnx.data(
            [
                Gemma3DecoderLayer(config, layer_id=i, dtype=dtype, mesh=mesh)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)
        self.hidden_size = config.hidden_size

    def embed(self, input_ids: jax.Array) -> jax.Array:
        hidden_states = self.embed_tokens(input_ids)
        # HF casts the sqrt(hidden_size) normalizer to the weight dtype.
        return hidden_states * jnp.array([self.hidden_size**0.5], dtype=hidden_states.dtype)

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        input_embeds: jax.Array | None = None,
    ):
        hidden_states = (
            self.embed(forward_batch.input_ids) if input_embeds is None else input_embeds
        )
        layers_kv_fused = []
        for layer in self.layers:
            hidden_states, kv_fused = layer(hidden_states, forward_batch, token_to_kv_pool)
            layers_kv_fused.append(kv_fused)
        hidden_states = self.norm(hidden_states)
        return hidden_states, layers_kv_fused


class Gemma3ForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = getattr(config, "text_config", config)
        self.dtype = dtype
        self.model = Gemma3TextModel(self.config, dtype=dtype, mesh=mesh)
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            soft_cap=getattr(self.config, "final_logit_softcapping", None),
            mesh=self.mesh,
            enable_dp_lm_head=getattr(config, "enable_dp_lm_head", False),
        )

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self, model_config=model_config, mesh=self.mesh, dtype=self.dtype
        )
        mappings = self._create_weight_mappings()
        if not loader.dummy_mode:
            # Checkpoints name the language model "model.*" (Gemma3ForCausalLM),
            # "language_model.model.*" (older Gemma3ForConditionalGeneration) or
            # "model.language_model.*" (transformers>=4.52). Keep only the keys
            # present; everything else (vision tower, projector) is skipped.
            present = loader.metadata
            mappings = {k: v for k, v in mappings.items() if k in present}
            if not mappings:
                raise ValueError("No Gemma 3 language-model weights found in checkpoint")
        loader.load(mappings)
        logger.info("Gemma3 weights loaded successfully!")

    def _create_weight_mappings(self) -> dict:
        base = {
            "model.embed_tokens.weight": WeightSpec(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightSpec(
                target_path="model.norm.weight", sharding=(None,), transpose=False
            ),
        }
        for layer_idx in range(self.config.num_hidden_layers):
            base.update(self._create_layer_mappings(layer_idx))

        mappings = dict(base)
        for k, v in base.items():
            mappings[f"language_model.{k}"] = v
            mappings[k.replace("model.", "model.language_model.", 1)] = v
        return mappings

    def _create_layer_mappings(self, layer_idx: int) -> dict:
        prefix = f"model.layers.{layer_idx}"
        mappings = {}
        for norm in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "self_attn.q_norm",
            "self_attn.k_norm",
        ):
            mappings[f"{prefix}.{norm}.weight"] = WeightSpec(
                target_path=f"{prefix}.{norm}.weight", sharding=(None,), transpose=False
            )
        for proj, is_kv in (("q_proj", False), ("k_proj", True), ("v_proj", True)):
            mappings[f"{prefix}.self_attn.{proj}.weight"] = WeightSpec(
                target_path=f"{prefix}.self_attn.{proj}.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=is_kv,
            )
        mappings[f"{prefix}.self_attn.o_proj.weight"] = WeightSpec(
            target_path=f"{prefix}.self_attn.o_proj.weight",
            sharding=("tensor", None),
            transpose=True,
            head_dim_padding=True,
            kv_head_padding=False,
        )
        for proj in ("gate_proj", "up_proj"):
            mappings[f"{prefix}.mlp.{proj}.weight"] = WeightSpec(
                target_path=f"{prefix}.mlp.{proj}.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )
        mappings[f"{prefix}.mlp.down_proj.weight"] = WeightSpec(
            target_path=f"{prefix}.mlp.down_proj.weight",
            sharding=("tensor", None),
            transpose=True,
        )
        return mappings

    def get_embed_and_head(self):
        emb = self.model.embed_tokens.embedding.value
        return emb, emb

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        kv_pool = memory_pools.token_to_kv_pool
        hidden_states, layers_kv_fused = self.model(forward_batch, kv_pool)
        output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)
        return output, {"token_to_kv_pool": layers_kv_fused}, True, None


EntryClass = Gemma3ForCausalLM
