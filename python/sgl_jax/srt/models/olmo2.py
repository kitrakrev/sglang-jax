"""OLMo 2 (allenai/OLMo-2-*) for sgl-jax.

Derived from qwen3.py. Differences: post-sublayer norms (no input norm),
QK-norm over the full projected q/k instead of per head.
"""

import logging
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, get_rope
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.model_loader.weights import WeightLoader, WeightSpec
from sgl_jax.srt.precision_tracer import precision_tracer
from sgl_jax.srt.utils.profiling_utils import named_scope

logger = logging.getLogger(__name__)

init_fn = nnx.initializers.uniform()


def _get_rope_parameters(config: PretrainedConfig) -> dict:
    """transformers>=5 exposes `rope_parameters`; older configs use rope_theta/rope_scaling."""
    params = getattr(config, "rope_parameters", None)
    if params:
        return dict(params)
    params = dict(getattr(config, "rope_scaling", None) or {})
    params.setdefault("rope_theta", getattr(config, "rope_theta", 10000))
    return params


class Olmo2Attention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int,
        mesh: jax.sharding.Mesh,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        head_dim: int | None = None,
        rms_norm_eps: float = None,
        layer_id: int = 0,
        attention_bias: bool = False,
        sliding_window_size: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        assert num_heads % num_kv_heads == 0
        self.head_dim = head_dim or hidden_size // num_heads
        self.q_head_num = num_heads
        self.kv_head_num = num_kv_heads

        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.mesh = mesh

        # OLMo 2: QK-norm is an RMSNorm over the *whole* projected q / k
        # (num_heads * head_dim), not per head as in Qwen3. Under TP the
        # reduction axis is sharded on "tensor"; XLA inserts the all-reduce.
        self.q_norm = RMSNorm(
            self.q_size, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="q_norm"
        )
        self.k_norm = RMSNorm(
            self.kv_size, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="k_norm"
        )

        self.q_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="q_proj",
        )
        self.k_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="k_proj",
        )
        self.v_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="v_proj",
        )
        self.o_proj = LinearBase(
            input_size=num_heads * self.head_dim,
            output_size=hidden_size,
            use_bias=attention_bias,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            rope_scaling=rope_scaling,
            dtype=dtype,
        )

        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window_size,
        )

    @named_scope
    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.reshape(
            -1,
            self.q_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        k = k.reshape(
            -1,
            self.kv_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        v = v.reshape(
            -1,
            self.kv_head_num,
            self.head_dim,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )

        q, k = self.rotary_emb(positions, q, k)
        attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)

        output, _ = self.o_proj(attn_output, out_sharding=out_sharding)
        return output, kv_fused


class Olmo2MLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.layer_id = layer_id

        self.gate_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="gate_proj",
        )

        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="up_proj",
        )

        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="down_proj",
        )

        self.act_fn = jax.nn.silu

    @named_scope
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        *,
        out_sharding: jax.sharding.Sharding | None = None,
    ):
        a1, _ = self.gate_proj(hidden_states)
        a2, _ = self.up_proj(hidden_states)
        intermediate_parallel = a2 * self.act_fn(a1)
        output, _ = self.down_proj(intermediate_parallel, out_sharding=out_sharding)
        return output


class Olmo2DecoderLayer(nnx.Module):
    """OLMo 2 block: *post*-norm on each sub-layer output, no input norm.

    h = x + post_attention_layernorm(attn(x))
    y = h + post_feedforward_layernorm(mlp(h))
    """

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        rope_parameters = _get_rope_parameters(config)
        rope_theta = rope_parameters.get("rope_theta", 10000)
        rope_scaling = rope_parameters
        if rope_scaling.get("rope_type", "default") == "default":
            rope_scaling = None
        max_position_embeddings = getattr(config, "max_position_embeddings", 4096)
        head_dim = getattr(config, "head_dim", None)
        # Olmo 3: per-layer sliding window (3 sliding : 1 full). OLMo 2 has no layer_types.
        layer_types = getattr(config, "layer_types", None)
        sliding_window_size = 0
        if layer_types is not None and layer_types[layer_id] == "sliding_attention":
            sliding_window_size = getattr(config, "sliding_window", 0) or 0
        self.self_attn = Olmo2Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            rms_norm_eps=config.rms_norm_eps,
            layer_id=layer_id,
            attention_bias=getattr(config, "attention_bias", False),
            sliding_window_size=sliding_window_size,
            dtype=dtype,
            mesh=mesh,
        )
        self.mlp = Olmo2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            layer_id=layer_id,
            dtype=dtype,
            mesh=mesh,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="post_attention_layernorm",
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="post_feedforward_layernorm",
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        layer_callback_flag = []
        residual = hidden_states
        attn_out, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )
        layer_callback_flag.append(
            precision_tracer.jit_pure_callback_record(
                attn_out, "self_attn_output", "SELF_ATTN", self.layer_id
            )
        )
        hidden_states = residual + self.post_attention_layernorm(attn_out)

        residual = hidden_states
        mlp_out = self.mlp(hidden_states)
        layer_callback_flag.append(
            precision_tracer.jit_pure_callback_record(mlp_out, "mlp_output", "MLP", self.layer_id)
        )
        hidden_states = residual + self.post_feedforward_layernorm(mlp_out)
        return hidden_states, kv_fused, layer_callback_flag


class Olmo2Model(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            kernel_axes=("tensor", None),
            param_dtype=dtype,
            mesh=mesh,
        )
        self.layers = nnx.data(
            [
                Olmo2DecoderLayer(config=config, layer_id=i, dtype=dtype, mesh=mesh)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="norm",
        )
        # For EAGLE3 support
        self.layers_to_capture = []

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        positions: jax.Array | None = None,
    ):
        hidden_states = (
            self.embed_tokens(forward_batch.input_ids)
            if forward_batch.input_embedding is None
            else forward_batch.input_embedding
        )
        layers_kv_fused = []
        layers_callback_flag = []
        aux_hidden_states = []
        positions = forward_batch.positions if positions is None else positions
        for layer_id, layer in enumerate(self.layers):
            if layer_id in self.layers_to_capture:
                aux_hidden_states.append(hidden_states)
            hidden_states, kv_fused, callback_flag = layer(
                positions, hidden_states, forward_batch, token_to_kv_pool
            )
            layers_kv_fused.append(kv_fused)
            layers_callback_flag.extend(callback_flag)

        hidden_states = self.norm(hidden_states)
        layers_callback_flag.append(
            precision_tracer.jit_pure_callback_record(
                hidden_states, "transformer_output", "TRANSFORMER"
            )
        )
        return hidden_states, aux_hidden_states, layers_kv_fused, layers_callback_flag


class Olmo2ForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        logger.info("Olmo2ForCausalLM config dtype: %s", self.dtype)
        self.model = Olmo2Model(config, dtype=self.dtype, mesh=mesh)
        if not getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                mesh=mesh,
                enable_dp_lm_head=getattr(config, "enable_dp_lm_head", False),
            )
        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            mesh=self.mesh,
            enable_dp_lm_head=getattr(config, "enable_dp_lm_head", False),
        )

        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )

        weight_mappings = self._create_olmo2_weight_mappings()

        loader.load(weight_mappings)
        logger.info("OLMo 2 weights loaded successfully!")

    def _create_olmo2_weight_mappings(self) -> dict:
        mappings = {
            "model.embed_tokens.weight": WeightSpec(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightSpec(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
        }

        if not getattr(self.config, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = self.lm_head.weight_mapping("lm_head.embedding")

        num_layers = self.config.num_hidden_layers
        for layer_idx in range(num_layers):
            layer_mappings = self._create_layer_mappings(layer_idx)
            mappings.update(layer_mappings)

        return mappings

    def _create_layer_mappings(self, layer_idx: int) -> dict:
        prefix = f"model.layers.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"

        mappings = {
            f"{prefix}.post_feedforward_layernorm.weight": WeightSpec(
                target_path=f"{target_prefix}.post_feedforward_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_attention_layernorm.weight": WeightSpec(
                target_path=f"{target_prefix}.post_attention_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.self_attn.q_proj.weight": WeightSpec(
                target_path=f"{target_prefix}.self_attn.q_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                kv_head_padding=False,
            ),
            f"{prefix}.self_attn.k_proj.weight": WeightSpec(
                target_path=f"{target_prefix}.self_attn.k_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                kv_head_padding=True,
            ),
            f"{prefix}.self_attn.v_proj.weight": WeightSpec(
                target_path=f"{target_prefix}.self_attn.v_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                kv_head_padding=True,
            ),
            f"{prefix}.self_attn.o_proj.weight": WeightSpec(
                target_path=f"{target_prefix}.self_attn.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
                kv_head_padding=False,
            ),
            f"{prefix}.self_attn.q_norm.weight": WeightSpec(
                target_path=f"{target_prefix}.self_attn.q_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.self_attn.k_norm.weight": WeightSpec(
                target_path=f"{target_prefix}.self_attn.k_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.mlp.gate_proj.weight": WeightSpec(
                target_path=f"{target_prefix}.mlp.gate_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.up_proj.weight": WeightSpec(
                target_path=f"{target_prefix}.mlp.up_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.down_proj.weight": WeightSpec(
                target_path=f"{target_prefix}.mlp.down_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
        }

        if getattr(self.config, "attention_bias", False):
            bias_mappings = {
                f"{prefix}.self_attn.q_proj.bias": WeightSpec(
                    target_path=f"{target_prefix}.self_attn.q_proj.bias",
                    sharding=(None,),
                    transpose=False,
                    kv_head_padding=False,
                ),
                f"{prefix}.self_attn.k_proj.bias": WeightSpec(
                    target_path=f"{target_prefix}.self_attn.k_proj.bias",
                    sharding=(None,),
                    transpose=False,
                    kv_head_padding=True,
                ),
                f"{prefix}.self_attn.v_proj.bias": WeightSpec(
                    target_path=f"{target_prefix}.self_attn.v_proj.bias",
                    sharding=(None,),
                    transpose=False,
                    kv_head_padding=True,
                ),
                f"{prefix}.self_attn.o_proj.bias": WeightSpec(
                    target_path=f"{target_prefix}.self_attn.o_proj.bias",
                    sharding=(None,),
                    transpose=False,
                ),
            }
            mappings.update(bias_mappings)

        return mappings

    def get_embed_and_head(self):
        return (
            self.model.embed_tokens.embedding.value,
            self.lm_head.embedding.value,
        )

    def set_embed_and_head(
        self,
        embed_weight: jax.Array | None = None,
        head_weight: jax.Array | None = None,
    ) -> None:
        """Set word embedding and LM Head weights.

        Args:
            embed_weight: Embedding matrix with shape [vocab_size, hidden_size].
            head_weight:  LM Head matrix with shape [vocab_size, hidden_size].
        """
        if embed_weight is not None:
            self.model.embed_tokens.embedding.value = embed_weight

        if head_weight is not None:
            self.lm_head.embedding.value = head_weight

    def set_eagle3_layers_to_capture(self, layer_ids: list[int] | None = None):

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]

    def set_dflash_layers_to_capture(self, layer_ids: list[int]):
        if layer_ids is None:
            raise ValueError("DFLASH requires explicit target layer ids.")
        self.capture_aux_hidden_states = True
        self.model.layers_to_capture = [val + 1 for val in layer_ids]

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        kv_pool = memory_pools.token_to_kv_pool
        hidden_states, aux_hidden_states, layers_kv_fused, layers_callback_flag = self.model(
            forward_batch, kv_pool
        )
        if not self.capture_aux_hidden_states:
            aux_hidden_states = None
        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(
                hidden_states, self.lm_head, logits_metadata, aux_hidden_states=aux_hidden_states
            )
        else:
            output = self.logits_processor(
                hidden_states,
                self.model.embed_tokens,
                logits_metadata,
                aux_hidden_states=aux_hidden_states,
            )

        return output, {"token_to_kv_pool": layers_kv_fused}, layers_callback_flag, None


class Olmo3ForCausalLM(Olmo2ForCausalLM):
    """Olmo 3 / 3.1: OLMo 2 block + interleaved sliding-window attention (layer_types) and YaRN RoPE."""


EntryClass = [Olmo2ForCausalLM, Olmo3ForCausalLM]
