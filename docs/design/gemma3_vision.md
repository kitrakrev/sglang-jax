# Design note: Gemma 3 vision (`Gemma3ForConditionalGeneration`) on sgl_jax

Status: **design only**. The text decoder is supported (`gemma3.py`, branch `add-gemma3`).
This note records why the vision path is not a small port, and what it would take.

## What Gemma 3 vision needs
1. **SigLIP vision tower** (27 layers, 1152 hidden, 896x896 input, patch 14, so 4096 patches)
   -> 4x4 average pool to 256 tokens -> Gemma RMSNorm (`mm_soft_emb_norm`) -> linear
   projection (`mm_input_projection_weight`). This is self-contained and comparable to the
   InternViT tower in `internvl.py` (fixed-size crops, so it fits the lane-packed encoder
   contract with merge_unit=16). Optional pan-and-scan adds more crops of the same shape.
2. **Bidirectional attention among the tokens of one image inside the language model.**
   HF builds the decoder mask as `causal OR (same image block)`
   (`token_type_ids_mask_function` in `modeling_gemma3.py`). Each image's 256 soft tokens attend
   to each other in both directions, in every decoder layer (sliding and global). Text tokens
   stay causal. Without this, image-token hidden states, and every later token, differ from HF.
   A causal-only implementation would still "work" (produce plausible text) but would not match
   the reference, so it is not acceptable as a port.

## Why (2) is the hard part in sgl_jax today
- The FA backend (`ragged_paged_attention_v3`) already accepts a dense `custom_mask`
  `[total_q_rows, 1, W]` (kv on the lane axis, W 128-aligned) and switches `causal=0` when one
  is present. That plumbing is used only for speculative-decoding verify
  (`flashattention_backend.py`, `metadata.custom_mask = batch.spec_info_padded.custom_mask`).
  Reusing it for image prefill needs:
  - **Mask construction for EXTEND batches with image items**: per request,
    rows = new tokens, cols = full kv length (prefix + new), value =
    `col <= row OR same_image(row, col)`, packed like `_pack_verify_mask`. For a 4096-token
    prefill chunk against a 4096-token kv, that is 16M int32 entries (64 MB) per batch, built on
    the host every step.
  - **Sliding-window layers**: Gemma 3 is 5:1 local:global. Each layer needs the window
    applied *and* the bidirectional image block. The custom-mask kernel path would need either
    a per-layer mask (two masks: local and global) or the kernel's sliding-window logic combined
    with the custom mask (needs verification; today `causal=0` is the only special case).
  - **Hybrid SWA KV pool**: the SWA page remap (`swa_page_indices`) has to stay consistent with
    the custom-mask column layout.
  - **Kernel cost**: with a custom mask on v5e/v6e the kernel clamps the query tile to 16 rows
    (VMEM limit), so every image prefill would run the slow path. This needs measurement.
- **Chunked prefill must not split an image block.** If a chunk boundary falls inside the 256
  image tokens, the earlier tokens cannot attend to later ones (they are not in the kv cache
  yet). The scheduler (`PrefillAdder` / `add_chunked_req`) would have to extend or shrink a
  chunk to an image boundary.
- **Prefix (radix) cache** stays correct: cached image KV was produced with the full block
  visible, and any reuse covers whole image spans (placeholder ranges are atomic).

## Proposed plan (if pursued)
1. SigLIP tower + projector + processor adapter (HF `Gemma3Processor`, fixed 896px crops),
   with encoder-only parity vs HF `get_image_features` (same method as the InternVL port).
2. `ImageBlockMask` in `ForwardBatch`: per-request image spans for the extend window, built
   from `mm_inputs` placeholder ranges (already available to the scheduler).
3. Flash-attention backend: when a batch has image spans and the model sets
   `bidirectional_image_attention=True`, build a dense custom mask (causal OR same-span,
   AND sliding window for local layers). Start with a dense mask per layer type. Then
   evaluate a cheaper kernel variant that takes span lists instead of a dense mask.
4. Scheduler: align chunk boundaries to image-span boundaries.
5. Tests: tiny-checkpoint CPU parity (native backend needs the same mask), then TPU
   real-checkpoint parity on gemma-3-4b-it, plus a TTFT comparison with and without images to
   cost the masked kernel.

Rough size: SigLIP tower and processor about 400 lines; mask, scheduler and backend changes
about 300 to 500 lines across shared code paths (FA backend, scheduler, forward batch), which
need maintainer agreement before an upstream PR.

## Status of this branch
Contains only this note (based on `add-gemma3`). No vision code is merged here, so as not to
ship a vision path that is knowingly non-equivalent to HF.
