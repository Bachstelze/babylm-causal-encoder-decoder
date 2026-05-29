BabyLM 2026 Baselines
=====================

This repository contains the code used to train the baselines for the 2026 iteration of the BabyLM Challenge:

- `causal-encoder-decoder` — Adapted encoder-decoder from GPT2 for Strict / Strict-Small / Multilingual tracks (see below).
- `causal_activation` — Clone of [causal generation](https://gitlab.com/Bachstelze/causal_generation)
- `strict-gpt2` — Code for the naive GPT-2 baseline for Strict/Strict-Small/Multilingual tracks
- `strict-interaction` — Past year's GPT2 Interaction track baseline adapted to Strict/Strict-Small.

---

## Files modified since the fork

Add the folder `causal-encoder-decoder` as clone from `strict-gpt2`.

| File | What changed |
|---|---|
| `causal-encoder-decoder/models.py` | Replaced `GPT2LMHeadModel` with `EncoderDecoderModel`. Added `_build_encoder_config()` and `_build_decoder_config()` to independently configure encoder/decoder layers, embedding dim, heads, feed-forward inner dim, and activation function. Supports both `from_config` and `from_pretrained` paths. Uses `EncoderDecoderConfig.from_encoder_decoder_configs()`. |
| `causal-encoder-decoder/training.py` | Swapped the forward pass from manual log-softmax loss computation to `model(**batch)` where the batch is `{input_ids, attention_mask, labels}` — `EncoderDecoderModel` handles the decoder shift and causal mask internally. |
| `causal-encoder-decoder/data_utils.py` | Replaced `FullBabyLMDataset` with `FullEncoderDecoderDataset` that splits each chunk into source (first N%) and target (remaining + bos/eos). Collate function pads source and target separately, masks pad tokens in labels with `-100`. |
| `causal-encoder-decoder/utils.py` | Added CLI arguments for all encoder/decoder config knobs (`--encoder_n_layer`, `--decoder_activation_function`, etc.) and `--source_ratio`, `--from_pretrained`, `--model_name_or_path`, `--decoder_start_token_id`. |
| `causal-encoder-decoder/config.yaml` | Added all new configuration keys with sensible defaults (6-layer, 768-embd encoder and decoder, `gelu_new` activations, 50% source ratio).

## `causal-encoder-decoder`: Encoder-Decoder Baseline

The `causal-encoder-decoder` pipeline trains a configurable **encoder-decoder** transformer built from independent GPT-2 encoder/decoder stacks (Hugging Face `EncoderDecoderModel`).  The decoder is trained **causally** (autoregressive next-token prediction conditioned on the encoder output).

### Architecture features

| Feature | Configuration |
|---|---|
| **Encoder size** | `encoder_n_layer`, `encoder_n_embd`, `encoder_n_head`, `encoder_n_inner` |
| **Decoder size** | `decoder_n_layer`, `decoder_n_embd`, `decoder_n_head`, `decoder_n_inner` |
| **Activation (per module)** | `encoder_activation_function`, `decoder_activation_function` |
| **Source/target split** | `source_ratio` (fraction of chunk fed to encoder, default 0.5) |
| **Pretrained checkpoint** | `from_pretrained` + `model_name_or_path` |

### Data pipeline

Each text chunk is split into **source tokens** (encoder input) and **target tokens** (decoder labels, wrapped in `<s>` / `</s>`).  Padding positions in the labels are masked with `-100` so the loss ignores them.  `EncoderDecoderModel` internally shifts labels right and prepends `decoder_start_token_id` to create `decoder_input_ids`, applying causal masking.

### Example runs

```bash
# Symmetric 6L/6L encoder-decoder (defaults)
python training.py

# Asymmetric: small encoder, large decoder, different activations
python training.py \
  --encoder_n_layer 2 --encoder_n_embd 256 --encoder_n_head 4 \
  --decoder_n_layer 12 --decoder_n_embd 768 --decoder_n_head 12 \
  --encoder_activation_function relu \
  --decoder_activation_function gelu_new \
  --source_ratio 0.3
```

### Causal activation replacement

The `causal_activation/` package provides **causal reduction functions** that replace standard element-wise activations (GELU, ReLU, etc.) in GPT-2 MLP layers.  Based on ["Breaking the Attention Bottleneck"](https://arxiv.org/html/2406.10906v1), these functions perform pairwise token reduction along the sequence dimension, introducing a structured inter-token operation without learned parameters.

#### Usage

**strict-gpt2** — pass `--causal_activation`:

```bash
python training.py --causal_activation matrix
```

**causal-encoder-decoder** — patch encoder and decoder independently:

```bash
python training.py \
  --encoder_causal_activation matrix \
  --decoder_causal_activation context
```

Available modes:

| Mode | Description |
|---|---|
| `matrix` | Pairwise min with previous token |
| `context` | Pairwise min + global mean (causal_context_unity_matrix) |
| `max_matrix` | Pairwise max with previous token |
| `max_context` | Pairwise max + global mean |
| `min_context` | Pairwise min + global min comparison |
| `mean` | Pairwise mean with previous token |

#### How it works

After model creation, `patch_gpt2_activations()` walks through every `GPT2MLP` layer and replaces `self.act` with a `CausalActivation(nn.Module)` wrapper.  The wrapper applies a `torch.vmap`-vectorized causal reduction across the batch dimension, so the operation is fully differentiable and GPU-friendly.

#### Manual programmatic use

```python
from causal_activation import CausalActivation, patch_gpt2_activations, register_causal_activations

# Option A: layer patching (recommended)
model = GPT2LMHeadModel(config)
patch_gpt2_activations(model, mode="matrix")           # all layers
patch_gpt2_activations(model, mode="context",
                       target_submodules=["encoder"])    # encoder only

# Option B: ACT2FN monkey-patch (call before creating any config)
register_causal_activations()
config.activation_function = "causal_matrix"  # now resolves correctly
```
