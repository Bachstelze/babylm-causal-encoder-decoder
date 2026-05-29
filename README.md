BabyLM 2026 Baselines
=====================

This repository contains the code used to train the baselines for the 2026 iteration of the BabyLM Challenge:

- `causal-encoder-decoder` — Adapted encoder-decoder from GPT2 for Strict / Strict-Small / Multilingual tracks (see below).
- `strict-gpt2`  — Code for the naive GPT-2 baseline for Strict/Strict-Small/Multilingual tracks
- `strict-interaction` — Past year's GPT2 Interaction track baseline adapted to Strict/Strict-Small.

---

## Files modified since the fork

Add the folder `causal-encoder-decoder`.

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

### Adding a custom activation function

The activation function is chosen via a string (`"gelu_new"`, `"relu"`, `"gelu"`, `"silu"`, ...) stored in `GPT2Config.activation_function`.  Hugging Face resolves this string to a PyTorch callable using the `ACT2FN` dictionary in `transformers/activations.py`.  To add a new activation (e.g. `"custom_swish"`):

1. **Option A — global monkey-patch** (simplest, no HF fork needed):

   ```python
   from transformers.activations import ACT2FN

   def custom_swish(x):
       return x * torch.sigmoid(2.0 * x)

   ACT2FN["custom_swish"] = custom_swish
   ```

   Place this at the top of `models.py` (before building any config) so the string is resolved when `GPT2Config` constructs the model layers.

2. **Option B — layer patching after model creation**:

   ```python
   import torch.nn as nn

   class CustomSwish(nn.Module):
       def forward(self, x):
           return x * torch.sigmoid(2.0 * x)

   # Patch every GPT2MLP in the encoder and decoder
   for module in student.modules():
       if hasattr(module, "act"):
           module.act = CustomSwish()
   ```

   This approach works if you need a stateful/parameterized activation (e.g. learned slope).

3. **Option C — fork HF transformers** and add your function to `ACT2FN` in `src/transformers/activations.py`.
