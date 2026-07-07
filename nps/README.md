# NPS — Neural Process Library

This directory contains the core neural process (NP) model library. It is organized into five compositional layers that build on each other from low-level operations up to full models.

```
Convolution
    └─► ConvolutionBlock  (residual wrapper + activation)
            └─► CNN  (stack of blocks)
                    └─► Encoder / Decoder  (grid projection + CNN)
                                └─► NeuralProcess  (encoder + decoder + likelihood)
```

---

## Directory Structure

```
nps/
├── models/          # High-level NP model classes (the user-facing API)
├── core/
│   ├── convolutions/        # Individual convolution operations (the core novelty)
│   ├── convolution_blocks/  # Residual wrappers around convolutions
│   ├── cnns/                # Stacks of convolution blocks
│   ├── encoders/            # Context → latent representation
│   ├── decoders/            # Latent → output parameters
│   ├── transformers/        # Attention-based context processors
│   ├── attention_layers/    # Cross-attention primitives
│   ├── attentions/          # Self-attention primitives
│   ├── embedding.py         # Random Fourier Feature embeddings
│   ├── interpolate.py       # Interpolation utilities
│   ├── mlp.py               # Multi-layer perceptron
│   ├── linear.py            # Linear layer variants
│   └── deepset.py           # DeepSet aggregation
├── likelihoods/     # Output likelihood functions
└── utils/           # Grid construction, distances, group actions, helpers
```

---

## Models (`nps/models/`)

All models extend `BaseNeuralProcess`. Off-grid models share one forward
signature:

```python
output: torch.distributions.Distribution = model(xc, yc, xq)
```

`GridConvCNP` is the exception: it takes grid tensors `model(y_mc, y, y_mq)`
(context mask, values, query mask) instead of scattered points.

| Model | Class | Encoder | Decoder | CNN backbone |
|---|---|---|---|---|
| CNP | `CNP` | `CNPEncoder` | `CNPDecoder` | — |
| ACNP | `ACNP` | `ACNPEncoder` | `ACNPDecoder` | — |
| TNP | `TNP` | `TNPEncoder` | `TNPDecoder` | — |
| TETNP | `TETNP` | `TETNPEncoder` | `TETNPDecoder` | — |
| ConvCNP | `ConvCNP` | `ConvCNPEncoder` | `ConvCNPDecoder` | `ConvNet` / `FNO` / `UNet` |
| GridConvCNP | `GridConvCNP` | `GridConvCNPEncoder` | `ConvCNPDecoder` | same |
| SetFourierConvCNP | `SetFourierConvCNP` | `SetFourierConvCNPEncoder` | `SetFourierConvCNPDecoder` | `SetFourierConvNet` |

`BaseNeuralProcess` enforces a three-field structure:
```python
class BaseNeuralProcess(nn.Module):
    encoder:    BaseEncoder
    decoder:    BaseDecoder
    likelihood: BaseLikelihood
```

The forward pass of every model follows the same pattern:
```python
encoder_out = self.encoder(xc, yc, xq)
decoder_out = self.decoder(encoder_out, xq)
return self.likelihood(decoder_out)
```

---

## Convolutions (`nps/core/convolutions/`)

This is the primary research contribution of the codebase. Six convolution types are available, organized from standard to hybrid:

### Standard

| Class | File | Description |
|---|---|---|
| `ConvNd` | `conv.py` | Wraps `nn.Conv1d/2d/3d` with optional interpolation |
| `VolterraConvNd` | `volterraconv.py` | 1st + 2nd order Volterra (nonlinear) convolution |
| `SpectralConv` | `specconv.py` | FFT-based spectral convolution on a regular grid |

### Set-based (off-the-grid, the key novelty)

| Class | File | Description |
|---|---|---|
| `SetConv` | `setconv.py` | Permutation-invariant set convolution via Gaussian kernels |
| `SetFourierConvBase` | `setfourierconvbase.py` | Abstract base: frequency grid, Gaussian kernel, Fourier transforms |
| `SetFourierConv` | `setfourierconv.py` | Set convolution performed in Fourier space |
| `SetFourierVolterraConv` | `setfouriervolterraconv.py` | `SetFourierConv` + 2nd-order Volterra terms |


### SetFourierConv — design notes

`SetFourierConv` is the central operation in the more powerful model variants. Key implementation decisions:

- **Positive half-space only**: The Fourier grid uses only positive frequencies and exploits Hermitian symmetry for the inverse transform. This halves memory and compute vs. a full complex grid.
- **Learnable lengthscale**: A Gaussian envelope `exp(−π‖ω‖²/ρ²)` with learnable per-channel bandwidth `ρ` smoothly localises the frequency response.
- **Low-rank factorisation**: The weight tensor `W` can be factorised as `W = U @ Vᵀ` to reduce parameter count.
- **Grouped convolutions**: Channels can be partitioned into groups for efficiency.
- **Precomputed phase factors**: `compute_translation_operands()` and `compute_ift_operands()` are called once per batch of positions and reused across layers (see `SetFourierConvNet` below).

### SetFourierVolterraConv — design notes

Extends `SetFourierConv` with a 2nd-order Volterra term. The output is:

```
z = z_order1 + sum_r(z_order2_r1 ⊙ z_order2_r2)
```

where `r` indexes `volterra_rank` low-rank factors. A `low_ranks_mixer` linear layer aggregates the product terms.

### VolterraConvNd — design notes

Standard CNN-based Volterra filters. Two modes:
- **Exact** (1D only): `Conv2d` on the outer product `x ⊗ x` followed by diagonal extraction.
- **Low-rank**: Factorised via `2 × volterra_rank + 1` grouped channels, then aggregated by `low_ranks_mixer`.

---

## Convolution Blocks (`nps/core/convolution_blocks/`)

Each convolution block wraps a convolution with a residual connection and activation. All extend `BaseConvolutionBlock`.

| Class | File | Wraps |
|---|---|---|
| `ConvBlock` | `conv_block.py` | `ConvNd` |
| `FourierBlock` | `fourier_block.py` | `SpectralConv` |
| `SetFourierConvBlock` | `setfourierconv_block.py` | `SetFourierConv` |
| `SetFourierVolterraConvBlock` | `setfourierconv_block.py` | `SetFourierVolterraConv` |

The `ResidualBlock` helper in `base.py` applies `layer_post(layer1(x) + layer2(x))` (a two-branch sum, not a bare skip connection), used internally by all block types.

---

## CNNs (`nps/core/cnns/`)

CNNs stack convolution blocks into a feedforward network. All extend `BaseCNN`.

| Class | File | Block type | Notes |
|---|---|---|---|
| `ConvNet` | `cnn.py` | `ConvBlock` | Standard feedforward CNN |
| `FNO` | `fno.py` | `FourierBlock` | Fourier Neural Operator — spectral convolutions on regular grids |
| `SetFourierConvNet` | `sfcnn.py` | `SetFourierConvBlock` or `SetFourierVolterraConvBlock` | See below |
| `UNet` | `unet.py` | `ConvBlock` | Encoder–decoder with skip connections |

### SetFourierConvNet — precomputed phase operators

`SetFourierConvNet` is the backbone for `SetFourierConvCNP`. Because computing the Fourier phase factors for a set of positions is expensive, `SetFourierConvNet` computes them **once** using the first block's convolution layer and passes them to all subsequent blocks:

```python
# Computed once, from the first block's conv layer:
context_translation_ft  = first_conv.compute_translation_operands(xc)
context_ift_operands     = first_conv.compute_ift_operands(xc)
query_ift_operands       = first_conv.compute_ift_operands(xq)
# The query stream reuses context_translation_ft (xkv = xc for both),
# so no separate query translation operand is computed.

# Then passed to every block in the stack.
```

This is the main efficiency mechanism for deep set-Fourier networks.

---

## Encoders (`nps/core/encoders/`)

Encoders map `(xc, yc, xq)` to a latent representation aligned with the query positions.

| Class | File | Mechanism |
|---|---|---|
| `CNPEncoder` | `cnp.py` | MLP per context point → DeepSet mean aggregation |
| `ConvCNPEncoder` | `convcnp.py` | SetConv onto grid → CNN → interpolate to query points |
| `GridConvCNPEncoder` | `convcnp.py` | As above but with binary mask channel for image grids |
| `ACNPEncoder` | `acnp.py` | Cross-attention from query points to context |
| `TNPEncoder` | `tnp.py` | Self-attention transformer over context + query points |
| `TETNPEncoder` | `tetnp.py` | Translation-equivariant transformer encoder |
| `SetFourierConvCNPEncoder` | `sfconvcnp.py` | MLP preprocess → `SetFourierConvNet` directly on scattered points |

### ConvCNP encoding pipeline

```
xc, yc  ──► SetConv (context → grid) ──► CNN on grid ──► SetConv (grid → xq)  ──► encoder_out
```

The grid is built on-the-fly with configurable margin, resolution, and divisibility constraints.

### SetFourierConvCNP encoding pipeline

```
xc, yc  ──► MLP ──► SetFourierConvNet(xc, zc, xq)  ──► encoder_out
```

No grid is needed — the Fourier convolution operates directly on scattered input positions.

---

## Decoders (`nps/core/decoders/`)

Decoders map `(encoder_out, xq)` to the parameters of the output distribution.

| Class | File | Mechanism |
|---|---|---|
| `CNPDecoder` | `cnp.py` | MLP applied per query point |
| `ConvCNPDecoder` | `convcnp.py` | Linear projection of encoder output |
| `ACNPDecoder` | `acnp.py` | MLP applied to attended representation |
| `TNPDecoder` | `tnp.py` | MLP per query token |
| `TETNPDecoder` | `tetnp.py` | MLP per query token (equivariant) |
| `SetFourierConvCNPDecoder` | `sfconvcnp.py` | Linear projection of SetFourierConvNet output |

---

## Transformers (`nps/core/transformers/`)

Used as the context-processing backbone in TNP/TETNP models.

| Class | File | Description |
|---|---|---|
| `TransformerEncoder` | `transformer.py` | Standard self-attention transformer (single stream) |
| `EfficientQueryTransformerEncoder` | `transformer.py` | Two-stream: context→context self-attention, then context→query cross-attention |
| `PerceiverEncoder` | `perceiver.py` | Cross-attention between latent tokens and context |
| `ISTransformerEncoder` | `istransformer.py` | Inducing-set transformer (sparse attention) |
| `TETransformerEncoder` | `tetransformer.py` | Translation-equivariant self-attention transformer |
| `TEEfficientQueryTransformerEncoder` | `tetransformer.py` | TE two-stream (efficient query) |
| `TEISTransformerEncoder` | `teistransformer.py` | TE + inducing-set |
| `TEPerceiverEncoder` | `teperceiver.py` | TE + Perceiver |

Both the plain and TE families offer a standard and an efficient-query form,
as two separate classes:
- **Standard** (`TransformerEncoder` / `TETransformerEncoder`): self-attention over all positions together; accepts an optional attention `mask`.
- **Efficient query** (`EfficientQueryTransformerEncoder` / `TEEfficientQueryTransformerEncoder`): separate context→context self-attention, then context→query cross-attention. This is what `TNPEncoder`/`TETNPEncoder` use.

---

## Likelihoods (`nps/likelihoods/`)

Likelihoods consume the decoder output and return a `torch.distributions.Distribution`.

| Class | File | Description |
|---|---|---|
| `HeteroscedasticNormalLikelihood` | `gaussian.py` | Splits output into mean + log-variance |
| `HomoscedasticNormalLikelihood` | `gaussian.py` | Fixed or learnable scalar noise |

Both support a flexible **transform pipeline** configured via YAML. Transforms are applied sequentially to the mean and scale parameters:

```yaml
# Examples
scale_transform: softplus          # single transform
location_transform:                # sequence
  - softplus
  - {multiply: {value: 2.0}}
scale_transform: {clamp: {min: 0.01, max: 5.0}}
```

Available transforms: `identity`, `sigmoid`, `softplus`, `exp`, `tanh`, `relu`, `abs`, `clamp`, `add`, `multiply`.

---

## Embeddings (`nps/core/embedding.py`)

| Class | Description |
|---|---|
| `BaseEmbedding` | Selects `active_dims`, embeds them, concatenates with remaining dims |
| `RandomFourierFeaturesEmbedding` | Maps inputs to `sin/cos(2π x Ω)` where `Ω` is a fixed random matrix |

RFF embeddings are used to lift low-dimensional coordinates into a higher-dimensional feature space before passing them to MLPs or attention layers.

---

## Utilities (`nps/utils/`)

| File | Contents |
|---|---|
| `grids.py` | Regular grid construction for convolutional encoders |
| `distances.py` | Pairwise distance computations |
| `aggregate.py` | Set aggregation: `Aggregator` (sum/mean/min/max/quantile reductions) and `PMAAggregator` (attention-based pooling with learnable seed queries) |
| `group_actions.py` | Symmetry group transformations |
| `helpers.py` | Miscellaneous tensor utilities |

---

## Adding a New Model

1. **Choose or create a convolution** in `core/convolutions/` extending `BaseConvolution`.
2. **Wrap it in a block** in `core/convolution_blocks/` extending `BaseConvolutionBlock`.
3. **Stack blocks into a CNN** in `core/cnns/` extending `BaseCNN`.
4. **Build an encoder** in `core/encoders/` extending `BaseEncoder`. The encoder receives `(xc, yc, xq)` and returns a representation at the query positions.
5. **Build a decoder** in `core/decoders/` extending `BaseDecoder`. The decoder receives `(encoder_out, xq)` and returns distribution parameters.
6. **Assemble the model** in `models/` extending `BaseNeuralProcess`.
7. **Register forward/metric wrappers** in `utils/experiment/` if the model has a non-standard call signature.
8. **Add a config fragment** at `conf/model/<benchmark>/<name>.yaml` (with a `# @package _global_` header). The benchmark's parameterized composer at `conf/experiment/<benchmark>/default.yaml` then picks it up via the CLI `model/<benchmark>=<name>` override — no new experiment composer file is needed. See the root `README.md` "Adding a new model" section for the full template.
