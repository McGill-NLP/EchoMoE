# Method

Expert parallelism sends every token's hidden state to the nodes that hold its experts (dispatch)
and every expert's output back (combine). EchoMoE compresses both all-to-alls. This page describes
what it sends, how its compressors are fit and trained, and where each part lives in the code.

## How EchoMoE works

**Dispatch.** Every `s`-th MoE layer is a *reference layer*: it publishes the token's full-width
input `x_l` once to every remote node, which caches it as the reference `z`. The `s - 1`
*non-reference layers* after it send only what the reference fails to predict, a code of width
`r_d` of the difference,

    c_d = C_d (x_l - a * z),        x_hat = D_d c_d + G z

where `C_d` and `D_d` are the dispatch compressor and decompressor, `a` is a per-dimension gain and
`G` is the reconstruction gain. In this code `G` is diagonal, `G = diag(b)`. `dplain`
configurations use plain compression and send `C_d x_l` with no reference.

**Combine.** Each expert sends `C_c y_i`, a code of width `r_c`. The token's owner decompresses the
weighted sum once, `y_hat = D_c sum_i w_i C_c y_i`, and optionally adds an owner-side correction
that sends nothing (bias and gated bias decoding, `bias` and `biasgate`).

**Initialization.** `C_d`, `D_d`, `C_c`, `D_c` and the gains come from a PCA of FP64 activation
moments captured from the uncompressed model. The student is then distilled from the teacher with
a coarsened top-1024 KL plus a 0.01-weighted LM loss, either in dev mode (backbone frozen, 1,000
updates) or with the backbone and compressors trained together (*full-model adaptation*, 4,152
updates).

**Folding.** The dispatch decompressor and the combine compressor sit next to expert matrices, so
for serving they are multiplied in. Expert `i` reads `[z, c_d]` through the folded input projection
`W_in,i = W_i [G, D_d]`, where `W_i` stacks its gate and up projections, and writes its code
through the folded output projection `W_out,i = C_c W_down,i`. The fold is computed in float64 and
checked to a relative error of 1e-6.

## Configurations

A configuration (s, r_d, r_c) is the run `dres{r_d}-s{s}-c{r_c}`. For example, `dres512-s4-c1280`
sends a dispatch code of width 512 against a reference every 4 layers and a combine code of width
1280. `dplain{r_d}-c{r_c}` uses plain compression, and `dfull` / `cfull` leave a side uncompressed.

## Implementation reference

1. **Codec:** [`echomoe/codec.py`](../echomoe/codec.py). The compressors and the compressed MoE forward.
2. **Byte model:** [`echomoe/communication_cost.py`](../echomoe/communication_cost.py). The inter-node
   bytes per token of a configuration, priced with the measured routing fanout.
3. **Fit:** [`echomoe/fit.py`](../echomoe/fit.py). The activation moments and the closed-form fit of
   every compressor.
4. **Training:** [`echomoe/training/`](../echomoe/training/). Megatron-Core and DeepEP training of the
   teacher (SFT) and the students (distillation), and the data preparation.
5. **Export and fold:** [`echomoe/inference/evaluation/export.py`](../echomoe/inference/evaluation/export.py).
   The Hugging Face export, with the codecs folded into the expert weights.
6. **vLLM serving:** [`echomoe/inference/evaluation/model.py`](../echomoe/inference/evaluation/model.py).
   The folded model classes and the vLLM plugin that registers them.
7. **Throughput runtime:** [`echomoe/inference/throughput/`](../echomoe/inference/throughput/). The
   expert-parallel runtime on vLLM 0.29 with DeepEP, online serving and the benches.
8. **Transformers remote code:** [`echomoe/hf/`](../echomoe/hf/). The folded students for
   `transformers`, shipped in each Hugging Face repo.
9. **LSH-MoE:** [`echomoe/lsh.py`](../echomoe/lsh.py). The baseline of Tables 15 and 16.

The codec, the byte model and the fit need only PyTorch and run on CPU.

## Names in the code

| Report | Code |
| --- | --- |
| compressors `C_d`, `C_c` and decompressors `D_d`, `D_c` | `encoder` and `decoder` of `dispatch_compressor` and `combine_compressor` |
| gain `a`, reconstruction gain `G = diag(b)` | `encoder_gains`, `decoder_gains` (the vector `b`) |
| folded input projection `W_in,i` | `readers` |
| folded output projection `W_out,i` | `factors` |
| code widths `r_d`, `r_c` | `rank` (`dispatch_rank`, `combine_rank`) |
| non-reference layer | dispatch role `consumer` |
| dev mode (backbone frozen) | `frozen` recipes in `configs/training/qwen3-30b-a3b/frozen/` |
| configuration (s, r_d, r_c) | run ID `dres{r_d}-s{s}-c{r_c}` |
