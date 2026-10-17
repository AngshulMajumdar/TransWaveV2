# Model card: TransWave-Meyer-96M

TransWave-Meyer-96M is a 96.13M-parameter, self-attention-free, causal raw-byte language model.

Architecture ID: `causal_transwave_blocklifting_meyer_byte_lm_g4_v2`

It combines block-local learned lifting, content-dependent cross-scale mixing, a short causal depthwise convolution, a long causal Meyer-structured FIR, gated residual fusion, a SwiGLU feed-forward network and an exact within-block recurrent decoder.

## Input and output

Input: raw byte IDs `0..255`. No external tokenizer is required. Context: up to 32,768 raw bytes. Output objective: exact next-byte likelihood.

## Metrics

The repository includes final stored causal results on arXiv, FineWeb-Edu, PG19 and WikiText-103, plus a separate completed 4.5B-token release-scale run.

## Important comparison limitation

The historical Mamba-2, Hyena and RetNet rows in the repository were produced under an earlier tokenized 15% masked-language-model protocol. They are retained as recorded numerical references and must not be described as an objective-matched causal benchmark.

## License

GNU Affero General Public License v3.0.
