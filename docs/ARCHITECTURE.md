# Architecture

The frozen released model is identified by:

`causal_transwave_blocklifting_meyer_byte_lm_g4_v2`

It is a **96,130,040-parameter, exact causal raw-byte language model**. The model does not use BPE, SentencePiece, WordPiece, or self-attention.

## 1. Raw-byte input

Each language-model token is one raw byte in `0..255`. A full sequence is 32,768 bytes.

The sequence is reshaped into 4,096 non-overlapping blocks of 8 bytes. The model owns a shared `256 x 96` byte embedding.

## 2. Learned block-local lifting

Every 8-byte block is processed independently before any temporal sequence mixing. Three lifting stages split even/odd positions and apply learned predict/update corrections:

- `predict(e) = e + Delta_P(e)`
- `detail = odd - predict(e)`
- `coarse = even + 0.5 * detail + Delta_U(detail)`

The three detail streams have lengths 4, 2 and 1; the final coarse stream has length 1. Packing them returns **8 learned scales x 96 channels**.

A content-dependent, group-wise `8 x 8` same-time scale mixer then mixes those scales. The resulting block width is `8 x 96 = 768`.

## 3. Twelve dual-memory TransWave layers

Each layer contains two causal sequence paths.

### Short path

A depthwise causal convolution over flattened 768-dimensional block states uses a kernel of 7 block positions.

### Long path

The long path applies:

`gate -> scale mix -> causal Meyer FIR -> scale mix -> gate`

The Meyer branch has:

- 12 smooth Meyer-style frequency bands;
- 8 Bernstein controls per band;
- group- and scale-specific learned real/imaginary coefficients;
- a 512-block causal FIR;
- zero-padded linear FFT convolution, cropped to the first `L` outputs.

The two paths are added residually with learned amplitudes, followed by RMSNorm and a SwiGLU feed-forward network of width 2816.

## 4. Exact block-level causality

After all 12 layers, completed block states are shifted by one block:

`context[b] = h[b-1]`

The first block receives a learned BOS context. Therefore the decoder for block `b` cannot access the representation of block `b` itself.

`src/transwave_architecture.py` contains an executable causality test that mutates future blocks and verifies that all earlier contexts are unchanged.

## 5. Exact within-block autoregression

The decoder predicts each 8-byte block one byte at a time with a GRUCell.

For byte position `r`, the recurrent state receives the previous true byte during training. Hence byte `r+1` may use bytes `0..r`, but never a future byte.

The objective is therefore the exact causal factorization:

`p(x_t | x_<t)`

and the reported NLL/PPL/BPB are next-byte likelihood metrics, not masked reconstruction metrics.
