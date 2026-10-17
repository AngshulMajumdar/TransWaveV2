# TransWave-Meyer-96M-4.5B release-scale run

The Google Drive release run is complete.

- Release: `TransWave-Meyer-96M-4.5B`
- Architecture: `causal_transwave_blocklifting_meyer_byte_lm_g4_v2`
- Parameters: 96,130,040
- Training exposure: 4,499,996,672 raw-byte tokens
- Full contexts: 137,329 x 32,768 bytes
- Steps: 954
- Normal physical batch: 144
- Fused AdamW: yes
- Training time: 9,222.19 s
- Peak allocated VRAM: 72.55 GiB
- Peak reserved VRAM: 91.61 GiB

Final held-out validation over 1,048,576 bytes:

| Scope | NLL | PPL | BPB |
|---|---:|---:|---:|
| Overall | 2.127545 | 8.394233 | 3.069398 |
| arXiv | 2.029655 | 7.611460 | 2.928173 |
| FineWeb-Edu | 2.189985 | 8.935075 | 3.159480 |
| PG19 | 2.113828 | 8.279874 | 3.049609 |
| WikiText-103 | 2.176712 | 8.817270 | 3.140332 |

The 384.6 MB `model.safetensors` and the approximately 1.15 GB full recovery checkpoint are intentionally not embedded in this source archive. `export_manifest.json` records the exported model metadata.
