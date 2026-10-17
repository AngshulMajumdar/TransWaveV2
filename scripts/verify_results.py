from pathlib import Path
import json, math

ROOT = Path(__file__).resolve().parents[1]
files = sorted((ROOT / 'benchmarks' / 'final_causal_results').glob('*.json'))
if len(files) != 4:
    raise SystemExit(f'Expected 4 result JSONs, found {len(files)}')

for p in files:
    d = json.loads(p.read_text())
    nll = float(d['best_validation_nll'])
    ppl = float(d['best_validation_ppl'])
    bpb = float(d['best_validation_bpb'])
    assert abs(math.exp(nll) - ppl) < 1e-5, (p.name, 'PPL mismatch')
    assert abs(nll / math.log(2.0) - bpb) < 1e-5, (p.name, 'BPB mismatch')
    assert d['future_leakage'] is False
    assert d['parameter_count'] == 96130040
    assert d['raw_context_bytes'] == 32768
    print(f'{p.name}: PASS')

release = json.loads((ROOT / 'release' / 'result_4p5b.json').read_text())
assert release['status'] == 'complete'
assert release['training_tokens'] == 4499996672
assert release['parameter_count'] == 96130040
print('4.5B release result: PASS')
