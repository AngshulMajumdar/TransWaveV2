from pathlib import Path
root = Path(__file__).resolve().parents[1]
bad = []
for p in root.rglob('*'):
    rel = p.relative_to(root)
    if any(part.startswith('.') for part in rel.parts):
        bad.append(str(rel))
if bad:
    print('Hidden/dot-prefixed paths found:')
    for p in bad:
        print(p)
    raise SystemExit(1)
print('PASS: no dot-prefixed files or folders.')
