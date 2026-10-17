from pathlib import Path
import csv
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'benchmarks' / 'comparison_tables.csv'
OUT = ROOT / 'figures' / 'benchmarks'
OUT.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(CSV.open(encoding='utf-8')))
datasets = []
for r in rows:
    if r['dataset'] not in datasets:
        datasets.append(r['dataset'])

def slug(s):
    return s.lower().replace('-', '_').replace(' ', '_')

for ds in datasets:
    part = [r for r in rows if r['dataset'] == ds]
    base = [r for r in part if r['method'] != 'TransWave']
    tw = next(r for r in part if r['method'] == 'TransWave')
    best_nll = min(float(r['nll']) for r in base)
    best_ppl = min(float(r['ppl']) for r in base)
    nll_imp = 100 * (best_nll - float(tw['nll'])) / best_nll
    ppl_imp = 100 * (best_ppl - float(tw['ppl'])) / best_ppl

    fig = plt.figure(figsize=(12, 6.75))
    ax = fig.add_axes([0.055, 0.11, 0.89, 0.80])
    ax.axis('off')
    fig.suptitle(f'{ds} — Recorded Benchmark Table', fontsize=24, fontweight='bold', y=0.955)
    fig.text(0.5, 0.895,
             f'TransWave: {nll_imp:.1f}% lower NLL and {ppl_imp:.1f}% lower PPL than the numerically best historical row',
             ha='center', fontsize=15, fontweight='bold')
    table = ax.table(
        cellText=[[r['method'], f"{float(r['nll']):.6f}", f"{float(r['ppl']):.6f}"] for r in part],
        colLabels=['Method', 'NLL ↓', 'PPL ↓'], cellLoc='center', colLoc='center',
        bbox=[0.08, 0.28, 0.84, 0.52], colWidths=[0.42, 0.26, 0.32])
    table.auto_set_font_size(False); table.set_fontsize(16)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('black'); cell.set_linewidth(1.0)
        if row == 0:
            cell.get_text().set_fontweight('bold'); cell.set_linewidth(1.8)
        if row == 4:
            cell.get_text().set_fontweight('bold'); cell.set_linewidth(2.8); cell.get_text().set_fontsize(18)
    fig.text(0.5, 0.19,
             'Protocol note: historical Mamba-2 / Hyena / RetNet rows used an earlier tokenized 15% MLM protocol;',
             ha='center', fontsize=10.5)
    fig.text(0.5, 0.155,
             'TransWave is exact causal raw-byte next-byte modeling. This is a recorded numerical reference, not a controlled causal head-to-head.',
             ha='center', fontsize=10.5)
    fig.savefig(OUT / f'{slug(ds)}_benchmark_table_regenerated.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
