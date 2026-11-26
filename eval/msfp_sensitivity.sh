#!/bin/bash
# MSFP leave-one-out sensitivity: 20 test pockets x 4 configs x 50 molecules.
# Resumable: skips pocket/config pairs whose stats JSON already exists.
set -u
cd "$(dirname "$0")/.."   # -> code/

export PATH=/home/longxu/miniconda3/envs/MSCoD/bin:$PATH
CKPT=checkpoints/last.ckpt
N_POCKETS=20
N_SAMPLES=50
N_CPU=8
OUT=eval/7xkj/output/msfp_sensitivity
mkdir -p "$OUT"

pockets=$(ls data/test_set | sort | head -"$N_POCKETS")

for cfg in none r1 r2 r3; do
  for d in $pockets; do
    outdir="$OUT/${d}_${cfg}"
    stats=$(ls "$outdir"/docking_analysis_*_stats.json 2>/dev/null | head -1)
    if [ -n "$stats" ]; then
      echo "SKIP $d $cfg (already done)"
      continue
    fi
    p=$(ls data/test_set/"$d"/*_rec.pdb | head -1)
    l=$(ls data/test_set/"$d"/*_lig_*.sdf | head -1)
    echo "=== [$cfg] $d ==="
    timeout 2400 python sample_for_pocket.py "$p" "$l" \
        --out_dir "$outdir" --num_samples "$N_SAMPLES" --n_cpu "$N_CPU" \
        --ablate "$cfg" --ckpt "$CKPT" > "$OUT/log_${d}_${cfg}.txt" 2>&1
    echo "exit=$? $d $cfg"
    # keep only the JSON summaries, drop the generated SDFs to save space
    find "$outdir" -name '*.sdf' -delete 2>/dev/null
  done
done

echo "=== AGGREGATE ==="
python - <<'EOF'
import json, glob, os, statistics as st
OUT='eval/7xkj/output/msfp_sensitivity'
configs=['none','r1','r2','r3']
metrics=['vina_score','qed','sa','clash']
rows={}
for cfg in configs:
    per={m:[] for m in metrics}
    for d in sorted(os.listdir(OUT)):
        if not d.endswith('_'+cfg): continue
        p=glob.glob(os.path.join(OUT,d,'docking_analysis_*_stats.json'))
        if not p: continue
        s=json.load(open(p[0]))
        for m in metrics:
            v=s.get(m,{})
            if v.get('mean') is not None:
                per[m].append(v['mean'])
    rows[cfg]=per
    print(f"--- {cfg} (pockets={len(per['vina_score'])}) ---")
    for m in metrics:
        vals=per[m]
        if vals:
            se=st.stdev(vals)/(len(vals)**0.5) if len(vals)>1 else float('nan')
            print(f"  {m:12s} mean={st.mean(vals):.3f}  se={se:.3f}  (per-pocket sd={st.stdev(vals):.3f})")

# save CSV
with open(os.path.join(OUT,'summary.csv'),'w') as f:
    f.write('config,n_pockets,'+','.join(metrics)+'\n')
    for cfg in configs:
        per=rows[cfg]
        f.write(f"{cfg},{len(per['vina_score'])},"+','.join(f"{st.mean(per[m]):.4f}" if per[m] else 'NA' for m in metrics)+'\n')
print('saved:', os.path.join(OUT,'summary.csv'))
EOF