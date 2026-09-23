"""Recreate neighbor metadata from training labels; export aggregate counts only."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path('/path/to/user/workspaces/m4oe-tcif-gradprobe-20260923')
sys.path.insert(0,str(ROOT/'server-code'))
from datasets.emotion_dataset import parse_temporal_sample_id,build_temporal_metadata,build_temporal_context_index
OUT=Path('/path/to/user/m4oe/tcif_mosi_gradprobe_20260923_r2')
LABEL=Path('/path/to/user/datasets/MER-unibench/cmumosi-process/label.npz')
labels=np.load(LABEL,allow_pickle=True)['train_corpus'].item()
ids=sorted(labels)
base={s:parse_temporal_sample_id(s)[0] for s in ids}
groupids={s:i for i,s in enumerate(sorted(set(base.values())))}
metadata={s:build_temporal_metadata(s,labels[s],group_id=groupids[base[s]]) for s in ids}
context=build_temporal_context_index(ids,metadata,radius=1,mode='neighbors',seed=123)
index={s:i for i,s in enumerate(ids)}
probe=json.loads((OUT/'runs/D1_E_FULL/result.json').read_text())
chosen=[r['sample_id'] for r in probe['forward']]
assert len(chosen)==len(set(chosen))==128
counts=dict(samples=128,zero_labels=0,nonzero_positive=0,nonzero_negative=0,
            no_valid_neighbor=0,with_opposite_nonzero_neighbor=0,with_same_nonzero_neighbor=0,
            valid_neighbor_slots=0,valid_opposite_slots=0,valid_same_slots=0)
for sid in chosen:
    y=float(labels[sid]['val']);counts['zero_labels']+=int(y==0)
    counts['nonzero_positive']+=int(y>0);counts['nonzero_negative']+=int(y<0)
    rows=[r for r in context[index[sid]] if r['valid']]
    counts['no_valid_neighbor']+=int(not rows);counts['valid_neighbor_slots']+=len(rows)
    opposite=sum(y*float(labels[ids[r['index']]]['val'])<0 for r in rows)
    same=sum(y*float(labels[ids[r['index']]]['val'])>0 for r in rows)
    counts['with_opposite_nonzero_neighbor']+=int(opposite>0)
    counts['with_same_nonzero_neighbor']+=int(same>0)
    counts['valid_opposite_slots']+=opposite;counts['valid_same_slots']+=same
assert counts['valid_neighbor_slots']==sum(r['valid_neighbor_slots'] for r in probe['batches'])
(OUT/'neighbor_coverage.json').write_text(json.dumps(counts,indent=2))
print(counts)
