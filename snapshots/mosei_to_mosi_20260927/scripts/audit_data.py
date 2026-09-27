"""Metadata-only overlap and cache audit; no media or digest computation."""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,sys.argv[1])
from datasets.emotion_dataset import parse_temporal_sample_id
ROOT=Path('/path/to/user/datasets/MER-unibench')
CACHE=Path('/path/to/user/m4oe/embedding_cache')
sets={'mosei':ROOT/'cmumosei-process','mosi':ROOT/'cmumosi-process'}
caches={'mosei':CACHE/'cmumosei_v752_nf4_vctx02_fp16_20260515','mosi':CACHE/'cmumosi_faceonly_nf4_vctx02_fp16_20260518'}
labels={name:np.load(root/'label.npz',allow_pickle=True) for name,root in sets.items()}
data={name:{split:labels[name][split+'_corpus'].item() for split in ['train','val','test']} for name in labels}
videos=lambda rows:{parse_temporal_sample_id(str(s))[0] for s in rows}
protected=videos(data['mosi']['val'])|videos(data['mosi']['test'])
report=dict(counts={name:{s:len(rows) for s,rows in parts.items()} for name,parts in data.items()},
            source_target_overlap={},cache_coverage={},cache_configs={})
for split in ['train','val']:
    intersection=sorted(videos(data['mosei'][split])&protected)
    report['source_target_overlap'][split]=dict(protected_target_video_count=len(intersection),
        protected_target_sample_count=sum(parse_temporal_sample_id(str(s))[0] in protected for s in data['mosei'][split]))
for name in caches:
    report['cache_configs'][name]=json.loads((caches[name]/'cache_config.json').read_text())
    for split in ['train','val'] if name=='mosei' else ['train','val','test']:
        cached=set()
        for f in (caches[name]/split).glob('*/ids.json'):
            if (f.parent/'done.json').exists():cached.update(map(str,json.loads(f.read_text())))
        needed=set(map(str,data[name][split]))
        report['cache_coverage'][name+'_'+split]=dict(required=len(needed),present=len(needed&cached),missing=len(needed-cached))
report['source_internal_train_val_video_overlap']=len(videos(data['mosei']['train'])&videos(data['mosei']['val']))
Path(sys.argv[2]).write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='cache_configs'},indent=2))
print('CACHE_CONFIGS',json.dumps(report['cache_configs'],indent=2))
