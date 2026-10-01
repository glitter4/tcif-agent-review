"""Use the executed dataset's parser, publish sequential group numbers, never original IDs."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'server-code'))
from datasets.emotion_dataset import parse_temporal_sample_id
root=Path(sys.argv[1]);split=sys.argv[2]
rows=np.load(root/'label.npz',allow_pickle=True)[split+'_corpus'].item();ids=sorted(rows)
videos=[parse_temporal_sample_id(str(s))[0] for s in ids];mapping={v:i for i,v in enumerate(sorted(set(videos)))}
Path(sys.argv[3]).write_text(json.dumps(dict(split=split,n=len(ids),video_groups=len(mapping),video_group_by_position=[mapping[v] for v in videos]),indent=2)+'\n')
print('VIDEO_GROUPS',split,len(ids),len(mapping))
