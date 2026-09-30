"""Read-only metadata audit. Execute on the authorized host; no data or identifiers exported."""
import argparse,json,wave
from pathlib import Path
import numpy as np

def stats(values):
    x=np.asarray(values,dtype=float)
    return dict(n=len(x),min=float(x.min()),median=float(np.median(x)),max=float(x.max())) if len(x) else dict(n=0)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-root',required=True);p.add_argument('--cache-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.dataset_root);cache=Path(a.cache_root)
    labels=np.load(root/'label.npz',allow_pickle=True)
    report=dict(audio_policy='source implementation keeps first6seconds after resampling/denoising; metadata audit cannot certify cached preprocessing provenance',
        visual_policy='4 uniform centers and ratio.2 left/right context; frame presence is not face-detection quality',splits={})
    for split,expected in [('train',1284),('val',229),('test',686)]:
        ids=sorted(labels[split+'_corpus'].item());assert len(ids)==expected
        durations=[];frames=[];missing_audio=missing_faces=invalid_audio=invalid_faces=0;duplicate_centers=0
        for sid in ids:
            f=root/'subaudio'/(str(sid)+'.wav')
            if not f.exists():missing_audio+=1
            else:
                try:
                    with wave.open(str(f),'rb') as w:durations.append(w.getnframes()/w.getframerate())
                except (wave.Error,EOFError):invalid_audio+=1
            f=root/'openface_face'/(str(sid)+'.npy')
            if not f.exists():missing_faces+=1
            else:
                arr=np.load(f,mmap_mode='r',allow_pickle=False)
                if arr.ndim!=4 or arr.shape[0]<1:invalid_faces+=1
                else:
                    n=arr.shape[0];frames.append(n);duplicate_centers+=int(len(set(np.round(np.linspace(0,n-1,4)).astype(int)))<4)
        covered=set();mask_lengths=[];invalid_masks=0;zero_masks=0;shard_shapes=[]
        for f in sorted((cache/split).glob('*/ids.json')):
            if not (f.parent/'done.json').exists():continue
            shard_ids=json.loads(f.read_text());positions=[i for i,s in enumerate(shard_ids) if s in ids];covered.update(shard_ids)
            shapes={}
            for name in ['audio','audio_mask','vision','text_mask']:
                arr=np.load(f.parent/(name+'.npy'),mmap_mode='r');shapes[name]=list(arr.shape)
                assert len(arr)==len(shard_ids)
                if name=='audio_mask':
                    for i in positions:
                        mask=np.asarray(arr[i]);invalid_masks+=int(not np.isin(mask,[0,1]).all());length=int(mask.sum());mask_lengths.append(length);zero_masks+=int(length==0)
            shard_shapes.append(shapes)
        report['splits'][split]=dict(expected=expected,audio_duration_seconds=stats(durations),
            audio_over6seconds=sum(x>6 for x in durations),audio_over6_fraction=sum(x>6 for x in durations)/len(durations) if durations else None,
            audio_missing=missing_audio,audio_unreadable_by_wave=invalid_audio,face_frame_counts=stats(frames),face_missing=missing_faces,face_invalid=invalid_faces,
            repeated_uniform_centers=duplicate_centers,cache_covered=len(set(ids)&covered),cache_missing=len(set(ids)-covered),
            audio_mask_valid_lengths=stats(mask_lengths),invalid_audio_masks=invalid_masks,zero_audio_masks=zero_masks,cache_shard_shapes=shard_shapes)
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(report,indent=2)+'\n')
    print('Input audit complete; unreadable media and mask limitations retained explicitly')
if __name__=='__main__':main()
