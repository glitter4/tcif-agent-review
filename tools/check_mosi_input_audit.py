"""Check published aggregate consistency, without accessing any original dataset."""
import json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/mosi_input_audit_20261002'
def main():
    r=json.loads((BASE/'summary.json').read_text(encoding='utf-8'))
    assert r['availability']['audio_files']==r['availability']['face_track_files']==2199
    for split,n in [('train',1284),('val',229)]:
        x=r['splits'][split];a=x['audio'];v=x['vision'];c=x['cache'];t=x['text']
        assert x['n']==a['readable']==v['readable_tracks']==n
        assert a['missing']==a['unreadable']==v['missing_tracks']==v['invalid_tracks']==0
        assert 0<=a['over12']<=a['over6']<=n
        for sec in [6,12]:
            assert math.isclose(a[f'discarded_seconds_at{sec}']/a['total_seconds'],a[f'discarded_fraction_at{sec}'],abs_tol=1e-12)
        assert sum(a['groups'][g]['n'] for g in ['weak_nonzero','other_nonzero','zero'])==n
        assert sum(a['groups'][g]['over6'] for g in ['weak_nonzero','other_nonzero','zero'])==a['over6']
        assert c['audio_mask_difference_from_expected_token_count']=={'0':n}
        assert sum(v['unique_indices_in12_context_slots'].values())==n
        assert v['sampled_center_frames']==n*4
        assert sum(v['black_centers_per_affected_clip'])==v['allblack_center_frames']
        assert len(v['black_centers_per_affected_clip'])==v['clips_with_black_centers']
        assert sum(k==4 for k in v['black_centers_per_affected_clip'])==v['clips_all4_centers_black']
        assert t['cached_mask_length_mismatches']==0
    for p in BASE.iterdir():
        text=p.read_text(encoding='utf-8')
        assert not any(s in text for s in ['/data/user/','/home/admin123','C:/Users/'])
    print('PASS: published train/val input audit counts and ratios; no original-data recomputation performed')
if __name__=='__main__':main()
