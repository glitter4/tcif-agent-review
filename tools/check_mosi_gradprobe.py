"""Validate the 12-condition no-update MOSI gradient diagnostic."""
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/mosi_gradprobe_20260923'
NAMES=[
 'D1_E_FULL','D1_E_DETACH','D1_T123_FULL','D1_T123_DETACH','D1_T124_FULL','D1_T124_DETACH',
 'R1_E58','R1_T58_123','R1_T58_124','R1_E149','R1_T149_123','R1_T149_124']


def main():
    summary=json.loads((DATA/'summary.json').read_text(encoding='utf-8'))
    assert summary['status']=='complete' and set(summary['conditions'])==set(NAMES)
    rows={n:json.loads((DATA/(n+'.json')).read_text(encoding='utf-8')) for n in NAMES}
    base=[x['sample_id'] for x in rows[NAMES[0]]['forward']]
    assert len(base)==len(set(base))==128 and all(s.startswith('train_') for s in base)
    for name,run in rows.items():
        assert run['status']=='complete' and run['weights_unchanged']
        assert len(run['batches'])==4 and len(run['microbatch_metadata'])==8
        assert [x['sample_id'] for x in run['forward']]==base
        sample_order=[s for b in run['microbatch_metadata'] for s in b['sample_ids']]
        assert sample_order==base
        groups=run['grouped_parameter_names']
        flattened=[n for names in groups.values() for n in names]
        assert len(flattened)==len(set(flattened)) and 'router_phi' in groups and 'expert_shared' in groups
        for batch in run['batches']:
            assert batch['sample_count']==32 and batch['global_preclip_norm']>0
            assert math.isclose(batch['clip_coefficient'],min(1,1/(batch['global_preclip_norm']+1e-6)),abs_tol=1e-12)
            assert math.isclose(batch['global_preclip_norm']**2,
                sum(x['norm']['total']**2 for x in batch['groups'].values()),rel_tol=2e-5)
            for group in batch['groups'].values():
                c=group['reg_cls_cosine'];reason=group['cosine_na_reason']
                assert (c is None)==(reason is not None)
                if c is not None:
                    assert -1.000001<=c<=1.000001
                    expected=group['reg_cls_dot']/(group['norm']['reg']*group['norm']['cls_weighted'])
                    assert math.isclose(c,expected,abs_tol=1e-8)
    for suffix in ['E','T123','T124']:
        a=rows['D1_'+suffix+'_FULL'];b=rows['D1_'+suffix+'_DETACH']
        assert [x['sample_ids'] for x in a['microbatch_metadata']]==[x['sample_ids'] for x in b['microbatch_metadata']]
        assert all(x['y_reg']==y['y_reg'] and x['logits']==y['logits'] for x,y in zip(a['forward'],b['forward']))
    cov=summary['sample_coverage']
    assert cov['samples']==128 and cov['valid_neighbor_slots']==sum(b['valid_neighbor_slots'] for b in rows['D1_E_FULL']['batches'])
    print('PASS: 12 no-update probes, 48 effective batches, 128 paired training samples, exact D1 forward equality, group metrics and neighbor coverage.')


if __name__=='__main__':main()
