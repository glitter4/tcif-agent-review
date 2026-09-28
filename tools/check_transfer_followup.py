"""Audit all authorized transfer follow-up targets, budgets and comparisons."""
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/transfer_followup_20260928'
NAMES=['E1_ROUTER','E2_S','E3_RESET','C_T','B_T','B_ST','COMPUTE_T']
CKPTS=['best_acc7_model','best_mae_model'];ETAS=[0,.2,.4,.6,.8,.9,1]
def read(p):return json.loads((DATA/p).read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in (DATA/p).read_text().splitlines()]

def main():
    complete=read('complete.json');assert complete['status']=='complete'
    assert {x['name'] for x in complete['results']}==set(NAMES) and all(x['exit_code']==0 for x in complete['results'])
    manifest=read('manifest.json');runs={}
    for name in NAMES:
        r=read(f'runs/{name}/result.json');runs[name]=r
        assert r['training_complete'] and r['target_epochs']==200
        assert r['config']==manifest['target_configs'][name]
        assert read(f'{name}_status.json')['exit_code']==0
        trace=lines(f'runs/{name}/compute_trace.jsonl')
        assert [t['epoch'] for t in trace]==list(range(1,201))
        assert all(t['optimizer_steps']==41 and t['examples']==1284 and t['microbatches']==81 for t in trace)
        assert trace[-1]['total_optimizer_steps']==8200
        assert sum(t['examples'] for t in trace)==256800
        assert len(r['points'])==14
        for ck in CKPTS:
            assert sorted(p['eta'] for p in r['points'] if p['checkpoint']==ck)==ETAS
            m=read(f'runs/{name}/checkpoints/{ck}.json');assert m['checkpoint_epoch']==r['checkpoint_epochs'][ck]
            assert m['checkpoint_selection_split']=='test'
        for p in r['points']:
            m=p['metrics'];assert p['T']==1 and p['readout']=='expected'
            assert m['num_samples_all']==686 and m['num_samples_non0']==656
            assert all(math.isfinite(m[k]) for k in ['Acc7','MAE','Acc2','Acc2non0','F1_macro_all','F1_macro_non0'])
            assert p['A_star']==max(m['Acc2'],m['Acc2non0'])==m[p['A_source']]
            assert p['F_star']==max(m['F1_macro_all'],m['F1_macro_non0'])==m[p['F_source']]
        expected=any(p['eta']>0 and p['metrics']['Acc7']>=46.1 and p['metrics']['MAE']<=.730 and p['A_star']>=85 and p['F_star']>=85 for p in r['points'])
        assert r['stage_pass']==expected
        final=any(p['eta']>0 and p['metrics']['Acc7']>48.5 and p['metrics']['MAE']<.697 and p['A_star']>86.95 and p['F_star']>86.94 for p in r['points'])
        assert r['final_pass']==final
        if name=='E2_S':assert r['config']['transfer_use_s'] and any(t['mean_S_weighted_loss']>0 for t in trace)
        init=read(f'runs/{name}/initialization_audit.json')
        assert init['optimizer_restored'] is False
        if name not in ['B_T','C_T']:
            assert init['status']=='passed' and init['strict_shapes'] and init['inherited_tensors_exact']
            drift=read(f'runs/{name}/router_drift.json');assert set(drift)==set(CKPTS)
            for layer in [v for per_ck in drift.values() for v in per_ck.values()]:
                assert math.isclose(layer['relative_delta'],layer['delta_norm']/(layer['source_norm']+1e-12),rel_tol=1e-10)
        if name=='E3_RESET':
            assert init['output_projection_zero'] and init['reset_matches_fresh_seeded_initialization']
            assert len(init['reset_tensor_names'])==52 and init['inherited_tensor_count']==743
            assert all(n.startswith(('tcif_regression.','tcif_ordinal.','tcif_context_reg_head.','tcif_context_cls7_head.')) for n in init['reset_tensor_names'])
        if name in ['B_T','B_ST']:
            assert r['config']['context_fusion']=='standard' and not r['config']['tcif_enable_transition_gate']
            assert r['config']['tcif_transition_gate_loss_weight']==r['config']['tcif_context_aux_weight']==0
            assert abs(init['context_variant']['relative_active_difference'])<.005
        if name in ['E1_ROUTER','E2_S','E3_RESET']:
            pairs=read(f'runs/{name}/paired_repairs.json');assert len(pairs)==28
            for p in pairs:
                assert sum(v['n'] for v in p['groups'].values())==(686 if p['split']=='test' else 229)
                for g in p['groups'].values():
                    assert g['sign_fixed']+g['new_sign_errors']<=g['n']
                    assert g['sign_fixed_and_acc7_correct']<=g['sign_fixed']
    details=read('runs/E2_S/paired_repairs_detailed.json');basic=read('runs/E2_S/paired_repairs.json')
    for d in details:
        b=next(x for x in basic if all(x[k]==d[k] for k in ['checkpoint','eta','split']))
        for group,v in d['groups'].items():
            assert v['sign_fixes']==b['groups'][group]['sign_fixed']==sum(v['fixed_sign_class_transition'].values())
            assert v['new_sign_errors']==b['groups'][group]['new_sign_errors']
    sources={}
    for name in ['B_SOURCE','COMPUTE_SOURCE']:
        c=read(f'source_runs/{name}/complete.json');trace=lines(f'source_runs/{name}/compute_trace.jsonl')
        assert c['trace']==trace and c['source_epochs']==4 and c['optimizer_steps']==2044 and c['examples']==65304
        assert [t['epoch'] for t in trace]==[1,2,3,4]
        assert all(t['optimizer_steps']==511 and t['examples']==16326 and t['microbatches']==1021 for t in trace)
        m=read(f'source_runs/{name}/checkpoints/recovery/latest_model.json');assert m['checkpoint_epoch']==4
        assert m['dataset']==('cmumosi' if name=='COMPUTE_SOURCE' else 'cmumosei')
        sources[name]={k:sum(t['input_elements'][k] for t in trace) for k in trace[0]['input_elements']}
    assert sources['B_SOURCE']==sources['COMPUTE_SOURCE']
    runs['C_ST']=read('C_ST_reused.json')
    factorial=read('factorial.json');assert len(factorial['rows'])==14 and factorial['main_eta']==.8
    for row in factorial['rows']:
        for name,p in row['cells'].items():
            assert p==next(x for x in runs[name]['points'] if x['checkpoint']==row['checkpoint_type'] and x['eta']==row['eta'])
        for key,actual in row['interaction_Q'].items():
            def q(name):
                p=row['cells'][name];v=p[key] if key.endswith('_star') else p['metrics'][key]
                return -v if key=='MAE' else v
            assert abs(actual-((q('C_ST')-q('B_ST'))-(q('C_T')-q('B_T'))))<1e-10
    comparison=read('validation_comparison.json');assert comparison['eta']==.8 and comparison['selection'] is False
    records=[comparison['rows']['source_initial']]+list(comparison['rows']['C_ST'].values())+list(comparison['rows']['E1_ROUTER'].values())
    for row in records:assert row['source_val']['num_samples']==1871 and row['target_val']['num_samples']==229
    print('PASS seven200-epoch targets/98 new eta points, two2044-step sources, exact exposure/input budgets, TCIF reset, weak-error transitions, router drift, fixed-val diagnostics and2x2 interaction.')

if __name__=='__main__':main()
