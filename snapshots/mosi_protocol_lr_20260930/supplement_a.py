"""Evaluate saved checkpoints after interrupted A; never call the training entrypoint."""
import json,os
import run_protocol as r

run=r.OUT/'runs/A_FUSION_LR'
assert r.read(r.OUT/'A_FUSION_LR_status.json')['exit_code']==1
with (run/'supplement_started.json').open('x') as f:json.dump(dict(mode='evaluation_only',pid=os.getpid(),training_complete=False),f)
cfg=r.read(run/'config.json')
trace=[json.loads(s) for s in (run/'optimizer_updates.jsonl').read_text().splitlines()]
assert trace[-1]['epoch']==168 and trace[-1]['optimizer_step']==6886
for ck in r.e.CKPTS:assert (run/'checkpoints'/(ck+'.pth')).stat().st_size>0
r.e.CODE=r.CODE;r.e.OUT=r.OUT;r.e.PY=r.PY
r.e.evaluate(run,cfg,0)
points,val=r.complete_points(run)
r.dump(run/'result.json',dict(id='A_FUSION_LR',config=cfg,training_complete=False,target_epochs=200,
    completed_epochs=167,interrupted_epoch=168,optimizer_steps=6886,failure='SIGSEGV11 during training',evaluation_complete=True,
    extra_supervised_data='MOSEI fixed4epochs reused',checkpoint_epochs={ck:r.read(run/'checkpoints'/(ck+'.json'))['checkpoint_epoch'] for ck in r.e.CKPTS},points=points,validation_points=val))
r.dump(run/'supplement_complete.json',dict(status='evaluation_complete',training_complete=False,points=14,validation_points=14))
print('A existing weights evaluation complete; training interruption retained',flush=True)
