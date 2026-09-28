from pathlib import Path
import run_followup as r
root=r.OUT/'reference_drift'
if not (root/'router_drift.json').exists():
 root.mkdir(exist_ok=True)
 link=root/'checkpoints'
 if not link.exists():link.symlink_to(r.REFERENCE/'checkpoints',target_is_directory=True)
 r.drift(root,r.SOURCE)
print('REFERENCE_DRIFT_READY')
