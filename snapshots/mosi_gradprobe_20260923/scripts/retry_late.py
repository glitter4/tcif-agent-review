"""Only the three missing R1 epoch149 cases after concurrent-GPU OOM."""
from pathlib import Path
import json
import probe_runner as r

r.OUT=Path('/path/to/user/m4oe/tcif_mosi_gradprobe_20260923_r3')
r.LANES=[['R1_E149','R1_T149_123','R1_T149_124']]
if __name__=='__main__':
    r.preflight()
    results=r.lane(0)
    r.runner.dump(r.OUT/'complete.json',dict(status='complete' if len(results)==3 and all(x['exit_code']==0 and x['result_exists'] for x in results) else 'attention',
        results=results,recovery_of='/path/to/user/m4oe/tcif_mosi_gradprobe_20260923_r2/R1_E149'))
