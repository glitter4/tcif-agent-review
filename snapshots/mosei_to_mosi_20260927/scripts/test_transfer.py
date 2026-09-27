import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import torch
CODE=Path(__file__).resolve().parents[1]/'server-code'
sys.path.insert(0,str(CODE))
import transfer_support as t

with tempfile.TemporaryDirectory() as temp:
    root=Path(temp);parent=torch.nn.Linear(3,2);target=torch.nn.Linear(3,2)
    ck=root/'source.pth';torch.save(parent.state_dict(),ck)
    ck.with_suffix('.json').write_text(json.dumps(dict(checkpoint_epoch=4,dataset='cmumosei',transfer_phase='source')))
    args=SimpleNamespace(transfer_phase='target',dataset='cmumosi',checkpoint_selection_split='test',
        evaluate_test_each_epoch=True,epochs=200,transfer_init_checkpoint=str(ck),save_dir=str(root/'target/checkpoints'))
    t.initialize(target,args)
    assert all(torch.equal(a,b) for a,b in zip(parent.state_dict().values(),target.state_dict().values()))
    try:t.initialize(torch.nn.Linear(4,2),args)
    except AssertionError:pass
    else:raise AssertionError('Mismatched model was accepted')
    # Source updater must neither select a best checkpoint nor inspect test metrics.
    tree=ast.parse((CODE/'train_emotion.py').read_text())
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_update_best_checkpoints')
    ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<checkpoint test>','exec'),ns)
    sourceargs=SimpleNamespace(transfer_phase='source',save_dir=str(root/'source/checkpoints'))
    Path(sourceargs.save_dir).mkdir(parents=True)
    result=ns['_update_best_checkpoints'](None,sourceargs.save_dir,4,{'mae':.7,'acc7':.4},None,{},sourceargs)
    assert result==[] and not list(Path(sourceargs.save_dir).glob('*.pth'))
    assert json.loads((root/'source/source_val_trajectory.jsonl').read_text())['epoch']==4
print('PASS strict keys/shapes and exact transfer; reject incompatible shape; source fixed-budget saver bypasses test/best selection')
