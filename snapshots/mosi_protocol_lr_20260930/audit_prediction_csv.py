"""Audit a fixed prediction CSV without loading weights. Both mappings use identical inputs."""
import argparse,csv,json
from pathlib import Path
from metric_protocol import audit

def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--output',required=True)
    p.add_argument('--dataset',choices=['mosi','mosei'],required=True)
    args=p.parse_args();rows=list(csv.DictReader(Path(args.csv).open(encoding='utf-8')))
    result=audit([float(x['true_value']) for x in rows],[float(x['pred_value']) for x in rows])
    result.update(dataset=args.dataset,selection='unchanged existing checkpoint/eta',input='stored unrounded final predictions, float32 evaluator dtype',sample_identifiers='dataset row positions only')
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print({k:result[k] for k in ['n','Acc7_legacy','Acc7_nearest_even','wrong_to_right','right_to_wrong']})
if __name__=='__main__':main()
