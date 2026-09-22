# MOSI 已完成的flat-logit CDF/EMD历史实验

旧2026-07配方，非当前TCIF/L1受控比较；保留失败证据，避免当成全新方向重复。

## 2026-07-16 wave38 cls7 CDF/EMD 0.5 seed123 Lab5090 result

- Controlled change from wave11 seed123 dropout0.5: only `cls7_emd_weight=0.5` was added to soft-CE.
- Eta best Acc7: 0.413994 via `best_hard_cls_acc7_model`, base `reg`, fusion `argmax`, eta 0.675; MAE 0.820418; Acc2/F1 0.801749/0.800221; nonzero Acc2/F1 0.820122/0.817929.
- Eta best binary/nonzero: Acc2/F1 0.811953/0.809917 and nonzero Acc2/F1 0.827744/0.824802 via `best_mae_model`, base `reg`, fusion `argmax`, eta 0.65; Acc7 0.397959.
- Best MAE row: 0.797300 with Acc7 0.387755. Direct test-selected expected Acc7 peaked at 0.406706 (epoch 20); hard-cls Acc7 peaked at 0.412536 (epoch 66).
- Checkpoint policy/cleanup: exactly five test-selected `.pth` files remain and no `epoch_checkpoints` directory exists.
- Conclusion: moderate EMD over soft targets hurts Acc7, binary metrics, and MAE versus wave11. Do not continue soft-CE EMD weights. The next controlled test should pair a lighter EMD term with wave24 hard-CE, where ordinal distance is otherwise absent and standalone Acc7 was strongest.

## 2026-07-16 wave39 hard-CE plus CDF/EMD 0.2 seed123 Lab5090 result

- Controlled change from wave24 hard-CE seed123: only `cls7_emd_weight=0.2` was added.
- Eta best Acc7: 0.418367 via `best_acc7_model`, base `reg`, fusion `argmax`, eta 0.7; MAE 0.816076; Acc2/F1 0.810496/0.808502; nonzero Acc2/F1 0.829268/0.826558.
- Eta best binary: Acc2/F1 0.820700/0.818646 via `best_acc7_model`, base `reg`, fusion `expected`, eta 0.525; Acc7 0.395044; nonzero Acc2/F1 0.838415/0.835589.
- Eta best nonzero binary: Acc2/F1 0.819242/0.816996 and nonzero Acc2/F1 0.839939/0.837073 at eta 0.725; Acc7 0.399417.
- Direct test-selected Acc7 peaked at 0.409621 (epoch 40); direct Acc2 peaked at 0.813411 (epoch 81).
- Checkpoint policy/cleanup: exactly five test-selected `.pth` files remain and no `epoch_checkpoints` directory exists.
- Conclusion: light EMD also degrades the strong wave24 hard-CE Acc7 line. Together with wave38, EMD hurts both soft- and hard-label training; stop this branch. Recent wave37/38/39 runs all remain below the reliable Acc7 and binary bests, so do not submit another loss-weight/seed variant without a materially different model design.
