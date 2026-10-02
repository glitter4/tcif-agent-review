# EMOE / FINE reproduction source

EMOE source was downloaded from the [official main archive](https://github.com/fuyyyyy/EMOE) on 2026-10-02. The official Python model, router, attention and utilities are retained under `vendor/EMOE-main`; authors: Yiyang Fang, Wenke Huang, Guancheng Wan, Kehua Su, Mang Ye (CVPR 2025). No upstream license grant is inferred.

`run.py` uses the official EMOE model and transcribes its training objective/update schedule with separately selected checkpoints and per-epoch metrics. `verify.py` checks loss equivalence and FINE gradient/inference isolation. `fine.py` is our independent reconstruction, not author code. The official EMOE trainer itself is retained for review.

[Results, assumptions and input mismatch](../../results/mosi_baseline_reproduction_20261003/README.md). Tested package versions are recorded in the result directory. Required packages include PyTorch, transformers, NumPy, scikit-learn, nvidia-ml-py, easydict and einops. This is a review snapshot: provide a standard MOSI pickle, local BERT and a fresh output directory; no data or weights are included.

```bash
python run.py --model emoe --seed 1111 --data /path/to/unaligned_50.pkl --bert /path/to/bert-base-uncased --out /path/to/new-run
python run.py --model fine --seed 3585 --data /path/to/unaligned_50.pkl --bert /path/to/bert-base-uncased --out /path/to/another-new-run
```

Scheduler/SSH scripts and machine-specific paths are excluded. The programs can produce per-sample local predictions when run by their owner, but those files and true sample IDs are not published here.
