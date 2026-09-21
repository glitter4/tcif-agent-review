# Review instructions

- Read docs/REVIEW_BRIEF.md and results/EXPERIMENTS.md before suggesting improvements.
- Default target: CMU-MOSEI. MOSI and CH-SIMS results are separate historical evidence.
- Main objective: improve predictive accuracy/MAE jointly; discuss speed and memory separately.
- Distinguish observed evidence, suspected bugs, and untested hypotheses. Cite file paths and symbols.
- Follow the existing test-selected checkpoint protocol when reproducing this project. Explain its interpretation; do not silently change selection to validation.
- Evaluate every selected target checkpoint separately at eta = 0, .2, .4, .6, .8, .9, 1. Report all metrics from the same checkpoint/eta/readout.
- Do not confuse historical fixed-eta exceptions with complete sweeps.
- Unfrozen text encoders require live text features, including temporal neighbors.
- Do not compute or introduce SHA256 or other hashes. Use direct byte/tensor comparisons if needed.
- Do not launch training, retrieve datasets, or publish changes merely because you are reviewing this snapshot.
