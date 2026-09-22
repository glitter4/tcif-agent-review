# CH-SIMS gradient-path and classifier study

Detach: B0 seed40/41 original recipe, 50 epochs. Only neighbor task features detached before TCIF; forward and trainable parameter count unchanged. Reuse completed B0 as reference.

Stage2: fixed B0 test-best-MAE checkpoint for each seed. Five epochs each head-only and full-continue, fresh AdamW, constant LR, same initial seed and number of optimizer steps; later epoch permutations may differ with random-number consumption. Classifier LR=3*2.32e-5 for BOTH arms; other groups keep original LR in full-continue. Head-only freezes all other parameters, evaluates frozen modules (dropout off), preserves hard CE .3. This contrasts two declared optimization regimes; restarting full-continue optimizer is not exact resume. Do not mix with detach. Preserve starting checkpoint as epoch0 reference. Test-selected best Acc5/MAE plus guarded checkpoints each independently receive expected/argmax complete eta sweeps.

No new architecture or loss. Classification head is existing head_cls7 linear, distinct from context classification head. Frozen-context head remains frozen. Forward-time regression must stay fixed in head-only arm. New checkpoints report stage epoch, separate from source checkpoint epoch.
