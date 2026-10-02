# CMU-MOSI EMOE / FINE reproduction attempt — 2026-10-02

Target host: **c1 (SSH endpoint configured locally)**, Slurm partition 4090.
Remote source: `/path/to/user/workspaces/mosi-emoe-fine-20261002`.
Independent workspace; existing TCIF code, checkpoints and jobs are untouched.

## Sources and targets

- EMOE official source downloaded from https://github.com/fuyyyyy/EMOE (main archive, 2026-10-02); original source retained under vendor/EMOE-main.
- EMOE CVPR 2025 Table 1, unaligned BERT: Acc7 47.8%, MAE .697, Acc2 85.4%, weighted F1 85.3%.
- FINE AAAI 2026: https://arxiv.org/html/2511.20167v1 and https://ojs.aaai.org/index.php/AAAI/article/download/37176/41138 . Target Acc7 48.50%, Acc2 86.95%, F1 86.94%. No MAE target is published in this table.
- No verified official FINE repository found after checking exact title, paper, author/lab publication page and GitHub search on 2026-10-02. `fine.py` is an **independent reconstruction**, not author code.
- Standard MMSA MOSI unaligned features: https://huggingface.co/datasets/tamb2203579/CMU-MOSI/resolve/main/Processed/unaligned_50.pkl . This is a third-party mirror; structural, split, finite-value and sample-identity audits are performed. Byte digests are not used.
- BERT: official bert-base-uncased model, downloaded through hf-mirror.com to c1. No TCIF backbone/cache substitution.

## Selection and reporting

Train/validation/test expected counts: 1284/229/686. Check ID disjointness before training. Report test MAE over all samples, nonzero Acc2, weighted and macro F1, and both NumPy and project half-away-from-zero Acc7. Also report all-sample zero-positive Acc2. Save test predictions for every epoch in original dataset order.

Following project instructions, save separate **test-best NumPy Acc7**, **test-best project Acc7**, and **test-best MAE** checkpoints. These are test-selected results, not unbiased held-out estimates. For EMOE also record the test results at the official validation-MAE-selected epoch as a secondary reference; this does not replace the project target checkpoints.

Neither baseline has a classification/regression interpolation parameter eta: both return a single scalar. Eta sweeps are **not applicable** to these original architectures. No artificial classification branch, interpolation or test-time fusion is introduced. If later comparing TCIF, retain its separate full eta sweeps for each target checkpoint.

Three independent seeds per method: EMOE 1111,1112,1113; FINE paper seeds 3585,7154,8757. Report all seeds and mean/std, not just the best seed. A single poor attempt does not establish that a paper's numbers are invalid.

## EMOE fidelity

Use the unchanged official model, BERT encoder, router and attention implementation. The separate driver transcribes the official loss and Adam training loop. Keep batch16, LR1e-4, update every 10 microbatches, unscaled accumulated loss, discarded incomplete final accumulation, official dropout and no weight decay/gradient clipping (the official loop does not apply the configuration's advertised values). The vectorized inverse-error routing target is algebraically the same as the official Python loop. Retain shuffled validation/test loaders and unweighted mean of batch MAE for its scheduler and early stopping.

Changes: explicitly select unaligned data, use local BERT path, num_workers=0, stream detached losses/predictions, replace unconditional per-epoch giant checkpoint saving with selected checkpoint saving, add per-epoch audits and a safety cap of 100 epochs. The original validation plateau schedule and patience10 early stopping remain. The 100-epoch cap is reported if reached.

## FINE assumptions and limitations

Implemented components: three modality-specific top-k mixtures of query transformers; shared/unique encoders and reconstruction; shared/unique task-relevant encoders; label encoders used **only in training losses**; four MI objectives; per-class FIFO contrastive queues and angle compensation; unimodal Transformer decoders and multimodal Transformer fusion.

MOSI Table7 settings: batch32, main LR6e-5/BERT LR4e-5, 100 epochs, four experts, four queries, top-k3, MoQ T/A/V widths256/128/256, initial factorization reduction .5, queue alpha .7, CL weight1, UP .4, router .2, MI .5. Prefer Table7's four experts over the conflicting statement of eight for every dataset.

Unspecified choices are explicit: two Transformer decoder layers per Q-Former, eight heads, dropout .1, FFN width4x; task-relevant shared/unique widths128; one layer per unimodal decoder; two fusion layers; MLP GELU activations; Transformer positions retain modality/block order but no additional positional embedding; CLS fusion concatenates three CLS outputs. AdamW decay .01, 10% step warmup followed by linear decay interprets the ambiguous `warmup epochs=0.1`; no early stopping. Queue minimum8, temperature .1, NumPy-style nearest-integer sentiment bins. Gate weights are top-k original softmax values without renormalization. Eq25's signed label difference is implemented literally.

MI critics are trainable pairwise MLPs. Lower bounds use InfoNCE; upper bounds use positive-minus-marginal critic scores. Critic parameters are fitted by detached-feature InfoNCE and held fixed in encoder objectives. Conditioning follows the paper's concatenated feature/label construction; marginal negatives are the minibatch rather than unspecified conditional resampling. These choices cannot be verified against unavailable author code and may affect results substantially.

**Feature mismatch:** the available standard MMSA MOSI input has audio5/vision20, matching EMOE's official configuration. FINE describes COVAREP74/FACET35. Running FINE on this common input is explicitly an implementation attempt on MMSA features, **not an exact input replication**. Do not attribute any gap solely to the model or use this run to disprove 48.50%. Feature dimensions are saved in each protocol.json.

Queue and label encoders are excluded from prediction: evaluation forward accepts no labels and neither reads nor updates queues. Queue capacities use training labels only. Nonfinite inputs/losses cause failure rather than silent replacement (except official EMOE's negative-infinity audio cleanup).

## Execution incidents and checks

`verify.py` passed on c1: exact EMOE objective parity against the official utility functions; FINE finite forward/backward including all-padding modality rows; encoder and critic gradient separation; predictions invariant to changing label-encoder parameters and clearing the training queue.

Initial array353586: EMOE tasks0–2 failed before a training batch because Intel MKL could not resolve an OpenMP symbol. `MKL_THREADING_LAYER=GNU` fixes the runtime selection. FINE tasks3/4 trained normally; task5 ran out of memory on the shared default GPU.

Array353592 retried EMOE but also encountered shared-GPU memory contention. c1's device namespace exposes shared GPU0 plus the job's allocated GPU while exporting CUDA_VISIBLE_DEVICES=0. Different allocations expose different second GPU UUIDs; a UUID seen via a separate SSH session is not portable between allocations. Attempts353596 and353603 exited without training because they used such an unavailable UUID. Canonical `train.sbatch` now resolves an idle GPU UUID inside its own allocation, requires at least20000 MiB free and acquires a per-device advisory lock. Array353606 reruns tasks0/1/2/5; scheduler quota controls parallelism. The two working original FINE tasks are retained. Failed attempts are retained for audit and excluded from results.

Dependencies `nvidia-ml-py`, `easydict`, `einops` are isolated in `deps-local`; the shared Python environment was not modified. A remote package-index connection failed; downloaded pure-Python wheels were transferred and installed offline instead.

## Final status

All six intended seeds completed successfully on c1. EMOE stopped at epochs36/39/30 for seeds1111/1112/1113; FINE completed100 epochs for each paper seed. GPU verification job353619 reloaded all18 target checkpoints (three per seed); all class metrics matched and maximum absolute prediction difference was0.00001872, attributable to changed evaluation minibatch order for EMOE. FINE predictions matched exactly. Small artifacts were copied locally; data, BERT weights and trained checkpoints remain on c1. See RESULTS.md for complete numbers and limitations.
