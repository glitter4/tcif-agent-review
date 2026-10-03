# MOSI rebuilt audio6/audio12: completed results

Updated 2026-10-03. Both runs completed 200 epochs / 8200 optimizer updates. Each includes 14 test and 14 validation points: separate test-selected best-Acc7 and best-MAE checkpoints, each eta={0,.2,.4,.6,.8,.9,1}, expected readout/T=1. Test sample counts are 686/all and 656/nonzero; validation 229/all. Seed123, fixed MOSEI epoch4 initialization, extra supervised source data. This is a development/test-selected comparison, not independent confirmation.

These two runs share the dl01 A800 runtime and rebuilt audio implementation; input audio duration is their intended single factor. Historical Lab/c1 candidates differ in environment. Complete source/trajectory/data-order auditing and final study publication remain pending; this is a metrics release, not full-study acceptance. B64/c1 results are not included because no completed result was available for this release.

| Run / criterion | epoch | eta | Acc7 project | Acc7 NumPy | MAE | Acc2 nonzero | macro F1 nonzero |
|---|---:|---:|---:|---:|---:|---:|---:|
| Audio6 max project Acc7 |12|1|46.355685|46.647230|.723921884|83.079268|82.483311|
| Audio12 max project Acc7 |17|.9|46.938776|46.938776|.722488062|83.689024|83.260158|
| Audio6 min MAE |31|.6|44.606414|44.606414|.708923948|84.756098|84.281401|
| Audio12 min MAE |31|.6|44.606414|44.606414|.710713791|84.603659|84.149851|

All metrics on each row are from the same checkpoint and eta. Above Acc7 candidates maximize the project rounding rule; they are not asserted to maximize NumPy Acc7. Full JSON retains both rounding rules, all/nonzero, macro/weighted, and source fields for A*/F*.

Audio12 increases the highest project Acc7 by .583090 percentage points, but its minimum MAE is .001790 worse. Comparing two independently selected maxima does not establish paired improvement at every operating point. The predeclared validation trigger failed: no adjacent compatible eta pair. Therefore this evidence does not trigger ordinary-fusion audio12 expansion. The conditional text L2-SP branch is not implemented or completed in this release.

Evidence: [audio6 full sweep](TCIF_AUDIO6.json), [audio12 full sweep](TCIF_AUDIO12.json), [duration groups and repairs/new errors](audio_group_analysis.json), [validation decision](audio_expansion_decision.json).

## Comparison with reproduced baselines

See [three-seed EMOE/FINE evidence](../mosi_baseline_reproduction_20261003/README.md). NumPy Acc7 must be compared to NumPy Acc7. The new audio12 project-maximum point has NumPy Acc7 46.939%, lower than FINE's test-Acc7 mean 48.445% and best seed 49.125%; its Acc2 83.689% and macro-F1 83.260% are also lower than FINE's corresponding means 83.740% / 83.398%. Its MAE .722488 is lower than that mean .7279. This new audio result is not a joint breakthrough.

The historical TCIF balanced point (NumPy Acc7 46.501%, MAE .706337, nonzero Acc2 84.756%, macro-F1 84.363%) remains a useful trade-off: compared with EMOE test-Acc7 three-seed mean, Acc7 is approximately tied (-.049pp), MAE is lower by .0144, but Acc2/macro-F1 are lower by .305/.321pp. Against FINE mean, it trades -1.944pp Acc7 for lower MAE and +1.016/.965pp Acc2/macro-F1. EMOE's best-MAE seed reaches .7062, so TCIF does not beat every EMOE MAE operating point.

## Best ours versus worst baseline: sensitivity example only

Define worst reproducibly as the lowest NumPy Acc7 among the three seeds, each already selected by test-Acc7. Do not choose the worst epoch or change the definition per metric.

- EMOE seed1113: NumPy Acc7 44.898%, MAE .7460, Acc2 83.994%, macro-F1 83.721%. Our historical balanced point beats this row on all four metrics; approximate differences +1.604pp / -.0397 / +.762pp / +.642pp. This is best-versus-worst selection, not evidence of typical superiority.
- FINE's lowest NumPy Acc7 is a tie: seed3585 and seed7154 both 48.105%. Both must be retained. Their MAE is .7303/.7325, Acc2 83.537/83.384%, macro-F1 83.249/82.997%. Our balanced point has lower MAE and higher polarity metrics but still lower Acc7. Even the historical two-hot project-maximum point has NumPy Acc7 46.793%, below these rows. Do not equate highest project Acc7 with highest NumPy Acc7 across the study.

A best-ours/worst-baseline main table would exaggerate the gap through asymmetric selection. Keep it, if useful, as an explicitly labeled sensitivity example alongside every seed and the mean/std. TCIF has additional MOSEI labels, different encoders/features/context, more recipe/eta searching, and mixed historical environments. FINE is an independent implementation with input/architecture assumptions, not a verified exact official reproduction. These limits remain even though all reported runs use test selection. No method-wide or statistically stable superiority is established by this comparison.
