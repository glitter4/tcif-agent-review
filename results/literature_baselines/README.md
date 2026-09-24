# Published-model baselines on CMU-MOSI and CH-SIMS

These CSVs transcribe the **other papers' rows** from the CMU-MOSI and CH-SIMS comparison tables in the local TCIF manuscript. They exclude TCIF and local-only results. Accuracy and F1 values are percentages; MAE is on the dataset's original scale. Empty cells mean that the selected source did not provide that metric. In particular, TFN's original MOSI paper reports five-class accuracy, which is not entered as ACC-7.

The tables preserve the evaluation conventions of their cited sources. F1 averaging, treatment of zero-valued labels, input alignment, and language features may differ by paper. These numbers should not be treated as a uniformly rerun benchmark or compared directly as if every protocol were identical.

## Files and provenance

- [`cmu_mosi.csv`](cmu_mosi.csv): 20 published-model rows. Numeric sources were checked individually for the local manuscript; [the source audit](#mosi-source-audit) records direct references and exceptions.
- [`ch_sims.csv`](ch_sims.csv): nine published-model rows transcribed from the local manuscript's CH-SIMS comparison table. They are **secondary comparison values**; this export did not independently verify every source paper's CH-SIMS numbers. The cited publications identify the models, while `numeric_source` identifies where these particular numbers were copied from. The comparison table did not supply MAE for these rows.

The local source artifacts were `paper/tcif_revision_20260909/revised/latex/acl_latex.tex` and `paper/tcif_revision_20260909/MOSI_LITERATURE_AUDIT_20260910.md`, inspected for this export. No checkpoint metrics from this repository were merged into these literature rows.

## MOSI source audit

| Model(s) | Numeric source | Link |
|---|---|---|
| EF-LSTM, LF-DNN, Graph-MFN, EMOE | EMOE, Table 1; the first three are later reproductions | [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Fang_EMOE_Modality-Specific_Enhanced_Dynamic_Emotion_Experts_CVPR_2025_paper.pdf) |
| TFN | TFN, Table 1 | [EMNLP 2017 paper](https://aclanthology.org/D17-1115.pdf) |
| LMF | LMF, Table 2 | [ACL 2018 paper](https://aclanthology.org/P18-1209.pdf) |
| MFN | MFN, Table 2 | [AAAI 2018 paper](https://ojs.aaai.org/index.php/AAAI/article/download/12021/11880) |
| MFM | MFM, Figure 3 and Table 2 | [ICLR 2019 manuscript](https://arxiv.org/pdf/1806.06176) |
| MulT | MulT, Table 1 (unaligned) | [ACL 2019 paper](https://aclanthology.org/P19-1656.pdf) |
| MISA | MISA, Table 1 (aligned BERT) | [paper text](https://arxiv.org/html/2005.03545v3) |
| Self-MM | Self-MM, Table 1 | [AAAI 2021 paper](https://ojs.aaai.org/index.php/AAAI/article/view/17289/17096) |
| MMIN | EUAR, Table 2, all modalities present | [ACM MM 2024 paper](https://www.atailab.cn/seminar2024Fall/pdf/2024_ACM%20MM_Enhanced%20Experts%20with%20Uncertainty-Aware%20Routing%20for%20Multimodal%20Sentiment%20Analysis.pdf) |
| DMD | DMD, Table 1 (aligned BERT) | [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Decoupled_Multimodal_Distilling_for_Emotion_Recognition_CVPR_2023_paper.pdf) |
| CubeMLP | CubeMLP, Table 1 | [paper](https://arxiv.org/pdf/2207.14087) |
| BBFN | BBFN, Table 1 | [paper](https://arxiv.org/pdf/2107.13669) |
| C-MIB | C-MIB, Table I | [paper](https://arxiv.org/pdf/2210.17444) |
| MSG | FINE, Table 1 (quoted MSG row); original paper not independently checked | [AAAI 2026 paper](https://ojs.aaai.org/index.php/AAAI/article/download/37176/41138) |
| ConFEDE | ConFEDE, Table 2 | [ACL 2023 paper](https://aclanthology.org/2023.acl-long.421.pdf) |
| EUAR | EUAR, Table 1 | [ACM MM 2024 paper](https://www.atailab.cn/seminar2024Fall/pdf/2024_ACM%20MM_Enhanced%20Experts%20with%20Uncertainty-Aware%20Routing%20for%20Multimodal%20Sentiment%20Analysis.pdf) |
| FINE | FINE, Table 1 | [AAAI 2026 paper](https://ojs.aaai.org/index.php/AAAI/article/download/37176/41138) |

The MOSI source audit selects the nonzero-label binary values where both binary conventions are reported. Some original papers report weighted F1; the CSV keeps their published value and does not relabel it as macro-F1.
