# 协议、依赖与运行

## 先区分阅读、测试与复现

`python tools/check_bundle.py` 和 `python -m unittest discover -s scripts -p 'test_tcif_paper_ablation.py' -v` 仅需 Python 标准库。模型测试需要 PyTorch 等依赖，使用 tiny backbones / 合成输入，不需下载真实模型或数据。

`requirements.txt` 来自 2026-09-22 对源执行环境的直接依赖查询，不是历史训练时的完整 lockfile。源环境为 Python 3.10 系列；建议自行准备兼容 Python 3.10 的环境。Torch/torchvision 请按所用 CPU/CUDA 安装匹配版本，再安装剩余依赖。`av` 用于视频路径；独立音频文件路径可能需要与 torch 匹配的 `torchaudio`。本文不承诺任意新环境与旧 CUDA 运行逐位相同。

已有模型测试入口：

```bash
python -m pip install -r requirements-dev.txt
cd server-code
python -m pytest test_tcif.py test_tcif_ablation.py test_structv6_masking.py test_optimizer_param_groups.py test_signed_reg_cls7_head.py test_sign_structured_heads.py test_temporal_kernels_sampler.py test_expert_load_output.py -q
```

## 外部输入约定

`dataset_root` 包含 `label.npz`（train_corpus/val_corpus/test_corpus 字典）、`transcription-engchi-polish.csv`（id及文本）、`openface_face`，以及非缓存路径所需的原始媒体。具体 ID、label 字段和媒体回退路径以 `CMUMOSEIProcessDataset` 为准。本包没有数据与标签文件。

MOSEI 已记录样本数：train 16326、val 1871、test 4659；test nonzero 3634。

`embedding_cache_root/<split>/<shard>/` 包含 `ids.json`、`done.json` 与 `vision.npy`、`text.npy`、`audio.npy`、`text_mask.npy`、`audio_mask.npy`。reader 按 mmap 加载、检查 split 覆盖。即使训练抛弃 text cache，该快照 reader 仍会读取这些字段；不能自行删去 shard 中的文本文件。历史缓存生成器不在本快照，使用匹配缓存或按 dataset 的原始输入路径另行准备。

需要本地 RoBERTa-base、ViT-base-patch16-224-in21k、HuBERT-base-ls960；`vision_backbone_path` 是旧 ResNet 参数，默认 ViT 分支不使用。路径以配置为准，代码中历史 BERT 默认值不代表主实验使用 BERT。

## 配置与运行

先将 `configs/tcif_paper_single_model.json` 中所有 `/path/to/...` 替换为自己的数据、cache 和模型位置。输出路径保持独立新目录。此配置对应消融对照：seed40、batch64×accum4、最多10轮、gate tau=.75。不要以此配置声称复现后续 tau=.5 主结果。

打印五组完整命令（不会训练）：

```bash
python scripts/tcif_paper_ablation.py plan
```

在分配到的 GPU 环境执行完整模型训练、双 checkpoint 七点评估：

```bash
python scripts/tcif_paper_ablation.py train --variant full --output-root outputs/ablation
python scripts/tcif_paper_ablation.py evaluate --variant full --output-root outputs/ablation
```

其他四个 variant：`standard_context`、`equal_weight`、`no_gate`、`shared_filter`。五组均训练和评估后执行：

```bash
python scripts/tcif_paper_ablation.py summarize --output-root outputs/ablation
```

后续 tau=.5 参考配置见 `configs/mosei_gate_tau05_reference.json`，它根据完整消融配置及历史敏感性 runner 的 overrides 重建；并非下载的原始 checkpoint sidecar。batch32×accum8、workers0、tau=.5；可通过 `python server-code/train_emotion.py --training_config configs/mosei_gate_tau05_reference.json` 使用。正式复现实验仍须保存原始 sidecar、环境和全部 sweep；历史 56.643 行只有固定 eta 结果证据。

## 指标与选模不变量

- 按 test 指标分别选 `best_acc7_model` 与 `best_mae_model`。训练时 eta 影响 epoch 选择；early stopping 配置另外查看，不能与 checkpoint 选择混为一谈。
- 每个目标 checkpoint 分别完整评估 eta={0,.2,.4,.6,.8,.9,1}；即使来自同 epoch，也保留两个记录。
- Acc7：全部样本，以 half-away-from-zero 四舍五入并截断到[-3,3]。MAE：全部样本、原量纲。
- Acc2non0/macro-F1non0：只按真实标签去掉 exact zero；预测 >=0 为正。F1 是两类算术平均，不是 weighted F1。
- 一行四指标必须来自同 checkpoint/eta/readout/温度；百分数与 0…1 比率不要混用。
- 训练日志中的分类 head accuracy、all-sample Acc2 与最终融合后的主表指标是不同对象。主表按 `score_details` 重算。
- 当前结果属于反复观察 test 后的开发比较；旧 seed/配置/环境下的结果不可替代新的同批对照。

历史敏感性实验明确只评固定 eta=.8；结果保留这个例外标注。本次整理没有补跑缺失实验。
