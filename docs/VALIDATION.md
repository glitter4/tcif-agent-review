# 打包验证记录

验证日期：2026-09-22。验证对象是整理后的上传目录，而非仅检查原服务器文件。

| 检查 | 结果 |
|---|---|
| Python语法、JSON解析、Markdown文件链接、本地导入依赖 | 通过，22个Python文件 |
| MOSEI消融结果完整性与CSV/JSON逐值一致 | 通过，70点 |
| MOSI D1/N1/N2双checkpoint七点eta | 通过，42点 |
| CH-SIMS四seed×双读出×双checkpoint×七点eta | 通过，112点 |
| 原有消融协议测试 | 3项通过 |
| 打包源码的小模型/消融/掩码/分类头测试 | 26项运行，25项通过，1项跳过 |
| 五种消融的计划命令生成 | 通过，未启动训练 |
| 模型/数据集/评估器/测试与来源的AST比较 | 除机器路径字面值外一致 |
| 训练器与来源比较 | 仅文档列明的路径、输出根和种子环境设置调整 |

本地标准库检查命令：

```bash
python tools/check_bundle.py
python -m unittest discover -s scripts -p 'test_tcif_paper_ablation.py' -v
python scripts/tcif_paper_ablation.py plan
```

模型测试在dl01独立临时副本中使用原执行环境、CPU和单线程运行：

```bash
cd server-code
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest test_tcif_ablation test_structv6_masking test_signed_reg_cls7_head -v
```

跳过项：`test_archived_full_filter_compatibility`，需要设置 `TCIF_REFERENCE_MODULE` 指向未打包的旧滤波器参考文件。本次没有以当前文件代替旧参考来制造兼容性结论。

模型测试覆盖五种变体的前向/反向、初始化、空邻域与padding、参数共享、保存重建、checkpoint元数据、标签映射和文本解冻时禁止cache等。测试使用合成小模型，不等于真实数据完整训练复现；本次没有训练、重新推理历史checkpoint或补跑敏感性eta网格。

结果证据保留原来源的数值和完成状态。对汇总结果的检查证明网格及导出完整，不能代替对未随包提供的逐样本预测、数据及权重的独立审计。
