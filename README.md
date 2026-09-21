# ECG-5001：逐搏变化与时间顺序探索

本仓库归档 2026-09-21 的两阶段纠错结果。唯一最终运行是 [run-final3](ecg-beat-exploration-fixed-20260921/run-final3/ECG-5001-fixed-report-zh.md)。旧探索结果仅供追溯，其有泄漏的性能结论已撤回。

## 目前能证明时间序列有用吗？

不能证明有用，也不能证明无用。区分无序的变化幅度（B−A）和依赖相邻关系的顺序信息（C−B）：

| 300 位患者，训练折内探索模板 | B−A AUROC（95%配对区间） | C−B AUROC（95%配对区间） |
|---|---|---|
| 阶段一：原检测结果 | +0.00627 [−0.00680, +0.01969] | −0.00191 [−0.01107, +0.00542] |
| 阶段二：开发中的检测修正 | +0.00951 [−0.00498, +0.02231] | +0.00076 [−0.00551, +0.00685] |

区间均跨零。阶段二原始 C AUROC 0.86951，5 次患者内打乱为 0.86418–0.86889（均值 0.86628），存在值得继续检验的弱提示，但仅 5 次打乱不足以作有力推断。Bootstrap 基于固定折外预测，不涵盖重新训练的全部不确定性。检测修正参考过当前样本，结果属于开发阶段；没有读取 fold 9/10 波形，没有最终独立测试。严格原型模板有训练折样本不足，不能与全 300 人探索方案混称。

具体下一步：先独立确认 R 峰及切分，冻结检测器和特征，再扩大置换次数，并在预先固定的独立评估上检验顺序特征增量。当前研究假设是“检测可靠时，相邻形态变化可能提供少量超出无序波动统计的信息”，尚未确证，亦不构成创新或临床结论。

## 查找入口

- [中文最终报告](ecg-beat-exploration-fixed-20260921/run-final3/ECG-5001-fixed-report-zh.md)
- [真实波形图册（63 页）](ecg-beat-exploration-fixed-20260921/run-final3/casebook/waveform_casebook.pdf)
- [待人工复核表](ecg-beat-exploration-fixed-20260921/run-final3/casebook/manual_review_form.csv)
- [特征定义](ecg-beat-exploration-fixed-20260921/run-final3/FEATURES.md)
- [患者折外预测](ecg-beat-exploration-fixed-20260921/run-final3/predictions/all_oof.csv)、[配对比较](ecg-beat-exploration-fixed-20260921/run-final3/tables/paired_bootstrap.csv)、[打乱汇总](ecg-beat-exploration-fixed-20260921/run-final3/tables/shuffle_summary.csv)
- [每折模板患者清单](ecg-beat-exploration-fixed-20260921/run-final3/templates/)、[验证结果](ecg-beat-exploration-fixed-20260921/run-final3/verification.json)
- [运行配置](ecg-beat-exploration-fixed-20260921/run-final3/run_config.json)、[依赖快照](ecg-beat-exploration-fixed-20260921/run-final3/dependency_snapshot.json)、[代码哈希](ecg-beat-exploration-fixed-20260921/run-final3/code_hashes.json)

## 复现

在本仓库根目录执行：

```powershell
python -m pytest ecg-beat-exploration-fixed-20260921/code/tests -q -p no:cacheprovider
python ecg-beat-exploration-fixed-20260921/code/fixed_pipeline.py --output run-reproduction
```

需要本机已有 PTB-XL 数据和冻结 final5，默认路径见配置；支持 `--data-root` 与 `--final5`。final5 的 pilot_manifest.csv 内包含原始波形绝对路径，换机器时必须在独立副本中重映射路径并记录变更，不能改写冻结原件。输出须为新目录。

代码按原哈希保存；旧探索目录与新代码的相对位置保持不变。仓库收录旧逐搏事件表和旧 ABC 汇总等执行依赖。原始数据、final5、大型逐搏波形缓存、纠错草稿未上传。逐搏缓存可以重建，原件保留在本地输出目录。[归档清单](archive_manifest.json)记录已收录与省略文件的原路径、大小、SHA256。旧 NPZ 未随 Git 收录，因此重新运行的“历史输入文件清单”会与当时全量本地目录不同；这是归档范围差异，不是声称逐字节重现历史审计环境。旧报告及旧代码只作溯源，不应当作当前有效分析。

本仓库含公开 PTB-XL 来源的派生波形图与研究产物，不含原始数据全集。使用数据时遵循其许可和引用要求；本归档不改变上游许可。
