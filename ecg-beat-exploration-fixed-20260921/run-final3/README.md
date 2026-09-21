# ECG-5001 corrected experiment

权威交付运行目录：`run-final3/`。`run-verified/`、`run-final/`、`run-final2/` 和本目录旧的空/中间产物是纠错期间预览，不用于报告结论。原始 final5 与上一轮 ecg-beat-exploration 均未覆盖。

从任意工作目录执行（PowerShell）：

```powershell
python "D:/UserFiles/desktop/dd/2026-09-20/ecg-5001-pilot-garch-egarch-ecg/outputs/ecg-beat-exploration-fixed-20260921/code/fixed_pipeline.py" --output "D:/UserFiles/desktop/dd/2026-09-20/ecg-5001-pilot-garch-egarch-ecg/outputs/ecg-beat-exploration-fixed-20260921/run-reproduction"
```

必须使用新的空目录；脚本拒绝覆盖已有输出。路径默认值指向本地既有数据和final5；可用 `--data-root` 与 `--final5` 显式替换。首次运行只读300条fold1–8波形，验证原始输入和所有旧输出前后SHA256完全一致。

```powershell
python -m pytest "D:/UserFiles/desktop/dd/2026-09-20/ecg-5001-pilot-garch-egarch-ecg/outputs/ecg-beat-exploration-fixed-20260921/code/tests" -q -p no:cacheprovider
```

依赖见run-final3/dependency_snapshot.json。无需下载数据、网络或复杂模型。代码副本baseline/保存实际调用的旧读入/逐搏/BSW-like模块；analysis_core.py实现等价批量距离和配对特征，fixed_pipeline.py为单一入口，deliverables.py从真实输出生成全部报告/图/检查。

重点入口：run-final3/ECG-5001-fixed-report-zh.md；run-final3/casebook/waveform_casebook.pdf；run-final3/predictions/paired_oof_wide.csv；run-final3/tables/paired_bootstrap.csv；run-final3/templates/*_contributors.csv；run-final3/verification.json。

完整标签编码NORM=0、LVH=1。score缺失严格标为insufficient_train_prototypes，不能当完整CV。所有AI审核字段均非人工专家确认。
