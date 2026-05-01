# HCM 心肌环轨迹首版执行说明

## 1) 安装依赖

```powershell
cd "e:\本基中间结果"
python -m pip install -r requirements.txt
```

## 2) 运行（先用 20-50 例，快速版）

```powershell
python "e:\本基中间结果\run_hcm_pipeline.py" --max-patients 50
```

## 3) 运行（配准增强版，较慢）

```powershell
python "e:\本基中间结果\run_hcm_pipeline.py" --max-patients 50 --use-registration --max-reg-frames 8
```

## 4) 全量运行

```powershell
python "e:\本基中间结果\run_hcm_pipeline.py" --max-patients 2000
```

## 5) 深化实验（自动选择CPU/GPU最佳后端）

```powershell
python "e:\本基中间结果\deep_experiment_full_gpu.py" --compute auto --cpu-jobs 0
```

- `--compute auto`：自动在 CPU/GPU 中选择更合适的运行方式（可改为 `cpu` 或 `gpu` 强制指定）
- `--cpu-jobs 0`：自动线程数（CPU核数-1）；可手动指定如 `--cpu-jobs 8`

## 6) 60例深化实验（自动选择CPU/GPU最佳后端）

```powershell
python "e:\本基中间结果\deep_experiment_60.py" --compute auto --cpu-jobs 0
```

## 7) 输出文件

- `outputs/patient_manifest.csv`：raw/seg 病人级匹配清单
- `outputs/trajectory_features.csv`：轨迹几何特征（Ldiast/Rdiast/kappa/Eclose/DeltaTheta）
- `outputs/features_with_labels.csv`：与生存标签合并后的建模数据
- `outputs/model_metrics.csv`：模型1/2/3 的 C-index、AUC(3y proxy)、NRI、IDI
- `outputs/cox_model*_summary.csv`：Cox 回归系数及显著性
- `outputs/segment_risk_map.csv`：17 段风险差异表（事件组 - 非事件组）
- `outputs/figures/km_rdiast_median.png`：`Rdiast` 的 KM 曲线
- `outputs/figures/km_deltatheta_median.png`：`DeltaTheta` 的 KM 曲线
- `outputs/figures/segment_risk_map.png`：17 段风险定位柱状图
- `outputs/figures/model2_coef_rank.png`：几何模型系数排序图

## 9) 学术报告补充统计图（不重跑模型）

在既定实验目录（如冠军输出 `outputs_clinical_endpoint_n320_bs400/`）已有 CSV 的前提下，可由脚本写入 `figures_extra/`（图 7–12：热力图‑柱复合、Sens–Spec 诊断散点、KDE+Q–Q 等）；若存在 `outer_test_predictions_bundle.csv`，仍会同步刷新 `figures/roc_endpoint_multimodel_test.png`（详见 `report_academic_integrated.md`）：

```powershell
python "e:\本基中间结果\make_report_extra_figures.py" --output-dir "e:\本基中间结果\outputs_clinical_endpoint_n320_bs400"
```

## 10) `report_academic_integrated.md` 导出 Word

依赖：`python-docx`（不要求安装 Pandoc）。`--base-dir` 应与文中图片相对路径所在根目录一致（一般为项目根）。

```powershell
python "e:\本基中间结果\export_md_to_docx.py" "e:\本基中间结果\report_academic_integrated.md" -o "e:\本基中间结果\report_academic_integrated.docx" --base-dir "e:\本基中间结果"
```

## 8) 说明

- 当前实现按你的数据先完成“可运行链路”：
  - 仅使用 `seg==2` 心肌环
  - 17 节段不同步指数（Hilbert 相位）
  - 嵌套 Cox 模型比较
- 三个主脚本均支持“自动最佳运行”：
  - `run_hcm_pipeline.py`：支持 `--cpu-jobs` 控制配准/预处理线程
  - `deep_experiment_60.py`：支持 `--compute auto|cpu|gpu` 与 `--cpu-jobs`
  - `deep_experiment_full_gpu.py`：支持 `--compute auto|cpu|gpu` 与 `--cpu-jobs`
- 使用 `--use-registration` 时，脚本会用 `SimpleITK` 位移场计算 `log(1+|u|)` 速度代理特征；这比纯几何版更接近“切空间轨迹”思路。
- 若后续需要严格复现 `SyN + Log-Euclidean`（ANTs/LDDMM 级），可将当前位移场模块替换为 ANTs 形变场及对数映射实现。
