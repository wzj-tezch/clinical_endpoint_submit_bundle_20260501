# 提交包说明（心脏 MR 几何 + endpoint 临床结局实验）

生成日期（打包日）：**20260501**

## 1. 目录结构

| 路径 | 内容 |
|------|------|
| `01_code/` | 主实验与报告制图、导出脚本（见下文清单） |
| `outputs_clinical_endpoint_n320_bs400/` | **主分析择优冠军**：CSV、协议、`figures/`、`figures_extra/`、`outer_test_predictions_bundle.csv` |
| `outputs_clinical_endpoint_full_bs400/` | **full-cohort** 敏感性（报告 §4.9）同源输出 |
| `outputs_clinical_endpoint_champion_n320_iso/` | **isotonic 校准副业**（报告 §4.10，可选对照） |
| `report_champion_bundle_20260501_1338/` | 冠军批次归档快照（与主冠军目录基本一致，便于单行递交） |
| `report_academic_integrated.md` | 综合学术报告（Markdown）；文中图片路径相对**本目录根**。 |
| `report_champion_selection.json` | 择优准则与选用的冠军目录记录 |
| `cohort_scout_summary.json` | 队列勘测摘要 |
| `requirements.txt` | Python 依赖 |
| `README.md` | 原项目说明（含环境与通用管线） |

## 2. 代码清单（`01_code/`）

- `deep_experiment_full_gpu.py`：端到端入口（读取影像/标签、外层划分、Cox / 树 / MLP / 融合、`figures/`）。
- `make_report_extra_figures.py`：在已有 CSV 上重绘 **`figures/`** 与 **`figures_extra/`**（不重训模型）。
- `report_figure_utils.py`：出版风格绘图与 DCA / 校准等。
- `export_md_to_docx.py`：Markdown → Word（需 `python-docx`）。
- `pick_report_champion.py`：按综合评分选冠军目录并归档。
- `scout_cohort_stats.py`：`cohort_scout_summary.json` 等勘测统计。
- `assemble_submission_bundle.py`：生成本提交包脚本（可再生成本目录）。

默认命令行数据中 **`--data-root` / `--label-xlsx`** 指向本机路径；**换机复现时**请将二者改为你环境下的标签表与分割数据根目录，或直接使用本包已附的 **`features_with_labels_full.csv`、`outer_*` 等中间结果**复核指标与制图。

## 3. 快速复现（不重训）

在**本目录根**打开终端（使相对路径生效）：

```powershell
pip install -r requirements.txt

python ".\01_code\make_report_extra_figures.py" --output-dir ".\outputs_clinical_endpoint_n320_bs400"
python ".\01_code\make_report_extra_figures.py" --output-dir ".\outputs_clinical_endpoint_n320_bs400" ^
  --compare-endpoint-csv ".\outputs_clinical_endpoint_full_bs400\metrics_endpoint_binary_test.csv"

python ".\01_code\export_md_to_docx.py" ".\report_academic_integrated.md" --base-dir "." -o ".\report_academic_integrated.docx"
```

`--base-dir` 必须为**本提交包根目录**，否则文中的 `outputs_*` 插图无法写入 Word。

## 4. 完整重跑主实验（需自备原始影像与 Excel）

需在可访问 **`--data-root`**（内含与脚本一致的 seg / 数据结构）及 **`survival_clean(1).xlsx` 同类标签表** 的机器上执行，例如：

```powershell
python ".\01_code\deep_experiment_full_gpu.py" --min-samples 320 --bootstrap 400 --compute auto ^
  --output-dir ".\outputs_clinical_endpoint_n320_bs400_retry" ^
  --data-root "《你的影像与特征根》" --label-xlsx "《你的标签 Excel 路径》"
```

本提交包**不**附带 `E:\可用数据` 级原始体量数据；审评以随包 **`outputs_*` CSV/图 + 源码**为准。

---
*可由 `python assemble_submission_bundle.py` 随时重新生成目录。删除旧包时注意勿误删仅此一份的输出。*
