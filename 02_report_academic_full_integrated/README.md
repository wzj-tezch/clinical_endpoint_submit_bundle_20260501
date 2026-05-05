# 02_report_academic_full_integrated

本目录为 **博士学位论文体例整合稿**（全文 Markdown + 第二章流程示意 PNG），用于与仓库 [clinical_endpoint_submit_bundle_20260501](https://github.com/wzj-tezch/clinical_endpoint_submit_bundle_20260501) 一并归档或提交。

## 内容

| 路径 | 说明 |
|------|------|
| `report_academic_full_integrated.md` | 整合正文（第一～八章、参考文献、附录） |
| `report_assets/flowcharts/*.png` | 短轴分割、预后评估流程示意（栅格图） |
| `report_assets/flowcharts/*.mmd` | 同上流程的 Mermaid 源（可选） |

## 插图路径约定

正文中的相对链接假定：**本目录位于该 Git 仓库根目录下的 `02_report_academic_full_integrated/`**，且仓库根目录已包含（与现网仓库一致）：

- `outputs_clinical_endpoint_n320_bs400/` 等输出目录  
- 若需完整显示中期/合作者插图，请在仓库根保持 `report_assets/midterm/`、`report_assets/zhao/`（可从本地主工作区同步）

第五章临床主图默认引用 `../outputs_clinical_endpoint_n320_bs400/...`。

## 本地更新本目录

在本地主工作区（生成该整合稿的工程根目录）执行：

```powershell
python tools/prepare_github_02_bundle.py
```

会先根据当前 `report_academic_full_integrated.md` 与 `report_assets/flowcharts` 覆盖刷新本文件夹。

## 推送到 GitHub

见仓库根目录 `GITHUB_PUSH_INSTRUCTIONS.md`。
