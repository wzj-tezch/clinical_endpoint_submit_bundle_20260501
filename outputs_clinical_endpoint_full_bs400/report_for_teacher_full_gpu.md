# 全量GPU深化实验终版报告

## 数据与协议
- 时间单位：天；endpoint作为事件标签（1=事件发生，0=未发生）。
- 评估时间窗：1y=365天，2y=730天，3y=1095天。
- 选入样本：771；总事件：58；3年事件：58。
- **树/MLP/融合**的监督标签为 `endpoint` 二分类（非 3 年窗代理）；内层 CV 优化目标：`0.4*AUC+0.3*BalancedAcc+0.3*F1`（概率阈值 0.5）。
- 评估口径：固定外层测试集 + 内层嵌套调参；同时报告原始 C-index、方向校正 C-index，以及 Cox 与时间依赖 AUC。
- OOF 上选定的融合阈值（Balanced Accuracy 主规则）：**0.147068**（详见 `clinical_thresholds_*.csv`）。
- 计算策略：requested=auto, selected=gpu, cpu_jobs=19。
- GPU状态：torch.cuda=True，xgboost可用=True。

- 树模型后端：xgboost（device=cuda）。
- MLP设备：cuda。

## Cox / 生存视角（固定外层测试）
```
               model  cindex_raw  cindex_adjusted  auc_365d  auc_730d  auc_1095d
     model1_base_cox    0.741042         0.741042  0.690637  0.690637   0.690637
     model2_geom_cox    0.662052         0.662052  0.608989  0.608989   0.608989
     model3_full_cox    0.735342         0.735342  0.670037  0.670037   0.670037
    model4_tree_auto    0.673453         0.673453  0.672285  0.672285   0.672285
     model5_mlp_auto    0.587948         0.587948  0.578652  0.578652   0.578652
model6_ensemble_auto    0.587948         0.587948  0.578652  0.578652   0.578652
```

## endpoint 二分类（固定外层测试 — 概率与阈值）
### 阈值（OOF→外测复述）与临床指标
```
 threshold  balanced_accuracy       f1       mcc  sensitivity  specificity      ppv      npv   lr_pos   lr_neg  youden_j           rule_name_train_oob  auc_probability
  0.147068           0.527528 0.148760  0.029624     0.600000     0.455056 0.084906 0.931034 1.101031 0.879012  0.055056         max_balanced_accuracy         0.578652
  0.147068           0.527528 0.148760  0.029624     0.600000     0.455056 0.084906 0.931034 1.101031 0.879012  0.055056                  max_youden_j         0.578652
  0.028840           0.489513 0.139785 -0.017669     0.866667     0.112360 0.076023 0.909091 0.976371 1.186667 -0.020974 conservative_high_specificity         0.578652
  0.147068           0.527528 0.148760  0.029624     0.600000     0.455056 0.084906 0.931034 1.101031 0.879012  0.055056   aggressive_high_sensitivity         0.578652
```

### 树 / MLP / 融合的概率型指标（含 Brier/ECE/logit校准）
```
                   model      auc  balanced_accuracy_050   f1_050  mcc_050    brier      ece  calibration_slope_logit  calibration_intercept_logit
    model4_tree_endpoint 0.672285               0.533333 0.125000 0.248607 0.062894 0.046966                 0.441758                    -0.663612
     model5_mlp_endpoint 0.578652               0.576779 0.192308 0.104440 0.144998 0.205115                 0.126075                    -2.273181
model6_ensemble_endpoint 0.578652               0.576779 0.192308 0.104440 0.144998 0.205115                 0.126075                    -2.273181
```

## 关键区间结果（bootstrap）
含 `*_auc_endpoint`、按 OOF 主阈值裁切的 `model6_ensemble_*_thr_oob`，以及 Cox/NRI。

```
                       metric      mean     ci2.5    ci97.5
     model1_base_cox_auc_1095  0.688394  0.518916  0.843184
      model1_base_cox_auc_365  0.688394  0.518916  0.843184
      model1_base_cox_auc_730  0.688394  0.518916  0.843184
 model1_base_cox_auc_endpoint  0.688394  0.518916  0.843184
         model1_base_cox_cadj  0.743851  0.571339  0.876086
         model1_base_cox_craw  0.743776  0.571339  0.876086
     model2_geom_cox_auc_1095  0.603129  0.402562  0.814894
      model2_geom_cox_auc_365  0.603129  0.402562  0.814894
      model2_geom_cox_auc_730  0.603129  0.402562  0.814894
 model2_geom_cox_auc_endpoint  0.603129  0.402562  0.814894
         model2_geom_cox_cadj  0.664486  0.510133  0.837855
         model2_geom_cox_craw  0.656658  0.441022  0.837855
       model2_geom_cox_idi_ep  0.346222 -0.337746  1.143570
       model2_geom_cox_nri_ep  0.400933 -0.146151  0.957343
     model3_full_cox_auc_1095  0.662829  0.472506  0.819240
      model3_full_cox_auc_365  0.662829  0.472506  0.819240
      model3_full_cox_auc_730  0.662829  0.472506  0.819240
 model3_full_cox_auc_endpoint  0.662829  0.472506  0.819240
         model3_full_cox_cadj  0.733015  0.542051  0.902116
         model3_full_cox_craw  0.731293  0.522820  0.902116
       model3_full_cox_idi_ep  1.178003 -0.049766  3.020878
       model3_full_cox_nri_ep  0.132369 -0.425590  0.586641
    model4_tree_auto_auc_1095  0.663059  0.457969  0.846722
     model4_tree_auto_auc_365  0.663059  0.457969  0.846722
     model4_tree_auto_auc_730  0.663059  0.457969  0.846722
model4_tree_auto_auc_endpoint  0.663059  0.457969  0.846722
        model4_tree_auto_cadj  0.673730  0.513147  0.865614
        model4_tree_auto_craw  0.665043  0.434532  0.865614
      model4_tree_auto_idi_ep -0.276539 -0.557319 -0.037925
      model4_tree_auto_nri_ep  0.000000  0.000000  0.000000
```

## 多seed复跑稳定性（固定外测）
含 endpoint AUC（`auc_endpoint`）与主阈值 Sens/Spec/BA/F1（`model6_*_thr_oob`）。

```
                                   metric     mean    ci2.5   ci97.5
                     model4_tree_auc_1095 0.677778 0.670487 0.684007
                 model4_tree_auc_endpoint 0.677778 0.670487 0.684007
                         model4_tree_cadj 0.666395 0.661523 0.670806
                         model4_tree_craw 0.666395 0.661523 0.670806
                      model5_mlp_auc_1095 0.603745 0.552865 0.642528
                  model5_mlp_auc_endpoint 0.603745 0.552865 0.642528
                          model5_mlp_cadj 0.609663 0.550570 0.651914
                          model5_mlp_craw 0.609663 0.550570 0.651914
                        model6_ba_thr_oob 0.517790 0.507734 0.526414
                      model6_ens_auc_1095 0.603745 0.552865 0.642528
                          model6_ens_cadj 0.609663 0.550570 0.651914
                          model6_ens_craw 0.609663 0.550570 0.651914
             model6_ensemble_auc_endpoint 0.603745 0.552865 0.642528
                        model6_f1_thr_oob 0.145554 0.142921 0.149384
                       model6_mcc_thr_oob 0.020062 0.009107 0.030342
                      model6_sens_thr_oob 0.666667 0.543333 0.733333
                      model6_spec_thr_oob 0.368914 0.282865 0.496348
threshold_used_from_main_oob_balanced_acc 0.147068 0.147068 0.147068
```

## 增益指标
- NRI/IDI（二元 **endpoint** 标签）已写入 `nri_idi_fixed_test.csv`。

## 解释性与定位
- seg07_amp: 186 次 / 200 次（93.00%）
- seg16_amp: 120 次 / 200 次（60.00%）
- seg10_amp: 110 次 / 200 次（55.00%）
- seg04_amp: 96 次 / 200 次（48.00%）
- seg14_amp: 76 次 / 200 次（38.00%）

## 图表与交付清单
- `figures/time_dependent_auc_curve.png`
- `figures/calibration_curve_endpoint_ensemble.png` / `figures/decision_curve_endpoint_ensemble.png`
- `figures/bullseye_standard_diff.png`
- `figures/bullseye_top5_stability.png`
- `metrics_endpoint_binary_test.csv` · `clinical_thresholds_test_performance.csv`
- `protocol_endpoint.json` · `report_clinical_endpoint.md`
- `seed_repeat_metrics.csv` / `bootstrap_metrics.csv` 等同 seed 可比文件。
